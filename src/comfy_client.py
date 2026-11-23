"""ComfyUI REST client for Qwen-Image-2.1 image editing."""

from __future__ import annotations

import copy
import io
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image

logger = logging.getLogger(__name__)

# workflow_api.json lives at the project root, next to the `src/` package.
DEFAULT_WORKFLOW_PATH = Path(__file__).resolve().parent.parent / "workflow_api.json"

# Node types of the official Qwen-Image-2.1 edit graph that receive runtime values.
IMAGE_NODE = "LoadImage"
ENCODE_NODE = "TextEncodeQwenImage21"
SAMPLER_NODE = "KSampler"


class ComfyUIError(RuntimeError):
    """Raised when communication with ComfyUI API fails."""


class ComfyUIClient:
    """Client for executing Qwen-Image-2.1 workflows on a ComfyUI instance."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8188",
        workflow_path: str | Path = DEFAULT_WORKFLOW_PATH,
        timeout: int = 180,
    ) -> None:
        """Initialize ComfyUI client.

        Args:
            base_url: Base URL where ComfyUI server is listening.
            workflow_path: Path to the exported API workflow JSON.
            timeout: Maximum seconds to wait for generation execution.
        """
        self.base_url = base_url.rstrip("/")
        self.workflow_path = Path(workflow_path)
        self.timeout = timeout
        self._workflow_template: dict[str, Any] | None = None

    @property
    def workflow_template(self) -> dict[str, Any]:
        """Loads and caches the workflow JSON template."""
        if self._workflow_template is None:
            if not self.workflow_path.exists():
                raise FileNotFoundError(
                    f"Workflow file '{self.workflow_path}' not found. "
                    "Ensure workflow_api.json exists in the project root."
                )
            with open(self.workflow_path, "r", encoding="utf-8") as f:
                self._workflow_template = json.load(f)
        return self._workflow_template

    def check_connection(self) -> bool:
        """Check whether the ComfyUI instance is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/system_stats", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def upload_image(self, image: Image.Image, filename: str = "base_reference.png") -> str:
        """Upload a PIL Image to ComfyUI's input directory.

        Args:
            image: PIL Image object.
            filename: Target file name on ComfyUI server.

        Returns:
            The uploaded image filename acknowledged by ComfyUI.
        """
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        files = {"image": (filename, buffer, "image/png")}
        data = {"overwrite": "true"}

        url = f"{self.base_url}/upload/image"
        try:
            resp = requests.post(url, files=files, data=data, timeout=30)
            resp.raise_for_status()
            result = resp.json()
            return result.get("name", filename)
        except requests.RequestException as e:
            raise ComfyUIError(f"Failed to upload reference image to ComfyUI: {e}") from e

    @staticmethod
    def _find_node(workflow: dict[str, Any], class_type: str) -> dict[str, Any]:
        """Return the first node of the given type.

        Nodes are located by `class_type` instead of hard-coded IDs, so any graph exported
        from ComfyUI with "Export (API)" works as long as it contains these node types.
        """
        for node in workflow.values():
            if isinstance(node, dict) and node.get("class_type") == class_type:
                return node
        raise ComfyUIError(
            f"Workflow has no '{class_type}' node. Export the official Qwen-Image-2.1 "
            "edit template from ComfyUI in API format (see README)."
        )

    def build_prompt_workflow(
        self,
        image_name: str,
        edit_prompt: str,
        steps: int = 25,
        seed: int | None = None,
        resolution: int | None = None,
    ) -> dict[str, Any]:
        """Inject runtime arguments into the Qwen-Image-2.1 edit workflow template.

        Args:
            image_name: Filename of the reference image already uploaded to ComfyUI.
            edit_prompt: Natural language edit instruction.
            steps: Sampling steps (the official pipeline uses ~40-50; 25 is a faster default).
            seed: Random seed (a random one is generated when None).
            resolution: Total pixel budget for the edit (e.g. 1024). Keeps the template value
                when None.

        Returns:
            Configured workflow dictionary ready to be queued.
        """
        workflow = copy.deepcopy(self.workflow_template)

        if seed is None:
            seed = random.randint(1, 2**48)

        # Reference image -> LoadImage
        self._find_node(workflow, IMAGE_NODE)["inputs"]["image"] = image_name

        # Instruction -> TextEncodeQwenImage21. Qwen-Image-2.1 refers to inputs as <image1>..<image10>.
        formatted_prompt = edit_prompt.strip()
        if "<image1>" not in formatted_prompt:
            formatted_prompt = f"<image1> {formatted_prompt}"
        encode_inputs = self._find_node(workflow, ENCODE_NODE)["inputs"]
        encode_inputs["prompt"] = formatted_prompt
        if resolution is not None:
            encode_inputs["resolution"] = int(resolution)

        # Sampler parameters -> KSampler
        sampler_inputs = self._find_node(workflow, SAMPLER_NODE)["inputs"]
        sampler_inputs["seed"] = seed
        sampler_inputs["steps"] = int(steps)

        return workflow

    def queue_prompt(self, workflow: dict[str, Any]) -> str:
        """Submit a prompt workflow to ComfyUI for execution.

        Args:
            workflow: Configured workflow dictionary.

        Returns:
            The prompt_id assigned to the generation job.
        """
        url = f"{self.base_url}/prompt"
        payload = {"prompt": workflow}

        try:
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code == 400:
                # ComfyUI validates the graph up front (missing weights, unknown nodes, ...)
                raise ComfyUIError(f"ComfyUI rejected the workflow: {resp.text[:500]}")
            resp.raise_for_status()
            data = resp.json()
            if "prompt_id" not in data:
                raise ComfyUIError(f"ComfyUI did not return prompt_id. Response: {data}")
            return data["prompt_id"]
        except requests.RequestException as e:
            raise ComfyUIError(f"Failed to queue prompt to ComfyUI: {e}") from e

    def wait_for_completion(
        self, prompt_id: str, poll_interval: float = 1.0
    ) -> list[dict[str, str]]:
        """Poll ComfyUI history until the requested prompt finishes.

        Args:
            prompt_id: ID of the queued prompt.
            poll_interval: Seconds between status polls.

        Returns:
            List of generated output image descriptors (filename, subfolder, type).
        """
        start_time = time.time()
        url = f"{self.base_url}/history/{prompt_id}"

        while time.time() - start_time < self.timeout:
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    history = resp.json()
                    if prompt_id in history:
                        item = history[prompt_id]
                        status = item.get("status", {})
                        if status.get("status_str") == "error":
                            raise ComfyUIError(
                                f"ComfyUI failed to execute prompt {prompt_id}: "
                                f"{self._describe_error(status)}"
                            )
                        outputs = item.get("outputs", {})
                        images = []
                        for node_output in outputs.values():
                            if "images" in node_output:
                                images.extend(node_output["images"])
                        if images:
                            return images
            except requests.RequestException:
                pass

            time.sleep(poll_interval)

        raise TimeoutError(
            f"Generation timed out after {self.timeout}s waiting for prompt {prompt_id}."
        )

    @staticmethod
    def _describe_error(status: dict[str, Any]) -> str:
        """Extract a readable message from a ComfyUI history `status` block."""
        for message_type, payload in status.get("messages", []):
            if message_type == "execution_error" and isinstance(payload, dict):
                node = payload.get("node_type", "unknown node")
                return f"{node}: {payload.get('exception_message', 'unknown error')}".strip()
        return "unknown error (check the ComfyUI console)"

    def fetch_image(
        self, filename: str, subfolder: str = "", folder_type: str = "output"
    ) -> Image.Image:
        """Download a generated image from ComfyUI's /view endpoint.

        Args:
            filename: Output image filename.
            subfolder: Subfolder within ComfyUI output dir.
            folder_type: Type of folder ('output', 'temp', or 'input').

        Returns:
            PIL Image object.
        """
        url = f"{self.base_url}/view"
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        except requests.RequestException as e:
            raise ComfyUIError(f"Failed to fetch image '{filename}' from ComfyUI: {e}") from e

    def generate_variant(
        self,
        base_image: Image.Image,
        edit_prompt: str,
        steps: int = 25,
        seed: int | None = None,
        resolution: int | None = None,
    ) -> Image.Image:
        """High-level method to run the full single-reference editing pipeline.

        Args:
            base_image: Original fixed reference image.
            edit_prompt: Natural language modification instructions.
            steps: Diffusion sampling steps.
            seed: Sampling seed.
            resolution: Total pixel budget for the edit.

        Returns:
            Generated modified PIL Image.
        """
        logger.info("Uploading reference image to ComfyUI...")
        image_name = self.upload_image(base_image)

        logger.info("Building workflow payload with prompt: %s", edit_prompt)
        workflow = self.build_prompt_workflow(
            image_name=image_name,
            edit_prompt=edit_prompt,
            steps=steps,
            seed=seed,
            resolution=resolution,
        )

        logger.info("Queueing generation task...")
        prompt_id = self.queue_prompt(workflow)

        logger.info("Waiting for execution of prompt ID: %s...", prompt_id)
        output_images = self.wait_for_completion(prompt_id)

        first_img = output_images[0]
        return self.fetch_image(
            filename=first_img["filename"],
            subfolder=first_img.get("subfolder", ""),
            folder_type=first_img.get("type", "output"),
        )
