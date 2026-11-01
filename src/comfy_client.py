"""ComfyUI API Client for Qwen-Image-2.1 generation and image editing."""

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


class ComfyUIError(RuntimeError):
    """Raised when communication with ComfyUI API fails."""


class ComfyUIClient:
    """Client for executing Qwen-Image-2.1 workflows on a ComfyUI instance."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8188",
        workflow_path: str | Path = "workflow_api.json",
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

    def build_prompt_workflow(
        self,
        image_name: str,
        edit_prompt: str,
        denoise: float = 0.75,
        steps: int = 25,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Inject runtime arguments into the Qwen-Image-2.1 workflow template.

        Args:
            image_name: The filename of the uploaded image.
            edit_prompt: The natural language editing instruction.
            denoise: Denoising strength for KSampler.
            steps: Number of sampling steps.
            seed: Random seed (if None, a random integer is generated).

        Returns:
            Configured workflow dictionary ready for queueing.
        """
        workflow = copy.deepcopy(self.workflow_template)

        # Ensure seed is set
        if seed is None:
            seed = random.randint(1, 10**14)

        # 1. Inject reference image filename in LoadImage node (ID: "2")
        if "2" in workflow and "inputs" in workflow["2"]:
            workflow["2"]["inputs"]["image"] = image_name

        # 2. Inject positive instruction prompt in CLIPTextEncode node (ID: "3")
        # Format instruction with <image1> tag as required by Qwen-Image-2.1
        formatted_prompt = edit_prompt.strip()
        if not formatted_prompt.startswith("<image1>"):
            formatted_prompt = f"<image1> {formatted_prompt}"

        if "3" in workflow and "inputs" in workflow["3"]:
            workflow["3"]["inputs"]["text"] = formatted_prompt

        # 3. Inject sampler parameters in KSampler node (ID: "6")
        if "6" in workflow and "inputs" in workflow["6"]:
            workflow["6"]["inputs"]["seed"] = seed
            workflow["6"]["inputs"]["steps"] = steps
            workflow["6"]["inputs"]["denoise"] = denoise

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
        denoise: float = 0.75,
        steps: int = 25,
        seed: int | None = None,
    ) -> Image.Image:
        """High-level method to run full single-reference editing pipeline.

        Args:
            base_image: Original fixed reference image.
            edit_prompt: Natural language modification instructions.
            denoise: Denoise factor (0.1 = subtle changes, 0.9 = strong changes).
            steps: Diffusion sampling steps.
            seed: Sampling seed.

        Returns:
            Generated modified PIL Image.
        """
        logger.info("Uploading reference image to ComfyUI...")
        image_name = self.upload_image(base_image)

        logger.info("Building workflow payload with prompt: %s", edit_prompt)
        workflow = self.build_prompt_workflow(
            image_name=image_name,
            edit_prompt=edit_prompt,
            denoise=denoise,
            steps=steps,
            seed=seed,
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
