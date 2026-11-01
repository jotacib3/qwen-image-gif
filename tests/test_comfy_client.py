"""Unit tests for the ComfyUI API client and workflow manipulation."""

import io
from pathlib import Path
from unittest.mock import patch

import pytest
import requests
from PIL import Image

from src.comfy_client import ComfyUIClient


@pytest.fixture
def dummy_workflow_path(tmp_path: Path) -> Path:
    """Creates a minimal mock workflow_api.json file for isolated tests."""
    wf_path = tmp_path / "mock_workflow.json"
    wf_path.write_text(
        """{
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "qwen.safetensors"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "old.png"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "default prompt"}},
        "6": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 20, "denoise": 0.7}}
    }"""
    )
    return wf_path


def test_missing_workflow_raises_filenotfound() -> None:
    """Ensure missing workflow file raises a clear FileNotFoundError."""
    client = ComfyUIClient(workflow_path="non_existent_file.json")
    with pytest.raises(FileNotFoundError):
        _ = client.workflow_template


def test_build_prompt_workflow_injection(dummy_workflow_path: Path) -> None:
    """Verify parameters and <image1> tag are properly injected into the workflow graph."""
    client = ComfyUIClient(workflow_path=dummy_workflow_path)
    wf = client.build_prompt_workflow(
        image_name="test_reference.png",
        edit_prompt="Raise both arms in victory",
        denoise=0.85,
        steps=30,
        seed=12345,
    )

    # Node 2: Image name
    assert wf["2"]["inputs"]["image"] == "test_reference.png"

    # Node 3: Prompt prepended with <image1>
    assert wf["3"]["inputs"]["text"] == "<image1> Raise both arms in victory"

    # Node 6: Sampler parameters
    assert wf["6"]["inputs"]["seed"] == 12345
    assert wf["6"]["inputs"]["steps"] == 30
    assert wf["6"]["inputs"]["denoise"] == 0.85


def test_check_connection_online(dummy_workflow_path: Path) -> None:
    """Verify check_connection returns True on HTTP 200."""
    client = ComfyUIClient(base_url="http://fake-host:8188", workflow_path=dummy_workflow_path)
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        assert client.check_connection() is True


def test_check_connection_offline(dummy_workflow_path: Path) -> None:
    """Verify check_connection returns False on failure."""
    client = ComfyUIClient(base_url="http://fake-host:8188", workflow_path=dummy_workflow_path)
    with patch("requests.get", side_effect=requests.RequestException("Connection refused")):
        assert client.check_connection() is False


def test_upload_image_success(dummy_workflow_path: Path) -> None:
    """Verify image upload payload handling."""
    client = ComfyUIClient(base_url="http://fake-host:8188", workflow_path=dummy_workflow_path)
    test_img = Image.new("RGB", (32, 32), color=(255, 0, 0))

    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"name": "uploaded_base.png"}

        result_name = client.upload_image(test_img, filename="base.png")
        assert result_name == "uploaded_base.png"
        assert mock_post.called


def test_queue_prompt_success(dummy_workflow_path: Path) -> None:
    """Verify prompt queueing returns prompt_id."""
    client = ComfyUIClient(base_url="http://fake-host:8188", workflow_path=dummy_workflow_path)

    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"prompt_id": "uuid-1234-abcd"}

        prompt_id = client.queue_prompt({"test": "graph"})
        assert prompt_id == "uuid-1234-abcd"


def test_wait_for_completion(dummy_workflow_path: Path) -> None:
    """Verify polling until prompt ID output is received."""
    client = ComfyUIClient(base_url="http://fake-host:8188", workflow_path=dummy_workflow_path)

    mock_history_response = {
        "uuid-1234": {
            "outputs": {
                "8": {
                    "images": [
                        {"filename": "out_0001.png", "subfolder": "", "type": "output"}
                    ]
                }
            }
        }
    }

    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_history_response

        outputs = client.wait_for_completion(prompt_id="uuid-1234")
        assert len(outputs) == 1
        assert outputs[0]["filename"] == "out_0001.png"


def test_fetch_image(dummy_workflow_path: Path) -> None:
    """Verify downloading output image content as PIL Image."""
    client = ComfyUIClient(base_url="http://fake-host:8188", workflow_path=dummy_workflow_path)

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(0, 255, 0)).save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = raw_bytes

        img = client.fetch_image(filename="out_0001.png")
        assert isinstance(img, Image.Image)
        assert img.size == (64, 64)
