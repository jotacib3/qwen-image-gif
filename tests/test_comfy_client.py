"""Unit tests for the ComfyUI API client and workflow manipulation."""

import io
from pathlib import Path
from unittest.mock import patch

import pytest
import requests
from PIL import Image

from src.comfy_client import ComfyUIClient, ComfyUIError


@pytest.fixture
def dummy_workflow_path(tmp_path: Path) -> Path:
    """Creates a minimal mock workflow_api.json file for isolated tests."""
    wf_path = tmp_path / "mock_workflow.json"
    wf_path.write_text(
        """{
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen.safetensors"}},
        "4": {"class_type": "LoadImage", "inputs": {"image": "old.png"}},
        "5": {"class_type": "TextEncodeQwenImage21", "inputs": {"prompt": "x", "resolution": 1024}},
        "6": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 20, "cfg": 1.0}}
    }"""
    )
    return wf_path


def test_missing_workflow_raises_filenotfound() -> None:
    """Ensure missing workflow file raises a clear FileNotFoundError."""
    client = ComfyUIClient(workflow_path="non_existent_file.json")
    with pytest.raises(FileNotFoundError):
        _ = client.workflow_template


def test_build_prompt_workflow_injection(dummy_workflow_path: Path) -> None:
    """Parameters and the <image1> tag are injected into the nodes found by class_type."""
    client = ComfyUIClient(workflow_path=dummy_workflow_path)
    wf = client.build_prompt_workflow(
        image_name="test_reference.png",
        edit_prompt="Raise both arms in victory",
        steps=30,
        seed=12345,
        resolution=768,
    )

    assert wf["4"]["inputs"]["image"] == "test_reference.png"
    assert wf["5"]["inputs"]["prompt"] == "<image1> Raise both arms in victory"
    assert wf["5"]["inputs"]["resolution"] == 768
    assert wf["6"]["inputs"]["seed"] == 12345
    assert wf["6"]["inputs"]["steps"] == 30
    assert wf["6"]["inputs"]["cfg"] == 1.0  # untouched template value


def test_build_prompt_workflow_keeps_existing_image_tag(dummy_workflow_path: Path) -> None:
    """A prompt that already references <image1> is not prefixed twice."""
    client = ComfyUIClient(workflow_path=dummy_workflow_path)
    wf = client.build_prompt_workflow(image_name="a.png", edit_prompt="Make <image1> wave")
    assert wf["5"]["inputs"]["prompt"] == "Make <image1> wave"


def test_build_prompt_workflow_missing_node_raises(tmp_path: Path) -> None:
    """A graph without the Qwen-Image-2.1 encode node fails with an actionable error."""
    wf_path = tmp_path / "old_workflow.json"
    wf_path.write_text('{"2": {"class_type": "LoadImage", "inputs": {"image": "x.png"}}}')
    client = ComfyUIClient(workflow_path=wf_path)
    with pytest.raises(ComfyUIError, match="TextEncodeQwenImage21"):
        client.build_prompt_workflow(image_name="a.png", edit_prompt="smile")


def test_bundled_workflow_is_valid() -> None:
    """The workflow shipped in the repo contains every node the client needs."""
    wf = ComfyUIClient().build_prompt_workflow(image_name="a.png", edit_prompt="smile", seed=1)
    assert any(n["class_type"] == "TextEncodeQwenImage21" for n in wf.values())


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
                "8": {"images": [{"filename": "out_0001.png", "subfolder": "", "type": "output"}]}
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


def test_wait_for_completion_surfaces_execution_error(dummy_workflow_path: Path) -> None:
    """A failed execution raises immediately instead of polling until the timeout."""
    client = ComfyUIClient(base_url="http://fake-host:8188", workflow_path=dummy_workflow_path)
    history = {
        "uuid-err": {
            "status": {
                "status_str": "error",
                "messages": [
                    [
                        "execution_error",
                        {"node_type": "UNETLoader", "exception_message": "file not found"},
                    ]
                ],
            },
            "outputs": {},
        }
    }
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = history
        with pytest.raises(ComfyUIError, match="UNETLoader: file not found"):
            client.wait_for_completion(prompt_id="uuid-err")
