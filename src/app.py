"""Interactive Gradio Web Application for Qwen-Image-2.1 Image Editing and GIF Creation."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import gradio as gr
from PIL import Image

from src.comfy_client import ComfyUIClient, ComfyUIError
from src.gif_engine import create_animated_gif

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_app() -> gr.Blocks:
    """Build and return the Gradio Blocks UI application."""
    default_comfy_url = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")

    custom_css = """
    .main-container { max-width: 1200px; margin: 0 auto; }
    .status-badge { font-weight: bold; padding: 4px 8px; border-radius: 4px; }
    """

    with gr.Blocks(title="Qwen Image 2.1 - Editor & GIF Creator", css=custom_css) as demo:
        # State stores
        # invariant_base_image: Always points to the original reference frame
        # accepted_frames: List of approved frames to be compiled into the GIF
        state_base_image = gr.State(None)
        state_candidate_image = gr.State(None)
        state_frames = gr.State([])

        gr.Markdown(
            """
            # 🎬 Qwen Image 2.1: Image Editor & GIF Creator
            ### Unified Multi-Frame Animation with Single-Reference Invariant Consistency
            
            This application implements the tutorial architecture for **Qwen-Image-2.1**. 
            To prevent cumulative visual decay across animation frames, **every modification references 
            the fixed base image (`<image1>`)** rather than recursively editing previously generated frames.
            """
        )

        with gr.Accordion("⚙️ Backend Configuration & Server Status", open=False):
            with gr.Row():
                comfy_url_input = gr.Textbox(
                    value=default_comfy_url,
                    label="ComfyUI Server Endpoint",
                    placeholder="http://127.0.0.1:8188 or https://<your-colab-tunnel>",
                    scale=3,
                )
                check_btn = gr.Button("Test Connection", scale=1)
                status_output = gr.Label(label="Backend Status", scale=1)

            def check_backend_status(url: str):
                client = ComfyUIClient(base_url=url)
                if client.check_connection():
                    return "🟢 Online (ComfyUI Reachable)"
                return "🔴 Offline (Cannot connect to endpoint)"

            check_btn.click(fn=check_backend_status, inputs=[comfy_url_input], outputs=[status_output])

        with gr.Row():
            # Left Column: Invariant Source & Controls
            with gr.Column(scale=1):
                gr.Markdown("### 1️⃣ Source Reference Image")
                input_base_image = gr.Image(
                    type="pil",
                    label="Base Reference Frame (<image1>)",
                    sources=["upload", "clipboard"],
                )

                gr.Markdown("### 2️⃣ Edit Instruction")
                prompt_input = gr.Textbox(
                    label="Natural Language Edit Instruction",
                    placeholder="e.g. Turn head slightly to the right, wink with left eye...",
                    lines=3,
                )

                with gr.Accordion("Sampling Hyperparameters", open=False):
                    denoise_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.75,
                        step=0.05,
                        label="Denoise Strength (Edit intensity)",
                    )
                    steps_slider = gr.Slider(
                        minimum=10,
                        maximum=50,
                        value=25,
                        step=1,
                        label="Sampling Steps",
                    )
                    seed_input = gr.Number(
                        label="Seed (-1 for random)",
                        value=-1,
                        precision=0,
                    )

                generate_btn = gr.Button("🚀 Generate Frame Variant", variant="primary")

            # Right Column: Candidate Inspection & Staging
            with gr.Column(scale=1):
                gr.Markdown("### 3️⃣ Candidate Output Preview")
                preview_output = gr.Image(
                    type="pil",
                    label="Current Candidate Frame",
                    interactive=False,
                )

                with gr.Row():
                    accept_btn = gr.Button("✅ Accept & Add to Sequence", variant="secondary")
                    discard_btn = gr.Button("🗑️ Discard", variant="stop")

        gr.Markdown("---")

        # Bottom Section: Timeline & GIF Compiler
        gr.Markdown("### 4️⃣ Animation Sequence & Export")
        with gr.Row():
            gallery = gr.Gallery(
                label="Accepted Animation Timeline",
                columns=6,
                rows=2,
                height="auto",
                object_fit="contain",
            )

        with gr.Row():
            with gr.Column(scale=1):
                fps_slider = gr.Slider(
                    minimum=1,
                    maximum=30,
                    value=8,
                    step=1,
                    label="Playback Speed (FPS)",
                )
                ping_pong_cb = gr.Checkbox(
                    value=False,
                    label="Ping-Pong Loop (Back & Forth bounce)",
                )
                max_dim_slider = gr.Slider(
                    minimum=256,
                    maximum=1024,
                    value=512,
                    step=64,
                    label="Max Dimension (Resize for lightweight GIF)",
                )
                with gr.Row():
                    compile_btn = gr.Button("📦 Compile Animated GIF", variant="primary")
                    clear_btn = gr.Button("🔄 Reset Timeline", variant="secondary")

            with gr.Column(scale=1):
                gif_result = gr.Image(label="Final Animated GIF Output", interactive=False)
                download_file = gr.File(label="Download GIF File")

        # --- Event Handlers & Business Logic ---

        def on_image_upload(img: Image.Image | None):
            """Sync uploaded image into state."""
            return img

        input_base_image.change(fn=on_image_upload, inputs=[input_base_image], outputs=[state_base_image])

        def run_generation(
            base_img: Image.Image | None,
            prompt: str,
            denoise: float,
            steps: int,
            seed: int,
            url: str,
        ):
            if base_img is None:
                raise gr.Error("Please upload a base reference image before generating variants.")

            if not prompt.strip():
                raise gr.Error("Please provide an edit instruction prompt.")

            client = ComfyUIClient(base_url=url)
            if not client.check_connection():
                raise gr.Error(
                    f"ComfyUI is not reachable at '{url}'. "
                    "Make sure ComfyUI is running locally or provide a valid Colab tunnel URL."
                )

            actual_seed = None if seed == -1 else int(seed)

            try:
                variant = client.generate_variant(
                    base_image=base_img,
                    edit_prompt=prompt,
                    denoise=denoise,
                    steps=steps,
                    seed=actual_seed,
                )
                return variant, variant
            except ComfyUIError as e:
                raise gr.Error(f"ComfyUI inference error: {e}") from e
            except Exception as e:
                raise gr.Error(f"Unexpected generation failure: {e}") from e

        generate_btn.click(
            fn=run_generation,
            inputs=[
                state_base_image,
                prompt_input,
                denoise_slider,
                steps_slider,
                seed_input,
                comfy_url_input,
            ],
            outputs=[preview_output, state_candidate_image],
        )

        def accept_candidate(candidate: Image.Image | None, current_frames: list[Image.Image]):
            if candidate is None:
                gr.Warning("No candidate frame to accept.")
                return current_frames, current_frames, None

            updated_list = list(current_frames)
            updated_list.append(candidate)
            gr.Info(f"Frame accepted! Total sequence frames: {len(updated_list)}")
            return updated_list, updated_list, None

        accept_btn.click(
            fn=accept_candidate,
            inputs=[state_candidate_image, state_frames],
            outputs=[state_frames, gallery, preview_output],
        )

        def discard_candidate():
            return None

        discard_btn.click(fn=discard_candidate, outputs=[preview_output])

        def compile_gif(
            frames: list[Image.Image],
            fps: int,
            ping_pong: bool,
            max_dim: int,
        ):
            if not frames:
                raise gr.Error("No frames in timeline to compile. Generate and accept frames first.")

            output_file = Path("outputs") / "qwen_animation.gif"
            try:
                saved_path = create_animated_gif(
                    frames=frames,
                    output_path=output_file,
                    fps=int(fps),
                    ping_pong=ping_pong,
                    max_dimension=int(max_dim),
                )
                return str(saved_path), str(saved_path)
            except Exception as e:
                raise gr.Error(f"Failed to assemble GIF: {e}") from e

        compile_btn.click(
            fn=compile_gif,
            inputs=[state_frames, fps_slider, ping_pong_cb, max_dim_slider],
            outputs=[gif_result, download_file],
        )

        def reset_timeline():
            return [], []

        clear_btn.click(fn=reset_timeline, outputs=[state_frames, gallery])

    return demo


def main() -> None:
    """Entry point for running the web application."""
    port = int(os.getenv("PORT", "7860"))
    host = os.getenv("HOST", "0.0.0.0")

    app = build_app()
    app.launch(server_name=host, server_port=port, share=False)


if __name__ == "__main__":
    main()
