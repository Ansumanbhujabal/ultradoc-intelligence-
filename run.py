"""Single entrypoint — mounts Gradio on FastAPI, single port."""

import uvicorn
import gradio as gr
from app.main import app as fastapi_app
from app.config import settings
from ui.gradio_app import build_ui


def main():
    demo = build_ui()
    app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")
    uvicorn.run(app, host="0.0.0.0", port=settings.api_port, log_level=settings.log_level.lower())


if __name__ == "__main__":
    main()
