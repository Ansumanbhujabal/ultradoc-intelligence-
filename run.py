"""Single entrypoint — starts FastAPI + Gradio."""

import threading
import uvicorn
from app.main import app as fastapi_app
from app.config import settings
from ui.gradio_app import build_ui


def start_fastapi():
    uvicorn.run(fastapi_app, host="0.0.0.0", port=settings.api_port, log_level="info")


def start_gradio():
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=settings.gradio_port, share=False)


if __name__ == "__main__":
    api_thread = threading.Thread(target=start_fastapi, daemon=True)
    api_thread.start()
    start_gradio()
