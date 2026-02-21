"""Run the LLM Manager web app. Uses port and config from config.yaml / env."""

import uvicorn

from app.config import load_config

if __name__ == "__main__":
    config = load_config()
    port = config["port"]
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
