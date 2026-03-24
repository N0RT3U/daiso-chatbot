from __future__ import annotations

import sys
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chatbot.app import create_app  # noqa: E402


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "run_chatbot:app",
        host=os.getenv("DAISO_HOST", "127.0.0.1"),
        port=int(os.getenv("DAISO_PORT", "8000")),
        reload=os.getenv("DAISO_RELOAD", "true").strip().lower() == "true",
    )
