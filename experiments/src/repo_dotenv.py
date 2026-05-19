"""Load repo-root `.env` (KEY=value) and apply experiment conveniences."""

from __future__ import annotations

import os
from pathlib import Path


def load_repo_dotenv(experiments_dir: str) -> None:
    root = Path(experiments_dir).resolve().parent
    path = root / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, val)
    if not os.environ.get("API_KEY") and os.environ.get("OPENAI_API_KEY"):
        os.environ["API_KEY"] = os.environ["OPENAI_API_KEY"]
    if not os.environ.get("BASE_URL") and os.environ.get("OPENAI_BASE_URL"):
        os.environ["BASE_URL"] = os.environ["OPENAI_BASE_URL"]
    java_exe = os.environ.get("JAVA_EXECUTABLE")
    if java_exe:
        java_exe = os.path.expanduser(java_exe)
        if os.path.isfile(java_exe):
            bin_dir = str(Path(java_exe).resolve().parent)
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
