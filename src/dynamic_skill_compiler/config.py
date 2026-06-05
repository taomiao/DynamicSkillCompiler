from __future__ import annotations

import getpass
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


CONFIG_DIR_ENV = "DSC_CONFIG_DIR"
CONFIG_PATH_ENV = "DSC_CONFIG_PATH"


@dataclass
class DSCConfig:
    semantic_optimization: bool = False
    openai_api_key: str = ""
    openai_base_url: str = ""

    @property
    def has_openai_credentials(self) -> bool:
        return bool(self.openai_api_key.strip())


def default_config_path() -> Path:
    if os.environ.get(CONFIG_PATH_ENV):
        return Path(os.environ[CONFIG_PATH_ENV]).expanduser()
    config_dir = os.environ.get(CONFIG_DIR_ENV, "~/.dynamic_skill_compiler")
    return Path(config_dir).expanduser() / "config.json"


def load_config(path: str | Path | None = None) -> DSCConfig:
    config_path = Path(path).expanduser() if path else default_config_path()
    if not config_path.is_file():
        return DSCConfig()
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return DSCConfig()
    if not isinstance(data, dict):
        return DSCConfig()
    return DSCConfig(
        semantic_optimization=bool(data.get("semantic_optimization", False)),
        openai_api_key=str(data.get("openai_api_key", "") or ""),
        openai_base_url=str(data.get("openai_base_url", "") or ""),
    )


def save_config(config: DSCConfig, path: str | Path | None = None) -> Path:
    config_path = Path(path).expanduser() if path else default_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "semantic_optimization": config.semantic_optimization,
        "openai_api_key": config.openai_api_key,
        "openai_base_url": config.openai_base_url,
    }
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        config_path.chmod(0o600)
    except OSError:
        pass
    return config_path


def env_config() -> DSCConfig:
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY") or ""
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("BASE_URL") or ""
    return DSCConfig(
        semantic_optimization=bool(api_key),
        openai_api_key=api_key,
        openai_base_url=base_url,
    )


def resolve_config(
    *,
    prompt_if_missing: bool,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stderr,
    path: str | Path | None = None,
) -> DSCConfig:
    env = env_config()
    if env.has_openai_credentials:
        return env

    stored = load_config(path)
    if stored.semantic_optimization and stored.has_openai_credentials:
        return stored
    if stored.semantic_optimization is False and path_exists(path):
        return stored

    if not prompt_if_missing or not _is_interactive(input_stream, output_stream):
        return stored

    return prompt_for_config(input_stream=input_stream, output_stream=output_stream, path=path)


def path_exists(path: str | Path | None = None) -> bool:
    config_path = Path(path).expanduser() if path else default_config_path()
    return config_path.exists()


def prompt_for_config(
    *,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stderr,
    path: str | Path | None = None,
) -> DSCConfig:
    print(
        "DSC can optionally use OpenAI embeddings for semantic skill optimization.",
        file=output_stream,
    )
    answer = _read_prompt(
        "Enable semantic optimization now? [y/N]: ",
        input_stream,
        output_stream,
    ).strip().lower()
    if answer not in {"y", "yes"}:
        config = DSCConfig(semantic_optimization=False)
        save_config(config, path)
        print("Skipped. DSC will use local lexical optimization.", file=output_stream)
        return config

    api_key = _read_secret(
        "OpenAI API key (input hidden): ",
        input_stream,
        output_stream,
    ).strip()
    base_url = _read_prompt(
        "OpenAI base URL (optional, press Enter to skip): ",
        input_stream,
        output_stream,
    ).strip()
    if not api_key:
        config = DSCConfig(semantic_optimization=False)
        save_config(config, path)
        print("No API key entered. DSC will use local lexical optimization.", file=output_stream)
        return config

    config = DSCConfig(
        semantic_optimization=True,
        openai_api_key=api_key,
        openai_base_url=base_url,
    )
    config_path = save_config(config, path)
    print(f"Saved DSC config to {config_path}", file=output_stream)
    return config


def _is_interactive(input_stream: TextIO, output_stream: TextIO) -> bool:
    return bool(
        getattr(input_stream, "isatty", lambda: False)()
        and getattr(output_stream, "isatty", lambda: False)()
    )


def _read_prompt(prompt: str, input_stream: TextIO, output_stream: TextIO) -> str:
    output_stream.write(prompt)
    output_stream.flush()
    return input_stream.readline()


def _read_secret(prompt: str, input_stream: TextIO, output_stream: TextIO) -> str:
    if input_stream is sys.stdin and _is_interactive(input_stream, output_stream):
        return getpass.getpass(prompt, stream=output_stream)
    return _read_prompt(prompt, input_stream, output_stream)
