from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
from dataclasses import asdict, is_dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from dynamic_skill_compiler.models import LocalEnvironment
from dynamic_skill_compiler.pipeline import CompilerConfig, DynamicSkillCompiler
from dynamic_skill_compiler.retriever import LocalSkillLibraryRetriever


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dsc-web",
        description="Serve a local Dynamic Skill Compiler visualization workbench.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind.")
    parser.add_argument(
        "--skills-dir",
        default="",
        help="Default local skill library directory shown in the web UI.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), _handler_factory(args.skills_dir))
    url = f"http://{args.host}:{args.port}"
    print(f"DSC web workbench running at {url}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down DSC web workbench.", file=sys.stderr)
    finally:
        server.server_close()
    return 0


def _handler_factory(default_skills_dir: str):
    class DSCWebHandler(BaseHTTPRequestHandler):
        server_version = "DSCWeb/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"{self.address_string()} - {fmt % args}", file=sys.stderr)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/health":
                self._send_json({"ok": True, "default_skills_dir": _default_skills_dir(default_skills_dir)})
                return
            if parsed.path == "/api/skills":
                params = parse_qs(parsed.query)
                skills_dir = params.get("skills_dir", [_default_skills_dir(default_skills_dir)])[0]
                self._handle_skills(skills_dir)
                return
            self._serve_static(parsed.path)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/compile":
                self._send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint.")
                return
            try:
                payload = self._read_json()
                self._handle_compile(payload)
            except ValueError as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
            except Exception as exc:
                self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Compile failed: {exc}")

        def _handle_skills(self, skills_dir: str) -> None:
            resolved = _resolve_skills_dir(skills_dir)
            retriever = LocalSkillLibraryRetriever(str(resolved))
            skills = retriever.retrieve(None)  # type: ignore[arg-type]
            self._send_json(
                {
                    "skills_dir": str(resolved),
                    "skill_count": len(skills),
                    "skills": [_skill_to_json(skill) for skill in sorted(skills, key=lambda item: item.skill_id)],
                }
            )

        def _handle_compile(self, payload: dict[str, Any]) -> None:
            query = str(payload.get("query", "")).strip()
            if not query:
                raise ValueError("query is required.")
            skills_dir = _resolve_skills_dir(str(payload.get("skills_dir") or _default_skills_dir(default_skills_dir)))
            if not skills_dir.is_dir():
                raise ValueError(f"skills_dir does not exist or is not a directory: {skills_dir}")

            retriever = LocalSkillLibraryRetriever(str(skills_dir))
            compiler = DynamicSkillCompiler(
                retriever=retriever,
                config=CompilerConfig(
                    min_relevance=_float(payload.get("min_relevance"), CompilerConfig.min_relevance),
                    preserve_top_k=_int(payload.get("preserve_top_k"), CompilerConfig.preserve_top_k),
                    max_selected_skills=_int(
                        payload.get("max_selected_skills"), CompilerConfig.max_selected_skills
                    ),
                ),
            )
            compiled = compiler.compile(
                query,
                environment=LocalEnvironment(
                    cwd=str(payload.get("cwd") or "."),
                    workspace_root=str(payload.get("workspace_root") or "."),
                    python_bin=str(payload.get("python_bin") or sys.executable or "python"),
                    benchmark="generic",
                ),
            )
            selected_by_id = {item.asset.skill_id: item for item in compiled.compiled_skills}
            self._send_json(
                {
                    "summary": compiler.summarize(compiled),
                    "query_plan": _to_jsonable(compiled.query_plan),
                    "subgoals": _to_jsonable(compiled.subgoals),
                    "metrics": _to_jsonable(compiled.metrics),
                    "candidate_skills": [
                        {
                            **_skill_to_json(skill),
                            "selected": skill.skill_id in selected_by_id,
                            "dropped_reason": compiled.dropped_skills.get(skill.skill_id, ""),
                        }
                        for skill in sorted(compiled.graph.skills.values(), key=lambda item: item.skill_id)
                    ],
                    "relations": _to_jsonable(compiled.graph.relations),
                    "selected_skills": [
                        {
                            "skill_id": item.asset.skill_id,
                            "name": item.asset.name,
                            "description": item.asset.description,
                            "assigned_subgoals": item.assigned_subgoals,
                            "localized_instructions": item.localized_instructions,
                            "selected_fragments": _to_jsonable(item.selected_fragments),
                            "utility_score": item.utility_score,
                            "selected_reason": item.selected_reason,
                        }
                        for item in compiled.compiled_skills
                    ],
                    "execution_order": compiled.execution_order,
                    "pass_trace": [
                        {
                            "pass_name": trace.pass_name,
                            "before_selected": trace.before_selected,
                            "after_selected": trace.after_selected,
                            "added": trace.added,
                            "removed": trace.removed,
                            "dropped_delta": trace.dropped_delta,
                        }
                        for trace in compiled.pass_traces
                    ],
                    "dropped_skills": compiled.dropped_skills,
                    "notes": compiled.notes,
                }
            )

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length <= 0:
                return {}
            raw = self.rfile.read(length).decode("utf-8")
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("JSON body must be an object.")
            return data

        def _serve_static(self, path: str) -> None:
            asset_name = "index.html" if path in ("", "/") else path.lstrip("/")
            if "/" in asset_name or asset_name.startswith("."):
                self._send_error(HTTPStatus.NOT_FOUND, "Static asset not found.")
                return
            try:
                asset = resources.files("dynamic_skill_compiler.web_assets").joinpath(asset_name)
                content = asset.read_bytes()
            except FileNotFoundError:
                self._send_error(HTTPStatus.NOT_FOUND, "Static asset not found.")
                return
            content_type = mimetypes.guess_type(asset_name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def _send_error(self, status: HTTPStatus, message: str) -> None:
            self._send_json({"ok": False, "error": message}, status)

    return DSCWebHandler


def _default_skills_dir(configured: str) -> str:
    if configured:
        return configured
    cwd = Path.cwd()
    candidates = [
        cwd / "experiments" / "src" / "skills" / "scienceworld",
        cwd / "SkillNet" / "experiments" / "src" / "skills" / "scienceworld",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate)
    return str(cwd)


def _resolve_skills_dir(skills_dir: str) -> Path:
    return Path(os.path.expanduser(skills_dir)).resolve()


def _skill_to_json(skill: Any) -> dict[str, Any]:
    return {
        "skill_id": skill.skill_id,
        "name": skill.name,
        "description": skill.description,
        "category": skill.category,
        "source": skill.source,
        "location": skill.location,
        "capabilities": sorted(skill.capabilities),
        "dependencies": sorted(skill.dependencies),
        "contains": sorted(skill.contains),
        "composes_with": sorted(skill.composes_with),
        "similar_to": sorted(skill.similar_to),
        "quality_scores": skill.quality_scores,
        "token_cost": skill.token_cost,
        "execution_cost": skill.execution_cost,
        "latency_ms": skill.latency_ms,
        "instruction_count": len(skill.instructions),
        "instructions": skill.instructions,
        "metadata": skill.metadata,
    }


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_to_jsonable(item) for item in value)
    return value


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    raise SystemExit(main())
