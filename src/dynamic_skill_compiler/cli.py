from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from typing import Any

from dynamic_skill_compiler.config import DSCConfig, prompt_for_config, resolve_config, save_config
from dynamic_skill_compiler.models import LocalEnvironment
from dynamic_skill_compiler.pipeline import CompilerConfig, DynamicSkillCompiler
from dynamic_skill_compiler.retriever import LocalSkillLibraryRetriever
from dynamic_skill_compiler.semantic import SemanticSoftMatcher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dsc",
        description="Compile a local skill library into a task-specific DSC package.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Task query to compile. If omitted, the query is read from stdin.",
    )
    parser.add_argument(
        "--skills-dir",
        help="Directory containing local skill folders with SKILL.md files.",
    )
    parser.add_argument(
        "--configure",
        action="store_true",
        help="Prompt for optional OpenAI settings and save them for future DSC runs.",
    )
    parser.add_argument(
        "--semantic",
        choices=("auto", "on", "off"),
        default="auto",
        help="Semantic optimization mode. auto uses saved/env credentials when available.",
    )
    parser.add_argument(
        "--no-config-prompt",
        action="store_true",
        help="Do not prompt for OpenAI settings when no DSC config exists.",
    )
    parser.add_argument(
        "--openai-api-key",
        default="",
        help="OpenAI API key for semantic optimization. Overrides env and saved config.",
    )
    parser.add_argument(
        "--openai-base-url",
        default="",
        help="Optional OpenAI-compatible base URL for semantic optimization.",
    )
    parser.add_argument(
        "--config-path",
        default="",
        help="Optional DSC config path. Defaults to ~/.dynamic_skill_compiler/config.json.",
    )
    parser.add_argument(
        "--benchmark",
        default="generic",
        help="Optional environment/benchmark label used by adaptive compiler profiles.",
    )
    parser.add_argument(
        "--cwd",
        default=".",
        help="Execution working directory used for localization.",
    )
    parser.add_argument(
        "--workspace-root",
        default=".",
        help="Workspace root used for localization.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable or "python",
        help="Python executable name/path used for localization.",
    )
    parser.add_argument(
        "--min-relevance",
        type=float,
        default=CompilerConfig.min_relevance,
        help="Minimum utility score required for skill selection.",
    )
    parser.add_argument(
        "--preserve-top-k",
        type=int,
        default=CompilerConfig.preserve_top_k,
        help="Always preserve at least this many top-scored skills.",
    )
    parser.add_argument(
        "--max-selected-skills",
        type=int,
        default=CompilerConfig.max_selected_skills,
        help="Hard cap for selected skills. Use 0 for no explicit cap.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    parser.add_argument(
        "--include-instructions",
        action="store_true",
        help="Include localized selected instructions in the JSON output.",
    )
    return parser


def compile_from_args(args: argparse.Namespace) -> dict[str, Any]:
    query = args.query if args.query is not None else sys.stdin.read().strip()
    if not query:
        raise SystemExit("A task query is required, either as an argument or on stdin.")
    if not args.skills_dir:
        raise SystemExit("--skills-dir is required unless you are running --configure.")

    retriever = LocalSkillLibraryRetriever(args.skills_dir)
    soft_matcher = _build_soft_matcher(args, retriever)
    compiler = DynamicSkillCompiler(
        retriever=retriever,
        config=CompilerConfig(
            min_relevance=args.min_relevance,
            preserve_top_k=args.preserve_top_k,
            max_selected_skills=args.max_selected_skills,
        ),
        soft_matcher=soft_matcher,
    )
    compiled = compiler.compile(
        query,
        environment=LocalEnvironment(
            cwd=args.cwd,
            workspace_root=args.workspace_root,
            python_bin=args.python_bin,
            benchmark=args.benchmark,
        ),
    )
    summary = compiler.summarize(compiled)
    summary["metrics"] = _to_jsonable(compiled.metrics)
    if args.include_instructions:
        summary["compiled_skills"] = [
            {
                "name": item.asset.name,
                "skill_id": item.asset.skill_id,
                "selected_reason": item.selected_reason,
                "utility_score": item.utility_score,
                "localized_instructions": item.localized_instructions,
                "selected_fragments": [_to_jsonable(fragment) for fragment in item.selected_fragments],
            }
            for item in compiled.compiled_skills
        ]
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.configure:
        prompt_for_config(path=args.config_path or None)
        return 0
    summary = compile_from_args(args)
    indent = 2 if args.pretty else None
    print(json.dumps(summary, ensure_ascii=False, indent=indent))
    return 0


def _build_soft_matcher(
    args: argparse.Namespace,
    retriever: LocalSkillLibraryRetriever,
) -> SemanticSoftMatcher | None:
    if args.semantic == "off":
        return None

    config = _resolve_cli_config(args)
    if not config.semantic_optimization or not config.has_openai_credentials:
        if args.semantic == "on":
            raise SystemExit(
                "Semantic optimization is enabled but no OpenAI API key is configured. "
                "Run `dsc --configure`, set OPENAI_API_KEY, or pass --openai-api-key."
            )
        return None

    try:
        matcher = SemanticSoftMatcher.from_openai(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url or None,
        )
        # Warm up against the local library once so graph scoring can use semantic signals.
        assets = retriever.retrieve(None)  # type: ignore[arg-type]
        matcher.warm_up_skills(assets)
        return matcher
    except ImportError as exc:
        if args.semantic == "on":
            raise SystemExit(
                "Semantic optimization requires the OpenAI and NumPy dependencies. "
                "Install with `pip install dynamic-skill-compiler[semantic]`."
            ) from exc
        print(
            "Semantic optimization unavailable; falling back to local lexical optimization.",
            file=sys.stderr,
        )
        return None
    except Exception as exc:
        if args.semantic == "on":
            raise SystemExit(f"Failed to initialize semantic optimization: {exc}") from exc
        print(
            f"Semantic optimization skipped ({exc}); falling back to local lexical optimization.",
            file=sys.stderr,
        )
        return None


def _resolve_cli_config(args: argparse.Namespace) -> DSCConfig:
    if args.openai_api_key:
        config = DSCConfig(
            semantic_optimization=True,
            openai_api_key=args.openai_api_key,
            openai_base_url=args.openai_base_url,
        )
        return config

    config = resolve_config(
        prompt_if_missing=(not args.no_config_prompt and args.semantic != "off"),
        path=args.config_path or None,
    )
    if args.openai_base_url and config.has_openai_credentials:
        config.openai_base_url = args.openai_base_url
        save_config(config, args.config_path or None)
    return config


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


if __name__ == "__main__":
    raise SystemExit(main())
