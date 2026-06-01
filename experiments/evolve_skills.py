from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


FAILURE_REWARD_EPS = 1e-9


@dataclass
class FailureCase:
    case_id: str
    path: str
    domain: str
    query: str
    reward: float
    task_done: bool
    steps: int
    selected_skills: list[str]
    failure_modes: list[str]
    evidence: list[str]
    runtime_recompile_count: int = 0
    compiler_metrics: dict[str, Any] | None = None
    last_actions: list[str] = field(default_factory=list)
    last_observations: list[str] = field(default_factory=list)


@dataclass
class SkillEvolutionProposal:
    proposal_id: str
    domain: str
    action: str
    target_skill: str
    failure_mode: str
    rationale: str
    expected_fix: list[str]
    evidence_cases: list[str]
    selected_skills_seen: list[str]
    validation_cases: list[str]
    acceptance_criteria: list[str]
    risk: str


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value if value is not None else default)
    except (TypeError, ValueError):
        return default


def _infer_domain(path: Path, row: dict[str, Any]) -> str:
    haystack = f"{path} {row.get('skill_strategy', '')} {row.get('query', '')}".lower()
    for domain in ("webshop", "scienceworld", "alfworld"):
        if domain in haystack:
            return domain
    return "generic"


def _load_result(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Skipping unreadable result {path}: {exc}")
        return None


def _iter_result_files(result_dirs: Iterable[Path]) -> Iterable[Path]:
    for result_dir in result_dirs:
        if result_dir.is_file() and result_dir.name.startswith("idx_") and result_dir.suffix == ".json":
            yield result_dir
            continue
        if result_dir.is_dir():
            yield from sorted(result_dir.glob("idx_*.json"))


def _message_texts(row: dict[str, Any]) -> list[str]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return []
    texts = []
    for item in messages:
        if isinstance(item, dict):
            content = item.get("content", "")
            if content:
                texts.append(str(content))
    return texts


def _last_actions(texts: list[str], limit: int = 6) -> list[str]:
    actions = []
    for text in texts:
        for match in re.finditer(r"\bAction:\s*([^\n]+)", text, flags=re.IGNORECASE):
            actions.append(match.group(1).strip())
    return actions[-limit:]


def _last_observations(texts: list[str], limit: int = 3) -> list[str]:
    observations = []
    for text in texts:
        if text.lower().startswith("observation:"):
            observations.append(text[len("Observation:") :].strip())
    return observations[-limit:]


def _contains_any(text: str, needles: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in needles)


def _classify_failure_modes(domain: str, row: dict[str, Any], texts: list[str]) -> tuple[list[str], list[str]]:
    joined = "\n".join(texts[-12:])
    query = str(row.get("query", ""))
    actions = _last_actions(texts, limit=10)
    evidence: list[str] = []
    modes: list[str] = []

    runtime_events = row.get("runtime_recompile_events") or []
    if isinstance(runtime_events, list):
        for event in runtime_events:
            if isinstance(event, dict) and event.get("reason"):
                modes.append(f"runtime_{event['reason']}")
                evidence.append(f"Runtime recompile reason: {event['reason']}")

    if _contains_any(joined, ["no results", "not found", "no available", "no viable matches", "no matches"]):
        modes.append("search_dead_end")
        evidence.append("Trace reports no viable candidates after search or relaxation.")
    if actions and actions[-1].lower() in {"none", "stop", "null"}:
        modes.append("abstained_while_incomplete")
        evidence.append("Final action abstained instead of continuing with a concrete environment action.")
    if _contains_any(joined, ["can't", "cannot", "not valid", "unknown action", "nothing happens", "you don't see", "there is no"]):
        modes.append("action_grounding_failure")
        evidence.append("Trace contains environment rejection for action syntax, target grounding, or missing preconditions.")
    if _contains_any(joined, ["ambiguous request", "please enter the number"]):
        modes.append("ambiguity_loop")
        evidence.append("Trace contains an unresolved ambiguous-choice prompt.")
    if _contains_any(query, ["clean", "heat", "cool", "wash", "slice", "toggle", "turn on", "turn off"]):
        modes.append("transform_precondition_failure")
        evidence.append("Task requires an object/state transformation with explicit preconditions.")
    if _contains_any(joined, ["same observation", "repeated", "stagnation"]) or len(set(actions[-4:])) < len(actions[-4:]):
        modes.append("stagnation_or_action_loop")
        evidence.append("Recent trace repeats actions or observations without new progress.")

    if not modes:
        modes.append("low_reward_generic")
        evidence.append("Case failed or scored low without a specific portable signature.")

    return list(dict.fromkeys(modes)), list(dict.fromkeys(evidence))


def _is_failure(row: dict[str, Any], reward_threshold: float) -> bool:
    reward = _safe_float(row.get("reward"))
    task_done = bool(row.get("task_done"))
    runtime_count = _safe_int(row.get("runtime_recompile_count"))
    return (not task_done) or reward <= reward_threshold + FAILURE_REWARD_EPS or runtime_count > 0


def collect_failure_cases(
    result_dirs: list[Path],
    *,
    reward_threshold: float,
    max_cases: int,
) -> list[FailureCase]:
    cases: list[FailureCase] = []
    for path in _iter_result_files(result_dirs):
        row = _load_result(path)
        if row is None or not _is_failure(row, reward_threshold):
            continue
        texts = _message_texts(row)
        domain = _infer_domain(path, row)
        modes, evidence = _classify_failure_modes(domain, row, texts)
        compiler_metrics = row.get("compiler_metrics")
        case = FailureCase(
            case_id=path.stem,
            path=str(path),
            domain=domain,
            query=str(row.get("query", "")),
            reward=_safe_float(row.get("reward")),
            task_done=bool(row.get("task_done")),
            steps=_safe_int(row.get("steps")),
            selected_skills=[str(item) for item in row.get("relevant_skill_names", [])],
            failure_modes=modes,
            evidence=evidence,
            runtime_recompile_count=_safe_int(row.get("runtime_recompile_count")),
            compiler_metrics=compiler_metrics if isinstance(compiler_metrics, dict) else None,
            last_actions=_last_actions(texts),
            last_observations=_last_observations(texts),
        )
        cases.append(case)
        if max_cases > 0 and len(cases) >= max_cases:
            break
    return cases


def _proposal_target(domain: str, mode: str, selected_skills: Counter[str]) -> tuple[str, str]:
    preferred = selected_skills.most_common()
    if preferred:
        return "edit_existing_skill", preferred[0][0]
    return "create_new_skill", f"{domain}-failure-recovery"


def _expected_fix(domain: str, mode: str) -> list[str]:
    fixes = {
        "search_dead_end": [
            "Use staged search recovery: preserve hard constraints, relax only soft descriptors, and record what changed.",
            "Before declaring failure, inspect or verify any plausible visible candidate with fresh evidence.",
        ],
        "abstained_while_incomplete": [
            "Replace terminal placeholders with one concrete legal environment action from the latest observation.",
            "If no exact candidate exists, either broaden search or verify the best partial candidate before stopping.",
        ],
        "action_grounding_failure": [
            "After a rejection, refresh state and retry with exact target names/actions from the latest observation.",
            "Repair only the failed precondition instead of restarting completed subgoals.",
        ],
        "ambiguity_loop": [
            "When the environment asks for a number, answer only the listed number.",
            "Use object identity evidence such as contents, location, and recent action intent to choose the index.",
        ],
        "transform_precondition_failure": [
            "Separate locate/acquire, transform, verify, and final placement phases.",
            "Verify required tools, object availability, and state before applying a transform.",
        ],
        "stagnation_or_action_loop": [
            "Track repeated actions and observations, then require a changed target, query, location, or precondition repair.",
            "Preserve completed subgoals while advancing to the next unresolved step.",
        ],
    }
    return fixes.get(mode, ["Add concrete recovery guidance for this recurring low-reward failure mode."])


def _risk(domain: str, mode: str) -> str:
    return "May create redundant broad guidance unless validated against held-out cases from multiple environments."


def build_proposals(cases: list[FailureCase], *, min_cases_per_proposal: int) -> list[SkillEvolutionProposal]:
    grouped: dict[tuple[str, str], list[FailureCase]] = defaultdict(list)
    for case in cases:
        for mode in case.failure_modes:
            grouped[(case.domain, mode)].append(case)

    proposals: list[SkillEvolutionProposal] = []
    for idx, ((domain, mode), mode_cases) in enumerate(sorted(grouped.items()), start=1):
        if len(mode_cases) < min_cases_per_proposal:
            continue
        selected_counter: Counter[str] = Counter()
        for case in mode_cases:
            selected_counter.update(case.selected_skills)
        action, target_skill = _proposal_target(domain, mode, selected_counter)
        evidence_cases = [case.case_id for case in mode_cases[:8]]
        validation_cases = [case.case_id for case in mode_cases[: min(5, len(mode_cases))]]
        selected_seen = [name for name, _ in selected_counter.most_common(8)]
        proposals.append(
            SkillEvolutionProposal(
                proposal_id=f"evo-{domain}-{idx:03d}",
                domain=domain,
                action=action,
                target_skill=target_skill,
                failure_mode=mode,
                rationale=(
                    f"{len(mode_cases)} failure case(s) share `{mode}`. "
                    f"Target `{target_skill}` because it is either already selected in these traces "
                    "or is the closest existing skill for the missing behavior."
                ),
                expected_fix=_expected_fix(domain, mode),
                evidence_cases=evidence_cases,
                selected_skills_seen=selected_seen,
                validation_cases=validation_cases,
                acceptance_criteria=[
                    "DSC+candidate improves average reward over current DSC on the validation cases.",
                    "No validation case that currently succeeds regresses to reward 0.",
                    "Average selected skill count and estimated token cost do not increase by more than 20%.",
                    "The candidate fixes the named failure mode in at least one inspected trajectory.",
                ],
                risk=_risk(domain, mode),
            )
        )
    return proposals


def _render_markdown(cases: list[FailureCase], proposals: list[SkillEvolutionProposal]) -> str:
    lines = [
        "# EvoSkill DSC Dry Run",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Failure Cases",
        "",
        "| Case | Domain | Reward | Done | Steps | Failure Modes | Selected Skills |",
        "| --- | --- | ---: | --- | ---: | --- | --- |",
    ]
    for case in cases:
        lines.append(
            "| {case_id} | {domain} | {reward:.3f} | {task_done} | {steps} | {modes} | {skills} |".format(
                case_id=case.case_id,
                domain=case.domain,
                reward=case.reward,
                task_done="yes" if case.task_done else "no",
                steps=case.steps,
                modes=", ".join(case.failure_modes),
                skills=", ".join(case.selected_skills[:6]),
            )
        )

    lines.extend(["", "## Proposals", ""])
    if not proposals:
        lines.append("No proposals met the minimum case threshold.")
        return "\n".join(lines) + "\n"

    for proposal in proposals:
        lines.extend(
            [
                f"### {proposal.proposal_id}: {proposal.target_skill}",
                "",
                f"- Domain: `{proposal.domain}`",
                f"- Action: `{proposal.action}`",
                f"- Failure mode: `{proposal.failure_mode}`",
                f"- Rationale: {proposal.rationale}",
                f"- Evidence cases: {', '.join(proposal.evidence_cases)}",
                f"- Selected skills seen: {', '.join(proposal.selected_skills_seen) or '(none)'}",
                "- Expected fix:",
            ]
        )
        lines.extend(f"  - {item}" for item in proposal.expected_fix)
        lines.append("- Acceptance criteria:")
        lines.extend(f"  - {item}" for item in proposal.acceptance_criteria)
        lines.append(f"- Risk: {proposal.risk}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _write_staging(
    staging_dir: Path,
    cases: list[FailureCase],
    proposals: list[SkillEvolutionProposal],
    *,
    overwrite: bool,
) -> None:
    if staging_dir.exists() and any(staging_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Staging directory is not empty: {staging_dir}")
    if staging_dir.exists() and overwrite:
        for child in staging_dir.iterdir():
            if child.name in {"proposals.json", "proposals.md"}:
                child.unlink()
            elif child.is_dir() and child.name.startswith("evo-"):
                shutil.rmtree(child)
    staging_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases": [asdict(case) for case in cases],
        "proposals": [asdict(proposal) for proposal in proposals],
    }
    (staging_dir / "proposals.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (staging_dir / "proposals.md").write_text(
        _render_markdown(cases, proposals),
        encoding="utf-8",
    )
    for proposal in proposals:
        proposal_dir = staging_dir / proposal.proposal_id
        proposal_dir.mkdir(exist_ok=True)
        (proposal_dir / "proposal.json").write_text(
            json.dumps(asdict(proposal), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        patch_lines = [
            f"# Candidate Patch: {proposal.target_skill}",
            "",
            f"Domain: `{proposal.domain}`",
            f"Failure mode: `{proposal.failure_mode}`",
            "",
            "## Proposed Changes",
            "",
        ]
        patch_lines.extend(f"- {item}" for item in proposal.expected_fix)
        patch_lines.extend(
            [
                "",
                "## Validation Plan",
                "",
                "Compare current DSC against DSC plus this candidate patch on:",
                "",
            ]
        )
        patch_lines.extend(f"- `{case_id}`" for case_id in proposal.validation_cases)
        patch_lines.extend(["", "## Acceptance Criteria", ""])
        patch_lines.extend(f"- {item}" for item in proposal.acceptance_criteria)
        patch_lines.extend(["", "## Risk", "", proposal.risk, ""])
        (proposal_dir / "PATCH_PLAN.md").write_text("\n".join(patch_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine failed DSC trajectories and produce EvoSkill-style skill evolution proposals."
    )
    parser.add_argument(
        "--result-dir",
        action="append",
        required=True,
        help="Result run directory or idx_*.json file. May be passed multiple times.",
    )
    parser.add_argument("--reward-threshold", type=float, default=0.0)
    parser.add_argument("--max-cases", type=int, default=50)
    parser.add_argument("--min-cases-per-proposal", type=int, default=2)
    parser.add_argument("--staging-dir", default="")
    parser.add_argument("--write-staging", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result_dirs = [Path(item) for item in args.result_dir]
    cases = collect_failure_cases(
        result_dirs,
        reward_threshold=args.reward_threshold,
        max_cases=args.max_cases,
    )
    proposals = build_proposals(cases, min_cases_per_proposal=args.min_cases_per_proposal)

    summary = {
        "case_count": len(cases),
        "proposal_count": len(proposals),
        "failure_modes": Counter(mode for case in cases for mode in case.failure_modes),
        "proposals": [asdict(proposal) for proposal in proposals],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.write_staging:
        staging_dir = Path(args.staging_dir) if args.staging_dir else Path("results/evolution_staging/latest")
        _write_staging(staging_dir, cases, proposals, overwrite=args.overwrite)
        print(f"[INFO] Wrote EvoSkill staging artifacts to {staging_dir}")


if __name__ == "__main__":
    main()
