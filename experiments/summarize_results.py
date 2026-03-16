import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


def infer_skill_mode(run_name: str, sample: dict) -> str:
    if sample.get("skill_strategy"):
        mode = str(sample["skill_strategy"])
        return "skillnet" if mode == "baseline" else mode
    if "_skill_True" in run_name:
        return "skillnet"
    if "_skill_False" in run_name:
        return "none"
    return "unknown"


def numeric_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_result_files(run_dir: Path) -> List[dict]:
    rows = []
    for path in sorted(run_dir.glob("idx_*.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return rows


def summarize_run(run_dir: Path) -> Optional[Dict]:
    rows = load_result_files(run_dir)
    if not rows:
        return None

    rewards = [float(row.get("reward", 0) or 0) for row in rows]
    steps = [float(row.get("steps", 0) or 0) for row in rows]
    successes = [
        1.0 if bool(row.get("task_done")) or float(row.get("reward", 0) or 0) > 0 else 0.0
        for row in rows
    ]
    selected_skill_counts = [len(row.get("relevant_skill_names", [])) for row in rows]

    compiler_rows = [row["compiler_metrics"] for row in rows if isinstance(row.get("compiler_metrics"), dict)]
    avg_selected_count = numeric_mean([float(item.get("selected_count", 0) or 0) for item in compiler_rows])
    avg_candidate_count = numeric_mean([float(item.get("candidate_count", 0) or 0) for item in compiler_rows])
    avg_coverage = numeric_mean([float(item.get("coverage_score", 0) or 0) for item in compiler_rows])
    avg_redundancy = numeric_mean([float(item.get("redundancy_reduction", 0) or 0) for item in compiler_rows])
    avg_token_before = numeric_mean([float(item.get("estimated_token_cost_before", 0) or 0) for item in compiler_rows])
    avg_token_after = numeric_mean([float(item.get("estimated_token_cost_after", 0) or 0) for item in compiler_rows])
    avg_subgoal_count = numeric_mean([float(item.get("subgoal_count", 0) or 0) for item in compiler_rows])
    avg_covered_subgoal_count = numeric_mean([float(item.get("covered_subgoal_count", 0) or 0) for item in compiler_rows])
    avg_fragment_before = numeric_mean([float(item.get("fragment_count_before", 0) or 0) for item in compiler_rows])
    avg_fragment_after = numeric_mean([float(item.get("fragment_count_after", 0) or 0) for item in compiler_rows])
    avg_fragment_token_after = numeric_mean([float(item.get("fragment_token_cost_after", 0) or 0) for item in compiler_rows])

    sample = rows[0]
    return {
        "run_name": run_dir.name,
        "path": str(run_dir),
        "task_count": len(rows),
        "skill_mode": infer_skill_mode(run_dir.name, sample),
        "avg_reward": numeric_mean(rewards),
        "avg_steps": numeric_mean(steps),
        "success_rate": numeric_mean(successes),
        "avg_selected_skill_names": numeric_mean(selected_skill_counts),
        "avg_compiler_selected_count": avg_selected_count,
        "avg_compiler_candidate_count": avg_candidate_count,
        "avg_compiler_coverage": avg_coverage,
        "avg_compiler_redundancy_reduction": avg_redundancy,
        "avg_compiler_token_cost_before": avg_token_before,
        "avg_compiler_token_cost_after": avg_token_after,
        "avg_compiler_token_reduction": avg_token_before - avg_token_after,
        "avg_compiler_subgoal_count": avg_subgoal_count,
        "avg_compiler_covered_subgoal_count": avg_covered_subgoal_count,
        "avg_compiler_fragment_count_before": avg_fragment_before,
        "avg_compiler_fragment_count_after": avg_fragment_after,
        "avg_compiler_fragment_token_cost_after": avg_fragment_token_after,
    }


def summarize_tree(results_root: Path) -> Dict:
    runs = []
    for run_dir in sorted(results_root.iterdir()):
        if not run_dir.is_dir():
            continue
        summary = summarize_run(run_dir)
        if summary:
            runs.append(summary)
    return {"results_root": str(results_root), "runs": runs}


def render_markdown(summary: Dict) -> str:
    lines = [
        f"# Result Summary: {summary['results_root']}",
        "",
        "| Run | Mode | Tasks | Success | Avg Reward | Avg Steps | Avg Skills | Compiler Coverage | Token Reduction |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in summary["runs"]:
        lines.append(
            "| {run_name} | {skill_mode} | {task_count} | {success_rate:.3f} | "
            "{avg_reward:.3f} | {avg_steps:.3f} | {avg_selected_skill_names:.2f} | "
            "{avg_compiler_coverage:.3f} | {avg_compiler_token_reduction:.3f} |".format(**run)
        )
    lines.extend([
        "",
        "| Run | Subgoals | Covered Subgoals | Fragments Before | Fragments After | Fragment Token Cost |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for run in summary["runs"]:
        lines.append(
            "| {run_name} | {avg_compiler_subgoal_count:.2f} | {avg_compiler_covered_subgoal_count:.2f} | "
            "{avg_compiler_fragment_count_before:.2f} | {avg_compiler_fragment_count_after:.2f} | "
            "{avg_compiler_fragment_token_cost_after:.3f} |".format(**run)
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, help="Directory containing experiment run folders")
    parser.add_argument("--json-out", default="", help="Optional JSON output path")
    parser.add_argument("--md-out", default="", help="Optional Markdown output path")
    args = parser.parse_args()

    summary = summarize_tree(Path(args.results_root))
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.md_out:
        Path(args.md_out).write_text(render_markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
