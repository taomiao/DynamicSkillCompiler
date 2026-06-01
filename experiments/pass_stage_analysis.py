from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent
DSC_SRC = ROOT.parent / "src"
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))
if str(DSC_SRC) in sys.path:
    sys.path.remove(str(DSC_SRC))
sys.path.insert(0, str(DSC_SRC))

from dynamic_skill_compiler import (  # noqa: E402
    CompilerConfig,
    DEFAULT_GRAPH_PASSES,
    DynamicSkillCompiler,
    GRAPH_PASS_PRESETS,
    LocalEnvironment,
    LocalSkillLibraryRetriever,
)


DOMAIN_TO_SKILLS_DIR = {
    "scienceworld": ROOT / "src" / "skills" / "scienceworld",
    "alfworld": ROOT / "src" / "skills" / "alfworld",
    "webshop": ROOT / "src" / "skills" / "webshop",
}

PASS_EXPECTATIONS = {
    "select_covering_skills": "Should seed the first core skill set for subgoal coverage.",
    "fallback_selection": "Should usually be a no-op; only rescues empty selections.",
    "add_dependencies": "Should close dependency holes exposed by the previous stage.",
    "repair_coverage": "Should patch missing subgoals or required capabilities.",
    "augment_compositional_support": "Should add workflow/composition support skills when structure needs them.",
    "prune_similar": "Should remove near-duplicate skills without hurting task coverage.",
    "prune_broad_containers": "Should prefer specific child skills over broad parent containers when safe.",
    "trim_low_contribution": "Should remove low-value residual skills while preserving coverage and executability.",
}


def load_cases_from_results(result_dir: Path, domain: str, limit: int | None = None) -> list[dict]:
    cases = []
    for path in sorted(result_dir.glob("idx_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        query = payload.get("query")
        if not isinstance(query, str) or not query.strip():
            continue
        cases.append(
            {
                "case_id": path.stem,
                "domain": domain,
                "query": query.strip(),
                "task_name": payload.get("name", ""),
            }
        )
        if limit is not None and len(cases) >= limit:
            break
    return cases


def load_cases_from_json(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = []
    for index, item in enumerate(payload):
        cases.append(
            {
                "case_id": item.get("case_id", f"case_{index}"),
                "domain": item["domain"],
                "query": item["query"],
                "task_name": item.get("task_name", ""),
            }
        )
    return cases


def compile_case(case: dict, graph_passes: tuple[str, ...], adaptive_workflow_bias: bool = True) -> dict:
    domain = case["domain"]
    retriever = LocalSkillLibraryRetriever(str(DOMAIN_TO_SKILLS_DIR[domain]))
    compiler = DynamicSkillCompiler(
        retriever=retriever,
        config=CompilerConfig(
            graph_passes=graph_passes,
            adaptive_workflow_bias=adaptive_workflow_bias,
        ),
    )
    compiled = compiler.compile(
        query=case["query"],
        environment=LocalEnvironment(
            cwd=str(ROOT),
            workspace_root=str(DOMAIN_TO_SKILLS_DIR[domain]),
            benchmark=domain,
        ),
    )
    return {
        "case_id": case["case_id"],
        "task_name": case.get("task_name", ""),
        "query": case["query"],
        "selected_skills": [item.asset.name for item in compiled.compiled_skills],
        "execution_order": compiled.execution_order,
        "metrics": compiled.metrics.__dict__,
        "dropped_skills": compiled.dropped_skills,
        "notes": compiled.notes,
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
    }


def build_stage_labels(pass_sequence: tuple[str, ...]) -> list[str]:
    occurrence_counter = defaultdict(int)
    labels = []
    for index, pass_name in enumerate(pass_sequence, start=1):
        occurrence_counter[pass_name] += 1
        occurrence = occurrence_counter[pass_name]
        label = f"{index:02d}_{pass_name}"
        if occurrence > 1:
            label = f"{label}_{occurrence}"
        labels.append(label)
    return labels


def build_leave_one_out_variants(pass_sequence: tuple[str, ...], stage_labels: list[str]) -> list[dict]:
    variants = [
        {
            "variant": "default",
            "removed_stage_index": None,
            "removed_stage_label": None,
            "graph_passes": pass_sequence,
        }
    ]
    for index, stage_label in enumerate(stage_labels):
        graph_passes = list(pass_sequence)
        del graph_passes[index]
        variants.append(
            {
                "variant": f"drop_{stage_label}",
                "removed_stage_index": index,
                "removed_stage_label": stage_label,
                "graph_passes": tuple(graph_passes),
            }
        )
    return variants


def stage_trace_summary(default_rows: list[dict]) -> list[dict]:
    pass_sequence = tuple(trace["pass_name"] for trace in default_rows[0]["pass_trace"])
    stage_labels = build_stage_labels(pass_sequence)
    summaries = []
    for index, stage_label in enumerate(stage_labels):
        traces = [row["pass_trace"][index] for row in default_rows]
        added_counter = Counter()
        removed_counter = Counter()
        dropped_reason_counter = Counter()
        for trace in traces:
            added_counter.update(trace["added"])
            removed_counter.update(trace["removed"])
            dropped_reason_counter.update(trace["dropped_delta"].values())
        summaries.append(
            {
                "stage_index": index,
                "stage_label": stage_label,
                "pass_name": pass_sequence[index],
                "expectation": PASS_EXPECTATIONS[pass_sequence[index]],
                "changed_cases": sum(
                    1
                    for trace in traces
                    if trace["added"] or trace["removed"] or trace["dropped_delta"]
                ),
                "avg_added": round(mean(len(trace["added"]) for trace in traces), 3),
                "avg_removed": round(mean(len(trace["removed"]) for trace in traces), 3),
                "avg_selected_after": round(mean(len(trace["after_selected"]) for trace in traces), 3),
                "top_added_skills": added_counter.most_common(6),
                "top_removed_skills": removed_counter.most_common(6),
                "top_drop_reasons": dropped_reason_counter.most_common(6),
            }
        )
    return summaries


def compare_variant_to_default(
    default_rows: dict[str, dict],
    variant_rows: list[dict],
    pass_sequence: tuple[str, ...],
    stage_labels: list[str],
    stage_index: int,
) -> dict:
    delta_rows = []
    for row in variant_rows:
        base = default_rows[row["case_id"]]
        base_metrics = base["metrics"]
        metrics = row["metrics"]
        base_skills = set(base["selected_skills"])
        variant_skills = set(row["selected_skills"])
        delta_rows.append(
            {
                "case_id": row["case_id"],
                "task_name": row["task_name"],
                "selected_changed": sorted(base_skills) != sorted(variant_skills),
                "skills_added_vs_default": sorted(variant_skills - base_skills),
                "skills_removed_vs_default": sorted(base_skills - variant_skills),
                "delta_selected_count": metrics["selected_count"] - base_metrics["selected_count"],
                "delta_coverage_score": round(
                    metrics["coverage_score"] - base_metrics["coverage_score"], 6
                ),
                "delta_covered_subgoal_count": (
                    metrics["covered_subgoal_count"] - base_metrics["covered_subgoal_count"]
                ),
                "delta_fragment_count_after": (
                    metrics["fragment_count_after"] - base_metrics["fragment_count_after"]
                ),
                "delta_estimated_token_cost_after": round(
                    metrics["estimated_token_cost_after"]
                    - base_metrics["estimated_token_cost_after"],
                    6,
                ),
                "default_selected_skills": base["selected_skills"],
                "variant_selected_skills": row["selected_skills"],
            }
        )

    changed_rows = [row for row in delta_rows if row["selected_changed"]]
    return {
        "stage_index": stage_index,
        "stage_label": stage_labels[stage_index],
        "pass_name": pass_sequence[stage_index],
        "changed_final_cases": len(changed_rows),
        "cases_with_lower_coverage": sum(row["delta_coverage_score"] < 0 for row in delta_rows),
        "cases_with_higher_coverage": sum(row["delta_coverage_score"] > 0 for row in delta_rows),
        "cases_with_more_selected_skills": sum(row["delta_selected_count"] > 0 for row in delta_rows),
        "cases_with_fewer_selected_skills": sum(row["delta_selected_count"] < 0 for row in delta_rows),
        "avg_delta_selected_count": round(mean(row["delta_selected_count"] for row in delta_rows), 3),
        "avg_delta_coverage_score": round(mean(row["delta_coverage_score"] for row in delta_rows), 6),
        "avg_delta_covered_subgoal_count": round(
            mean(row["delta_covered_subgoal_count"] for row in delta_rows), 3
        ),
        "avg_delta_fragment_count_after": round(
            mean(row["delta_fragment_count_after"] for row in delta_rows), 3
        ),
        "avg_delta_estimated_token_cost_after": round(
            mean(row["delta_estimated_token_cost_after"] for row in delta_rows), 6
        ),
        "sample_changed_cases": changed_rows[:5],
    }


def summarize_domain(rows: list[dict]) -> dict:
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)

    default_rows_list = by_variant["default"]
    default_rows = {row["case_id"]: row for row in default_rows_list}
    pass_sequence = tuple(trace["pass_name"] for trace in default_rows_list[0]["pass_trace"])
    stage_labels = build_stage_labels(pass_sequence)

    return {
        "default_metrics": {
            "cases": len(default_rows_list),
            "avg_selected_count": round(
                mean(row["metrics"]["selected_count"] for row in default_rows_list), 3
            ),
            "avg_coverage_score": round(
                mean(row["metrics"]["coverage_score"] for row in default_rows_list), 3
            ),
            "avg_covered_subgoal_count": round(
                mean(row["metrics"]["covered_subgoal_count"] for row in default_rows_list), 3
            ),
            "avg_fragment_count_after": round(
                mean(row["metrics"]["fragment_count_after"] for row in default_rows_list), 3
            ),
            "avg_estimated_token_cost_after": round(
                mean(row["metrics"]["estimated_token_cost_after"] for row in default_rows_list), 3
            ),
        },
        "default_stage_trace_summary": stage_trace_summary(default_rows_list),
        "leave_one_out_summary": [
            compare_variant_to_default(
                default_rows=default_rows,
                variant_rows=by_variant[f"drop_{stage_labels[index]}"],
                pass_sequence=pass_sequence,
                stage_labels=stage_labels,
                stage_index=index,
            )
            for index in range(len(stage_labels))
        ],
        "rows": rows,
    }


def run_analysis(cases: list[dict], pass_sequence: tuple[str, ...]) -> dict:
    stage_labels = build_stage_labels(pass_sequence)
    variants = build_leave_one_out_variants(pass_sequence, stage_labels)
    results_by_domain: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        for variant in variants:
            compiled = compile_case(case, variant["graph_passes"])
            compiled.update(
                {
                    "variant": variant["variant"],
                    "removed_stage_index": variant["removed_stage_index"],
                    "removed_stage_label": variant["removed_stage_label"],
                    "domain": case["domain"],
                }
            )
            results_by_domain[case["domain"]].append(compiled)
    return {
        "stage_labels": stage_labels,
        "default_graph_passes": list(pass_sequence),
        "domains": {
            domain: summarize_domain(domain_rows)
            for domain, domain_rows in sorted(results_by_domain.items())
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="", choices=["", *sorted(DOMAIN_TO_SKILLS_DIR)])
    parser.add_argument("--result-dir", default="")
    parser.add_argument("--cases-json", default="")
    parser.add_argument("--preset", default="default")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    if args.cases_json:
        cases = load_cases_from_json(Path(args.cases_json))
    elif args.result_dir:
        if not args.domain:
            raise SystemExit("--domain is required with --result-dir")
        cases = load_cases_from_results(Path(args.result_dir), args.domain, args.limit)
    else:
        raise SystemExit("Either --result-dir or --cases-json is required.")

    pass_sequence = GRAPH_PASS_PRESETS.get(args.preset.strip().lower())
    if pass_sequence is None:
        raise SystemExit(f"Unsupported preset: {args.preset}")

    output = json.dumps(run_analysis(cases, pass_sequence), indent=2, ensure_ascii=False)
    print(output)
    if args.json_out:
        Path(args.json_out).write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
