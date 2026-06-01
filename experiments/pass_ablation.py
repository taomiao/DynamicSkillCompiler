import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent
DSC_SRC = ROOT.parent / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(DSC_SRC) not in sys.path:
    sys.path.append(str(DSC_SRC))

from dynamic_skill_compiler import (
    CompilerConfig,
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


def resolve_variant(name: str) -> dict:
    normalized = name.strip().lower()
    if normalized == "default_no_bias":
        return {
            "name": normalized,
            "graph_passes": GRAPH_PASS_PRESETS["default"],
            "adaptive_workflow_bias": False,
        }
    if normalized in GRAPH_PASS_PRESETS:
        return {
            "name": normalized,
            "graph_passes": GRAPH_PASS_PRESETS[normalized],
            "adaptive_workflow_bias": True,
        }
    raise ValueError(f"Unsupported variant: {name}")


def compile_case(case: dict, variant: dict) -> dict:
    domain = case["domain"]
    retriever = LocalSkillLibraryRetriever(str(DOMAIN_TO_SKILLS_DIR[domain]))
    compiler = DynamicSkillCompiler(
        retriever=retriever,
        config=CompilerConfig(
            graph_passes=variant["graph_passes"],
            adaptive_workflow_bias=variant["adaptive_workflow_bias"],
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
        "variant": variant["name"],
        "case_id": case["case_id"],
        "task_name": case.get("task_name", ""),
        "domain": domain,
        "query": case["query"],
        "selected_skills": [item.asset.name for item in compiled.compiled_skills],
        "execution_order": compiled.execution_order,
        "metrics": compiled.metrics.__dict__,
        "notes": compiled.notes,
        "pass_trace": [
            {
                "pass_name": trace.pass_name,
                "added": trace.added,
                "removed": trace.removed,
                "dropped_delta": trace.dropped_delta,
                "after_selected": trace.after_selected,
            }
            for trace in compiled.pass_traces
        ],
    }


def aggregate(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)

    variants = []
    for variant_name, items in sorted(grouped.items()):
        skill_counter = Counter()
        pass_counter = defaultdict(list)
        for item in items:
            skill_counter.update(item["selected_skills"])
            for trace in item["pass_trace"]:
                pass_counter[trace["pass_name"]].append(trace)

        pass_summary = []
        for pass_name, traces in pass_counter.items():
            pass_summary.append(
                {
                    "pass_name": pass_name,
                    "changed_cases": sum(
                        1
                        for trace in traces
                        if trace["added"] or trace["removed"] or trace["dropped_delta"]
                    ),
                    "avg_added": mean(len(trace["added"]) for trace in traces),
                    "avg_removed": mean(len(trace["removed"]) for trace in traces),
                    "avg_selected_after": mean(len(trace["after_selected"]) for trace in traces),
                }
            )

        variants.append(
            {
                "variant": variant_name,
                "cases": len(items),
                "avg_selected_count": mean(item["metrics"]["selected_count"] for item in items),
                "avg_coverage_score": mean(item["metrics"]["coverage_score"] for item in items),
                "avg_subgoal_count": mean(item["metrics"]["subgoal_count"] for item in items),
                "avg_covered_subgoal_count": mean(
                    item["metrics"]["covered_subgoal_count"] for item in items
                ),
                "avg_fragment_count_after": mean(
                    item["metrics"]["fragment_count_after"] for item in items
                ),
                "top_selected_skills": skill_counter.most_common(8),
                "pass_summary": sorted(pass_summary, key=lambda item: item["pass_name"]),
            }
        )

    return {
        "variants": variants,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="scienceworld", choices=sorted(DOMAIN_TO_SKILLS_DIR))
    parser.add_argument("--result-dir", default="")
    parser.add_argument("--cases-json", default="")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--variants",
        default="default,coverage_first,conservative,minimal,default_no_bias",
    )
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    if args.cases_json:
        cases = load_cases_from_json(Path(args.cases_json))
    elif args.result_dir:
        cases = load_cases_from_results(Path(args.result_dir), args.domain, args.limit)
    else:
        raise SystemExit("Either --result-dir or --cases-json is required.")

    variant_defs = [resolve_variant(name) for name in args.variants.split(",") if name.strip()]
    rows = []
    for case in cases:
        for variant in variant_defs:
            rows.append(compile_case(case, variant))

    summary = aggregate(rows)
    output = json.dumps(summary, indent=2, ensure_ascii=False)
    print(output)
    if args.json_out:
        Path(args.json_out).write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
