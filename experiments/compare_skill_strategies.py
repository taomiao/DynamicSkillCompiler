import argparse
import json
import sys
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.skill import SkillModule


DEFAULT_CASES = [
    {
        "domain": "alfworld",
        "query": "cool some pan and put it in stoveburner.",
        "skills_dir": "src/skills/alfworld",
        "examples": "src/alfworld/alfworld_overall_procedure_examples.txt",
        "template": "src/alfworld/alfworld_procedure_code_template.py",
    },
    {
        "domain": "alfworld",
        "query": "put some box on dresser.",
        "skills_dir": "src/skills/alfworld",
        "examples": "src/alfworld/alfworld_overall_procedure_examples.txt",
        "template": "src/alfworld/alfworld_procedure_code_template.py",
    },
    {
        "domain": "scienceworld",
        "query": "Your task is to measure the melting point of mercury, which is located around the kitchen. First, focus on the thermometer. Next, focus on the mercury. If the melting point of mercury is above 200.0 degrees celsius, focus on the blue box. If the melting point of mercury is below 200.0 degrees celsius, focus on the orange box. The boxes are located around the kitchen.",
        "skills_dir": "src/skills/scienceworld",
        "examples": "src/scienceworld/scienceworld_overall_procedure_examples.txt",
        "template": "src/scienceworld/scienceworld_procedure_code_template.py",
    },
    {
        "domain": "scienceworld",
        "query": "Your task is to find a(n) living thing. First, focus on the thing. Then, move it to the yellow box in the bedroom.",
        "skills_dir": "src/skills/scienceworld",
        "examples": "src/scienceworld/scienceworld_overall_procedure_examples.txt",
        "template": "src/scienceworld/scienceworld_procedure_code_template.py",
    },
    {
        "domain": "webshop",
        "query": "Find a product with the right size and lowest price, then purchase it.",
        "skills_dir": "src/skills/webshop",
        "examples": "src/webshop/webshop_overall_procedure_examples.txt",
        "template": "src/webshop/webshop_procedure_code_template.py",
    },
    {
        "domain": "webshop",
        "query": "Search for a product, compare attributes, and choose the cheapest acceptable option to buy now.",
        "skills_dir": "src/skills/webshop",
        "examples": "src/webshop/webshop_overall_procedure_examples.txt",
        "template": "src/webshop/webshop_procedure_code_template.py",
    },
]


def token_proxy(text: str) -> int:
    return len(text.split())


def payload_tokens(payload) -> int:
    return sum(token_proxy(content) for _, content in payload)


def make_module(case: dict, strategy: str, args) -> SkillModule:
    return SkillModule(
        skills_dir=case["skills_dir"],
        overall_procedure_examples_path=case["examples"],
        procedure_code_template_path=case["template"],
        model=args.model,
        selection_strategy=strategy,
        compiler_min_relevance=args.compiler_min_relevance,
        compiler_preserve_top_k=args.compiler_preserve_top_k,
        compiler_similar_prune_margin=args.compiler_similar_prune_margin,
        compiler_keep_parent_if_better_by=args.compiler_keep_parent_if_better_by,
        compiler_coverage_weight=args.compiler_coverage_weight,
        compiler_quality_weight=args.compiler_quality_weight,
        compiler_cost_weight=args.compiler_cost_weight,
        compiler_latency_weight=args.compiler_latency_weight,
        compiler_top_k=args.compiler_top_k,
    )


def evaluate_case(case: dict, args) -> dict:
    baseline = make_module(case, "baseline", args)
    baseline_names = baseline.retrieve_relevant_skills(case["query"])
    baseline_payload = baseline._build_full_skill_payload(baseline_names)

    dsc = make_module(case, "dsc", args)
    dsc_names = dsc.retrieve_relevant_skills(case["query"])
    dsc_payload, _ = dsc._build_compiled_skill_payload(dsc_names)
    metrics = dsc.last_compilation.metrics

    baseline_tokens = payload_tokens(baseline_payload)
    dsc_tokens = payload_tokens(dsc_payload)
    return {
        "domain": case["domain"],
        "query": case["query"],
        "baseline_selected": len(baseline_names),
        "dsc_selected": len(dsc_names),
        "baseline_payload_tokens": baseline_tokens,
        "dsc_payload_tokens": dsc_tokens,
        "payload_token_reduction": baseline_tokens - dsc_tokens,
        "payload_token_reduction_ratio": 1 - (dsc_tokens / max(baseline_tokens, 1)),
        "baseline_skill_names": baseline_names,
        "dsc_skill_names": dsc_names,
        "dsc_subgoals": [subgoal.description for subgoal in dsc.last_compilation.subgoals],
        "dsc_metrics": metrics.__dict__,
    }


def summarize(rows: list[dict]) -> dict:
    by_domain = {}
    for row in rows:
        by_domain.setdefault(row["domain"], []).append(row)

    domain_summaries = []
    for domain, items in sorted(by_domain.items()):
        domain_summaries.append(
            {
                "domain": domain,
                "cases": len(items),
                "avg_baseline_selected": mean(item["baseline_selected"] for item in items),
                "avg_dsc_selected": mean(item["dsc_selected"] for item in items),
                "avg_baseline_payload_tokens": mean(item["baseline_payload_tokens"] for item in items),
                "avg_dsc_payload_tokens": mean(item["dsc_payload_tokens"] for item in items),
                "avg_payload_token_reduction": mean(item["payload_token_reduction"] for item in items),
                "avg_payload_token_reduction_ratio": mean(item["payload_token_reduction_ratio"] for item in items),
                "avg_subgoal_count": mean(item["dsc_metrics"]["subgoal_count"] for item in items),
                "avg_covered_subgoal_count": mean(item["dsc_metrics"]["covered_subgoal_count"] for item in items),
                "avg_fragment_count_after": mean(item["dsc_metrics"]["fragment_count_after"] for item in items),
                "avg_fragment_token_cost_after": mean(item["dsc_metrics"]["fragment_token_cost_after"] for item in items),
            }
        )

    overall = {
        "cases": len(rows),
        "avg_baseline_payload_tokens": mean(row["baseline_payload_tokens"] for row in rows),
        "avg_dsc_payload_tokens": mean(row["dsc_payload_tokens"] for row in rows),
        "avg_payload_token_reduction": mean(row["payload_token_reduction"] for row in rows),
        "avg_payload_token_reduction_ratio": mean(row["payload_token_reduction_ratio"] for row in rows),
    }
    return {"overall": overall, "by_domain": domain_summaries, "cases": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="o4-mini")
    parser.add_argument("--cases-json", default="", help="Optional JSON file for custom cases")
    parser.add_argument("--json-out", default="", help="Optional output path")
    parser.add_argument("--compiler_min_relevance", type=float, default=0.15)
    parser.add_argument("--compiler_preserve_top_k", type=int, default=3)
    parser.add_argument("--compiler_similar_prune_margin", type=float, default=0.15)
    parser.add_argument("--compiler_keep_parent_if_better_by", type=float, default=0.10)
    parser.add_argument("--compiler_coverage_weight", type=float, default=0.65)
    parser.add_argument("--compiler_quality_weight", type=float, default=0.20)
    parser.add_argument("--compiler_cost_weight", type=float, default=0.10)
    parser.add_argument("--compiler_latency_weight", type=float, default=0.05)
    parser.add_argument("--compiler_top_k", type=int, default=6)
    args = parser.parse_args()

    cases = DEFAULT_CASES
    if args.cases_json:
        cases = json.loads(Path(args.cases_json).read_text(encoding="utf-8"))

    rows = [evaluate_case(case, args) for case in cases]
    summary = summarize(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
