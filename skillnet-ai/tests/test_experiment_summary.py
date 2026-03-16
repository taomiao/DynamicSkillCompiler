import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.append(str(EXPERIMENTS_DIR))

from summarize_results import render_markdown, summarize_tree


class ExperimentSummaryTest(unittest.TestCase):
    def test_summarize_tree_aggregates_compiler_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline_dir = root / "dev_smoke_skill_baseline"
            dsc_dir = root / "dev_smoke_skill_dsc"
            baseline_dir.mkdir()
            dsc_dir.mkdir()

            (baseline_dir / "idx_0.json").write_text(
                json.dumps(
                    {
                        "reward": 0,
                        "steps": 20,
                        "task_done": False,
                        "relevant_skill_names": ["a", "b"],
                        "skill_strategy": "skillnet",
                    }
                ),
                encoding="utf-8",
            )
            (dsc_dir / "idx_0.json").write_text(
                json.dumps(
                    {
                        "reward": 1,
                        "steps": 10,
                        "task_done": True,
                        "relevant_skill_names": ["a"],
                        "skill_strategy": "dsc",
                        "compiler_metrics": {
                            "candidate_count": 6,
                            "selected_count": 3,
                            "coverage_score": 0.8,
                            "redundancy_reduction": 0.5,
                            "estimated_token_cost_before": 12.0,
                            "estimated_token_cost_after": 7.0,
                            "subgoal_count": 3,
                            "covered_subgoal_count": 2,
                            "fragment_count_before": 8,
                            "fragment_count_after": 4,
                            "fragment_token_cost_after": 2.5,
                        },
                    }
                ),
                encoding="utf-8",
            )

            summary = summarize_tree(root)

            self.assertEqual(len(summary["runs"]), 2)
            skillnet_run = next(run for run in summary["runs"] if run["skill_mode"] == "skillnet")
            self.assertEqual(skillnet_run["avg_selected_skill_names"], 2.0)
            dsc_run = next(run for run in summary["runs"] if run["skill_mode"] == "dsc")
            self.assertEqual(dsc_run["avg_compiler_selected_count"], 3.0)
            self.assertEqual(dsc_run["avg_compiler_token_reduction"], 5.0)
            self.assertEqual(dsc_run["avg_compiler_subgoal_count"], 3.0)
            self.assertEqual(dsc_run["avg_compiler_fragment_count_after"], 4.0)
            markdown = render_markdown(summary)
            self.assertIn("dev_smoke_skill_dsc", markdown)
            self.assertIn("Covered Subgoals", markdown)


if __name__ == "__main__":
    unittest.main()
