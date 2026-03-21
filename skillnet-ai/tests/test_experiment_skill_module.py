import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.append(str(EXPERIMENTS_DIR))

from src.skill import SkillModule


class ExperimentSkillModuleTest(unittest.TestCase):
    def test_dsc_strategy_compiles_local_skill_subset(self):
        module = SkillModule(
            skills_dir=str(ROOT_DIR / "experiments" / "src" / "skills" / "webshop"),
            overall_procedure_examples_path=str(
                ROOT_DIR / "experiments" / "src" / "webshop" / "webshop_overall_procedure_examples.txt"
            ),
            procedure_code_template_path=str(
                ROOT_DIR / "experiments" / "src" / "webshop" / "webshop_procedure_code_template.py"
            ),
            selection_strategy="dsc",
            compiler_min_relevance=0.3,
        )

        fake_response = (
            "<Relevant_Skill_Names>"
            "[\"webshop-price-checker\", \"webshop-product-selector\", \"webshop-purchase-initiator\"]"
            "</Relevant_Skill_Names>"
        )
        with patch("src.skill.get_llm_response", return_value=fake_response):
            selected = module.retrieve_relevant_skills(
                "Find a product with the right size and lowest price, then purchase it."
            )

        self.assertTrue(selected)
        self.assertIsNotNone(module.last_compilation)
        self.assertLessEqual(len(selected), module.compiler_top_k)
        self.assertGreaterEqual(module.last_compilation.metrics.coverage_score, 0.0)
        self.assertLessEqual(
            module.last_compilation.metrics.estimated_token_cost_after,
            module.last_compilation.metrics.estimated_token_cost_before,
        )
        self.assertTrue(set(selected).issubset({
            "webshop-price-checker",
            "webshop-product-selector",
            "webshop-purchase-initiator",
        }))

    def test_dsc_uses_broader_llm_seed_pool_than_baseline(self):
        module = SkillModule(
            skills_dir=str(ROOT_DIR / "experiments" / "src" / "skills" / "scienceworld"),
            selection_strategy="dsc",
        )

        captured = {}

        def fake_llm(messages, is_string, model):
            captured["prompt"] = messages
            return (
                "<Relevant_Skill_Names>"
                "[\"scienceworld-temperature-measurer\", \"scienceworld-threshold-evaluator\", "
                "\"scienceworld-conditional-focus-executor\"]"
                "</Relevant_Skill_Names>"
            )

        with patch("src.skill.get_llm_response", side_effect=fake_llm):
            module.retrieve_relevant_skills("Measure mercury and focus on the correct box.")

        prompt_text = captured["prompt"][1]["content"]
        self.assertIn("Target 8 skills", prompt_text)

    def test_conductivity_quality_first_selection_keeps_core_stack(self):
        module = SkillModule(
            skills_dir=str(ROOT_DIR / "experiments" / "src" / "skills" / "scienceworld"),
            selection_strategy="dsc",
        )

        selected = module._select_quality_first_skill_names(
            "Your task is to determine if unknown substance S is electrically conductive.",
            [
                "scienceworld-object-locator",
                "scienceworld-object-focuser",
                "scienceworld-conductivity-tester",
                "scienceworld-object-classifier",
            ],
            [
                "scienceworld-conditional-placer",
                "scienceworld-circuit-builder",
                "scienceworld-object-locator",
                "scienceworld-object-focuser",
                "scienceworld-conductivity-tester",
                "scienceworld-object-classifier",
            ],
        )

        self.assertIn("scienceworld-object-locator", selected)
        self.assertIn("scienceworld-object-focuser", selected)
        self.assertIn("scienceworld-conductivity-tester", selected)
        self.assertIn("scienceworld-object-classifier", selected)

    def test_conductivity_static_code_uses_drop_not_table(self):
        module = SkillModule(
            skills_dir=str(ROOT_DIR / "experiments" / "src" / "skills" / "scienceworld"),
            selection_strategy="dsc",
        )

        code = module._scienceworld_static_procedure_code(
            "Your task is to determine if unknown substance S is electrically conductive. "
            "If it is electrically conductive, place it in the orange box. "
            "If it is electrically nonconductive, place it in the yellow box.",
            "placeholder guidance",
        )

        self.assertIn('run_action(f"drop {target_object}")', code)
        self.assertNotIn("move unknown substance S to table", code)
        self.assertIn("target_box = conductive_box if bulb_is_on else nonconductive_box", code)
        self.assertIn("source_room = 'workshop'", code)

    def test_conductivity_static_code_fetches_object_before_workshop(self):
        module = SkillModule(
            skills_dir=str(ROOT_DIR / "experiments" / "src" / "skills" / "scienceworld"),
            selection_strategy="dsc",
        )

        code = module._scienceworld_static_procedure_code(
            "Your task is to determine if paper clip is electrically conductive. "
            "The paper clip is located around the kitchen. "
            "If it is electrically conductive, place it in the yellow box. "
            "If it is electrically nonconductive, place it in the purple box.",
            "placeholder guidance",
        )

        self.assertIn("source_room = 'kitchen'", code)
        self.assertIn('run_action(f"teleport to {source_room}")', code)
        self.assertIn('run_action(f"pick up {target_object}")', code)
        self.assertLess(
            code.index('run_action(f"pick up {target_object}")'),
            code.index('workshop_observation = run_action("teleport to workshop")'),
        )

    def test_temperature_quality_first_selection_prefers_stable_measurement_stack(self):
        module = SkillModule(
            skills_dir=str(ROOT_DIR / "experiments" / "src" / "skills" / "scienceworld"),
            selection_strategy="dsc",
        )

        selected = module._select_quality_first_skill_names(
            "Your task is to measure the temperature of unknown substance B. "
            "First, focus on the thermometer. Next, focus on the unknown substance B. "
            "If it is above 50 degrees celsius, place it in the green box.",
            [
                "scienceworld-object-locator",
                "scienceworld-task-focuser",
                "scienceworld-temperature-measurer",
                "scienceworld-conditional-box-placer",
            ],
            [
                "scienceworld-room-navigator",
                "scienceworld-object-locator",
                "scienceworld-task-focuser",
                "scienceworld-temperature-measurer",
                "scienceworld-conditional-box-placer",
                "scienceworld-room-scanner",
            ],
        )

        self.assertIn("scienceworld-temperature-measurer", selected)
        self.assertIn("scienceworld-conditional-box-placer", selected)
        self.assertIn("scienceworld-object-locator", selected)

    def test_growth_quality_first_selection_keeps_core_growth_workflow(self):
        module = SkillModule(
            skills_dir=str(ROOT_DIR / "experiments" / "src" / "skills" / "scienceworld"),
            selection_strategy="dsc",
        )

        selected = module._select_quality_first_skill_names(
            "Your task is to grow a cherry plant from seed. Seeds can be found in the bedroom. "
            "First, focus on a seed. Then, make changes to the environment that grow the plant.",
            [
                "scienceworld-room-navigator",
                "scienceworld-object-focuser",
                "scienceworld-planting-coordinator",
                "controlled-waiting",
                "scienceworld-growth-focuser",
            ],
            [
                "scienceworld-room-navigator",
                "scienceworld-object-focuser",
                "scienceworld-pot-preparer",
                "scienceworld-planting-coordinator",
                "scienceworld-liquid-filler",
                "controlled-waiting",
                "scienceworld-growth-focuser",
                "scienceworld-ambiguous-action-resolution",
            ],
        )

        self.assertIn("scienceworld-planting-coordinator", selected)
        self.assertIn("scienceworld-pot-preparer", selected)
        self.assertIn("scienceworld-liquid-filler", selected)
        self.assertIn("scienceworld-growth-focuser", selected)

    def test_runtime_recompile_merge_preserves_previous_workflow_backbone(self):
        module = SkillModule(
            skills_dir=str(ROOT_DIR / "experiments" / "src" / "skills" / "scienceworld"),
            selection_strategy="dsc",
        )
        previous = [
            "scienceworld-room-navigator",
            "scienceworld-object-focuser",
            "scienceworld-pot-preparer",
            "scienceworld-planting-coordinator",
            "soil-extraction",
            "scienceworld-growth-focuser",
            "controlled-waiting",
        ]
        refreshed = ["scienceworld-liquid-filler"]
        module.last_compilation = SimpleNamespace(
            compiled_skills=[
                SimpleNamespace(asset=SimpleNamespace(name=name))
                for name in refreshed + previous
            ]
        )

        with patch.object(
            module,
            "_adaptive_compiler_config",
            return_value=SimpleNamespace(
                profile_name="workflow",
                max_selected_skills=0,
                preserve_top_k=4,
            ),
        ):
            merged = module.merge_runtime_recompile_skill_names(
                "[Runtime Recompile]\nRepair the failed planting action.",
                previous,
                refreshed,
                {"reason": "action_failure", "task_reward": 38.0},
            )

        self.assertEqual(merged[0], "scienceworld-liquid-filler")
        self.assertIn("scienceworld-pot-preparer", merged)
        self.assertIn("scienceworld-planting-coordinator", merged)
        self.assertIn("scienceworld-growth-focuser", merged)
        self.assertGreaterEqual(len(merged), 5)


if __name__ == "__main__":
    unittest.main()
