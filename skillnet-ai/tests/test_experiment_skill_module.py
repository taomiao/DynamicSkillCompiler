import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.append(str(EXPERIMENTS_DIR))

from src.skill import SkillModule
from src.scienceworld.prompts.system_prompt import scienceworld_system_prompt
from src.runtime_recompile import (
    RuntimeRecompileController,
    RuntimeRecompileEnvProxy,
    RuntimeSkillRecompileRequested,
    execute_compiled_procedure,
)
from skillnet_ai.compiler import (
    CompilationMetrics,
    CompiledSkill,
    CompiledSkillPackage,
    GRAPH_PASS_PRESETS,
    QueryPlan,
    SkillAsset,
    SkillFragment,
    SkillGraph,
)

try:
    from scienceworld_run import parse_action
except Exception:
    parse_action = None


class ExperimentSkillModuleTest(unittest.TestCase):
    def _fake_compilation(self, module, skill_names):
        compiled_skills = []
        for skill_name in skill_names:
            compiled = module._fallback_compiled_skill(skill_name)
            if compiled is None:
                description = module.metadata.get(skill_name, {}).get("description", skill_name)
                compiled = CompiledSkill(
                    asset=SkillAsset(
                        skill_id=skill_name,
                        name=skill_name,
                        description=description,
                        capabilities={
                            token
                            for token in description.lower().replace("-", " ").split()
                            if len(token) > 2
                        },
                        instructions=[description],
                    ),
                    selected_fragments=[],
                    assigned_subgoals=[],
                    localized_instructions=module._fallback_localized_instructions(skill_name),
                    utility_score=0.0,
                    selected_reason="test_fixture",
                )
            compiled_skills.append(compiled)
        return CompiledSkillPackage(
            query_plan=QueryPlan(
                raw_query="task",
                normalized_query="task",
                keyword_query="task",
                semantic_queries=[],
                intents=[],
                required_capabilities={"focus", "measure", "place"},
                optional_capabilities=set(),
            ),
            subgoals=[],
            graph=SkillGraph(
                skills={item.asset.skill_id: item.asset for item in compiled_skills},
                relations=[],
            ),
            compiled_skills=compiled_skills,
            execution_order=[item.asset.name for item in compiled_skills],
            metrics=CompilationMetrics(
                candidate_count=len(compiled_skills),
                selected_count=len(compiled_skills),
                coverage_score=0.20,
                subgoal_count=1,
                covered_subgoal_count=1,
                estimated_token_cost_before=50.0,
                estimated_token_cost_after=35.0,
                estimated_execution_cost_before=8.0,
                estimated_execution_cost_after=5.0,
            ),
            dropped_skills={},
            notes=[],
        )

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

    def test_skill_module_accepts_graph_pass_preset_name(self):
        module = SkillModule(
            skills_dir=str(ROOT_DIR / "experiments" / "src" / "skills" / "scienceworld"),
            selection_strategy="dsc",
            compiler_graph_passes="minimal",
        )

        self.assertEqual(module.compiler_graph_passes, GRAPH_PASS_PRESETS["minimal"])

    def test_skill_module_accepts_slim_default_graph_pass_preset_name(self):
        module = SkillModule(
            skills_dir=str(ROOT_DIR / "experiments" / "src" / "skills" / "scienceworld"),
            selection_strategy="dsc",
            compiler_graph_passes="slim_default",
        )

        self.assertEqual(module.compiler_graph_passes, GRAPH_PASS_PRESETS["slim_default"])

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

    def test_dsc_shared_retrieval_reuses_single_candidate_pool_call(self):
        module = SkillModule(
            skills_dir=str(ROOT_DIR / "experiments" / "src" / "skills" / "scienceworld"),
            selection_strategy="dsc",
        )
        fake_compilation = self._fake_compilation(
            module,
            [
                "scienceworld-object-locator",
                "scienceworld-object-focuser",
                "scienceworld-temperature-measurer",
            ],
        )

        llm_calls = []

        def fake_llm(messages, is_string, model):
            llm_calls.append(messages[1]["content"])
            return (
                "<Relevant_Skill_Names>"
                "[\"scienceworld-object-locator\", \"scienceworld-object-focuser\", "
                "\"scienceworld-temperature-measurer\"]"
                "</Relevant_Skill_Names>"
            )

        with patch.object(module, "_compile_task", return_value=fake_compilation):
            with patch("src.skill.get_llm_response", side_effect=fake_llm):
                selected = module.retrieve_relevant_skills(
                    "Measure the temperature of mercury, then focus on the correct box."
                )

        self.assertEqual(len(llm_calls), 1)
        self.assertIn("Target 8 skills", llm_calls[0])
        self.assertEqual(selected[:3], fake_compilation.execution_order)
        self.assertIn("scienceworld-threshold-evaluator", selected)
        self.assertEqual(module.last_payload_strategy, "quality_first")

    def test_dsc_retrieval_timeout_falls_back_to_local_compilation(self):
        module = SkillModule(
            skills_dir=str(ROOT_DIR / "experiments" / "src" / "skills" / "scienceworld"),
            selection_strategy="dsc",
        )
        fake_compilation = self._fake_compilation(
            module,
            [
                "scienceworld-object-locator",
                "scienceworld-object-focuser",
                "scienceworld-animal-identifier",
            ],
        )
        fake_compilation.query_plan.raw_query = "Find the animal and focus on it."
        fake_compilation.metrics.coverage_score = 0.91
        fake_compilation.metrics.subgoal_count = 2
        fake_compilation.metrics.covered_subgoal_count = 2

        with patch.object(module, "_compile_task", return_value=fake_compilation):
            with patch("src.skill.get_llm_response", side_effect=RuntimeError("Request timed out.")):
                selected = module.retrieve_relevant_skills("Find the animal and focus on it.")

        self.assertEqual(
            selected,
            [
                "scienceworld-object-locator",
                "scienceworld-object-focuser",
                "scienceworld-animal-identifier",
            ],
        )
        self.assertEqual(module.last_payload_strategy, "quality_first")
        self.assertTrue(
            any("fell back to compiler-local selection" in note for note in module.last_compilation.notes)
        )

    def test_observation_heavy_query_uses_reference_style_payload(self):
        module = SkillModule(
            skills_dir=str(ROOT_DIR / "experiments" / "src" / "skills" / "scienceworld"),
            selection_strategy="dsc",
        )
        fake_compilation = self._fake_compilation(
            module,
            [
                "scienceworld-object-locator",
                "scienceworld-object-focuser",
                "scienceworld-animal-identifier",
            ],
        )
        fake_compilation.query_plan.raw_query = "Find the animal and focus on it."
        fake_compilation.metrics.coverage_score = 0.92
        fake_compilation.metrics.subgoal_count = 2
        fake_compilation.metrics.covered_subgoal_count = 2

        with patch.object(module, "_compile_task", return_value=fake_compilation):
            with patch(
                "src.skill.get_llm_response",
                return_value=(
                    "<Relevant_Skill_Names>"
                    "[\"scienceworld-object-locator\", \"scienceworld-object-focuser\", "
                    "\"scienceworld-animal-identifier\"]"
                    "</Relevant_Skill_Names>"
                ),
            ):
                selected = module.retrieve_relevant_skills("Find the animal and focus on it.")

        self.assertEqual(module.last_payload_strategy, "quality_first")
        self.assertEqual(
            selected,
            [
                "scienceworld-object-locator",
                "scienceworld-object-focuser",
                "scienceworld-animal-identifier",
            ],
        )
        self.assertTrue(
            any("Reference-style payload activated" in note for note in module.last_compilation.notes)
        )

    def test_phase_change_family_priority_skills_are_included_in_compile_seed_pool(self):
        module = SkillModule(
            skills_dir=str(ROOT_DIR / "experiments" / "src" / "skills" / "scienceworld"),
            selection_strategy="dsc",
        )

        captured = {}

        def fake_compile(task, seed_skill_names=None):
            captured["seed_skill_names"] = list(seed_skill_names or [])
            return self._fake_compilation(
                module,
                [
                    "scienceworld-task-focuser",
                    "scienceworld-heating-apparatus-setup",
                ],
            )

        with patch.object(
            module,
            "_llm_retrieve_relevant_skill_names",
            side_effect=[
                ["scienceworld-task-focuser"],
                ["scienceworld-task-focuser"],
            ],
        ), patch.object(module, "_compile_task", side_effect=fake_compile):
            module.retrieve_relevant_skills(
                "Your task is to boil lead. For compounds without a boiling point, combusting the substance is also acceptable. "
                "First, focus on the substance. Then, take actions that will cause it to change its state of matter."
            )

        self.assertIn("scienceworld-substance-preparator", captured["seed_skill_names"])
        self.assertIn("scienceworld-heating-apparatus-setup", captured["seed_skill_names"])
        self.assertIn("scienceworld-process-monitor", captured["seed_skill_names"])
        self.assertIn("controlled-waiting", captured["seed_skill_names"])

    def test_conductivity_quality_first_selection_preserves_canonical_four(self):
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

        self.assertEqual(
            selected,
            [
                "scienceworld-object-locator",
                "scienceworld-object-focuser",
                "scienceworld-conductivity-tester",
                "scienceworld-object-classifier",
            ],
        )

    def test_generate_overall_procedure_uses_compiled_payload_when_no_fallback_is_active(self):
        module = SkillModule(
            skills_dir=str(ROOT_DIR / "experiments" / "src" / "skills" / "scienceworld"),
            selection_strategy="dsc",
        )
        module.last_compilation = self._fake_compilation(
            module,
            ["scienceworld-object-locator"],
        )
        module.last_payload_strategy = "compiled"
        module.last_quality_reference_skill_names = ["scienceworld-object-locator"]

        captured = {}

        def fake_llm(messages, is_string, model):
            captured["prompt"] = messages[1]["content"]
            return (
                "<Analysis>ok</Analysis>"
                "<Overall_Procedure>Follow the compiled skill.</Overall_Procedure>"
            )

        with patch("src.skill.get_llm_response", side_effect=fake_llm):
            procedure = module.generate_overall_procedure(
                "Locate the object and focus on it.",
                ["scienceworld-object-locator"],
            )

        self.assertEqual(procedure, "Follow the compiled skill.")
        self.assertIn("=== Compiled Skill: scienceworld-object-locator ===", captured["prompt"])
        self.assertIn("[Compressed SKILL.md]", captured["prompt"])
        self.assertNotIn("[Additional References]", captured["prompt"])

    def test_high_coverage_compiled_payload_prefers_fragments_over_full_skill_markdown(self):
        module = SkillModule(
            skills_dir=str(ROOT_DIR / "experiments" / "src" / "skills" / "scienceworld"),
            selection_strategy="dsc",
        )

        compiled_skill = CompiledSkill(
            asset=SkillAsset(
                skill_id="scienceworld-process-monitor",
                name="scienceworld-process-monitor",
                description="Monitor an active process and inspect state changes.",
                capabilities={"monit", "state", "chang"},
                instructions=["look at apparatus", "examine substance"],
            ),
            selected_fragments=[
                SkillFragment(
                    fragment_id="monitor-1",
                    skill_id="scienceworld-process-monitor",
                    title="Monitor active heater",
                    content="Use `look at <APPARATUS>` to verify it is turned on before examining the substance.",
                    capabilities={"monit", "apparatu"},
                )
            ],
            assigned_subgoals=["subgoal-2"],
            localized_instructions=[
                "Use `look at <APPARATUS>` before each `examine <SUBSTANCE>` check.",
                "Keep the monitoring loop short and grounded in the latest observation.",
            ],
            utility_score=0.9,
            selected_reason="matched required capabilities monit, apparatu; utility=0.900",
        )
        module.last_compilation = CompiledSkillPackage(
            query_plan=QueryPlan(
                raw_query="Boil mercury after focusing on it.",
                normalized_query="boil mercury after focusing on it",
                keyword_query="boil mercury focus",
                semantic_queries=[],
                intents=["transform"],
                required_capabilities={"boil", "mercury", "focu"},
                optional_capabilities={"monit", "apparatu"},
            ),
            subgoals=[],
            graph=SkillGraph(
                skills={compiled_skill.asset.skill_id: compiled_skill.asset},
                relations=[],
            ),
            compiled_skills=[compiled_skill],
            execution_order=[compiled_skill.asset.name],
            metrics=CompilationMetrics(
                candidate_count=3,
                selected_count=1,
                coverage_score=0.82,
                subgoal_count=2,
                covered_subgoal_count=2,
                fragment_count_before=2,
                fragment_count_after=1,
                estimated_token_cost_before=40.0,
                estimated_token_cost_after=18.0,
                fragment_token_cost_after=6.0,
            ),
            dropped_skills={},
            notes=[],
        )
        module.last_payload_strategy = "compiled"

        payload, summary = module._build_compiled_skill_payload(
            ["scienceworld-process-monitor"]
        )

        self.assertEqual(len(payload), 1)
        self.assertIn("[Selected Fragments]", payload[0][1])
        self.assertIn("[Source Description]", payload[0][1])
        self.assertNotIn("[Compressed SKILL.md]", payload[0][1])
        self.assertIn("Monitor active heater", payload[0][1])
        self.assertIn("Prefer the localized instructions and selected fragments", summary)

    def test_quality_first_payload_expands_to_full_source_when_coverage_is_low(self):
        module = SkillModule(
            skills_dir=str(ROOT_DIR / "experiments" / "src" / "skills" / "scienceworld"),
            selection_strategy="dsc",
        )
        module.last_compilation = self._fake_compilation(
            module,
            ["scienceworld-object-locator"],
        )
        module.last_compilation.metrics.coverage_score = 0.20
        module.last_compilation.metrics.subgoal_count = 3
        module.last_compilation.metrics.covered_subgoal_count = 1
        module.last_payload_strategy = "quality_first"
        module.last_quality_reference_skill_names = ["scienceworld-object-locator"]

        captured = {}

        def fake_llm(messages, is_string, model):
            captured["prompt"] = messages[1]["content"]
            return (
                "<Analysis>ok</Analysis>"
                "<Overall_Procedure>Use the full source skill.</Overall_Procedure>"
            )

        with patch("src.skill.get_llm_response", side_effect=fake_llm):
            procedure = module.generate_overall_procedure(
                "Locate the object and classify it.",
                ["scienceworld-object-locator"],
            )

        self.assertEqual(procedure, "Use the full source skill.")
        self.assertIn("Dynamic Skill Compiler Recovery Summary", captured["prompt"])
        self.assertIn("[File: SKILL.md]", captured["prompt"])
        self.assertNotIn("[Compressed SKILL.md]", captured["prompt"])

    def test_quality_first_payload_expands_to_full_source_for_observation_heavy_query(self):
        module = SkillModule(
            skills_dir=str(ROOT_DIR / "experiments" / "src" / "skills" / "scienceworld"),
            selection_strategy="dsc",
        )
        module.last_compilation = self._fake_compilation(
            module,
            ["scienceworld-object-locator"],
        )
        module.last_compilation.query_plan.raw_query = "Find the animal and focus on it."
        module.last_compilation.metrics.coverage_score = 0.95
        module.last_compilation.metrics.subgoal_count = 2
        module.last_compilation.metrics.covered_subgoal_count = 2
        module.last_payload_strategy = "quality_first"
        module.last_quality_reference_skill_names = ["scienceworld-object-locator"]

        captured = {}

        def fake_llm(messages, is_string, model):
            captured["prompt"] = messages[1]["content"]
            return (
                "<Analysis>ok</Analysis>"
                "<Overall_Procedure>Use the full source skill.</Overall_Procedure>"
            )

        with patch("src.skill.get_llm_response", side_effect=fake_llm):
            procedure = module.generate_overall_procedure(
                "Find the animal and focus on it.",
                ["scienceworld-object-locator"],
            )

        self.assertEqual(procedure, "Use the full source skill.")
        self.assertIn("Dynamic Skill Compiler Recovery Summary", captured["prompt"])
        self.assertIn("[File: SKILL.md]", captured["prompt"])
        self.assertNotIn("[Compressed SKILL.md]", captured["prompt"])

    def test_generate_overall_procedure_code_sanitizes_wrapped_output(self):
        module = SkillModule(
            skills_dir=str(ROOT_DIR / "experiments" / "src" / "skills" / "alfworld"),
            selection_strategy="dsc",
        )

        wrapped_response = """
` tag.
</Analysis>

<Overall_Procedure_Code>
#v2
def overall_procedure_code(env, llm, model, process_ob, parse_action, messages=[], max_steps=30):
    return messages, False, 0, 0
</Overall_Procedure_Code>
"""

        with patch("src.skill.get_llm_response", return_value=wrapped_response):
            code = module.generate_overall_procedure_code("task", "procedure")

        self.assertTrue(code.startswith("#v2"))
        self.assertIn("def overall_procedure_code", code)
        self.assertNotIn("<Overall_Procedure_Code>", code)
        self.assertNotIn("</Analysis>", code)

    def test_generate_overall_procedure_sanitizes_wrapped_output(self):
        module = SkillModule(
            skills_dir=str(ROOT_DIR / "experiments" / "src" / "skills" / "alfworld"),
            selection_strategy="dsc",
        )
        module.last_compilation = self._fake_compilation(
            module,
            ["alfworld-goal-interpreter"],
        )
        module.last_payload_strategy = "compiled"

        wrapped_response = """
` artifact
</Analysis>

<Overall_Procedure>
Phase 1: Look around.
Phase 2: Pick up the target.
</Overall_Procedure>
"""

        with patch("src.skill.get_llm_response", return_value=wrapped_response):
            procedure = module.generate_overall_procedure(
                "Put the object in the receptacle.",
                ["alfworld-goal-interpreter"],
            )

        self.assertIn("Phase 1: Look around.", procedure)
        self.assertIn("Phase 2: Pick up the target.", procedure)
        self.assertNotIn("<Overall_Procedure>", procedure)
        self.assertNotIn("</Analysis>", procedure)

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

    def test_conductivity_static_code_handles_tight_step_budget_without_crashing(self):
        module = SkillModule(
            skills_dir=str(ROOT_DIR / "experiments" / "src" / "skills" / "scienceworld"),
            selection_strategy="dsc",
        )

        code = module._scienceworld_static_procedure_code(
            "Your task is to determine if unknown substance S is electrically conductive. "
            "The unknown substance S is located around the workshop. "
            "First, focus on the unknown substance S. "
            "If it is electrically conductive, place it in the orange box. "
            "If it is electrically nonconductive, place it in the yellow box.",
            "placeholder guidance",
        )

        namespace = {}
        exec(code, namespace)
        overall_procedure_code = namespace["overall_procedure_code"]

        class FakeEnv:
            def __init__(self):
                self.actions = []

            def step(self, action):
                self.actions.append(action)
                observation_map = {
                    "teleport to workshop": "You teleport to the workshop.",
                    "look around": (
                        "This room is called the workshop. In it, you see: "
                        "a battery, a black wire, a blue wire, a yellow wire, "
                        "a green light bulb, which is off, "
                        "a orange box (containing nothing), "
                        "unknown substance S, a yellow box (containing nothing)."
                    ),
                    "pick up unknown substance S": "You move the unknown substance S to the inventory.",
                    "focus on unknown substance S": "You focus on the unknown substance S.",
                    "connect battery anode to black wire terminal 1": (
                        "You connect the battery anode to the black wire terminal 1."
                    ),
                }
                return observation_map.get(action, ""), 0, False, {"score": 0}

        fake_env = FakeEnv()
        messages, task_done, task_reward, current_steps = overall_procedure_code(
            env=fake_env,
            llm=None,
            model="o4-mini",
            parse_action=None,
            messages=[],
            max_steps=5,
        )

        self.assertFalse(task_done)
        self.assertEqual(task_reward, 0)
        self.assertEqual(current_steps, 5)
        self.assertEqual(
            fake_env.actions,
            [
                "teleport to workshop",
                "look around",
                "pick up unknown substance S",
                "focus on unknown substance S",
                "connect battery anode to black wire terminal 1",
            ],
        )
        self.assertTrue(messages)

    def test_growth_static_code_uses_direct_seed_outside_greenhouse_route(self):
        module = SkillModule(
            skills_dir=str(ROOT_DIR / "experiments" / "src" / "skills" / "scienceworld"),
            selection_strategy="dsc",
        )

        code = module._scienceworld_growth_static_procedure_code(
            "Your task is to grow a cherry plant from seed. Seeds can be found in the bedroom. "
            "First, focus on a seed. Then, make changes to the environment that grow the plant until it reaches the reproduction life stage.",
            "placeholder guidance",
        )

        self.assertIn('run_action(f"teleport to {seed_room}")', code)
        self.assertIn('run_action("teleport to outside")', code)
        self.assertIn('run_action("teleport to greenhouse")', code)
        self.assertIn('handle_ambiguity(run_action(f"focus on {seed_name}"))', code)
        self.assertIn('handle_ambiguity(run_action(f"pick up {seed_name}"))', code)
        self.assertNotIn('teleport to workshop', code)
        self.assertNotIn('teleport to living room', code)

    def test_scienceworld_system_prompt_marks_tasks_as_simulated(self):
        self.assertIn("fictional ScienceWorld benchmark environment", scienceworld_system_prompt)
        self.assertIn("do not give real-world advice or safety commentary", scienceworld_system_prompt)

    def test_scienceworld_parse_action_normalizes_common_aliases(self):
        if parse_action is None:
            self.skipTest("scienceworld_run.parse_action is unavailable in this test environment.")

        self.assertEqual(parse_action("Action: look"), "look around")
        self.assertEqual(parse_action("Action: close the blast furnace door"), "close blast furnace")
        self.assertEqual(parse_action("Action: turn on the blast furnace"), "activate blast furnace")
        self.assertEqual(parse_action("Action: take the thermometer from the table"), "pick up thermometer")
        self.assertEqual(
            parse_action("Action: put the ceramic cup into the blast furnace"),
            "move ceramic cup to blast furnace",
        )
        self.assertEqual(
            parse_action("Action: activate the blast furnace"),
            "activate blast furnace",
        )

    def test_runtime_recompile_task_context_includes_recent_trace(self):
        module = SkillModule(
            skills_dir=str(ROOT_DIR / "experiments" / "src" / "skills" / "scienceworld"),
            selection_strategy="dsc",
        )
        module.last_selected_skill_names = [
            "scienceworld-object-locator",
            "scienceworld-object-focuser",
        ]

        runtime_task = module.build_runtime_recompile_task(
            "Your task is to determine if unknown substance S is electrically conductive.",
            [
                {"role": "assistant", "content": "Action: open drawer"},
                {"role": "user", "content": "Observation: Nothing happens."},
                {"role": "assistant", "content": "Action: look around"},
                {"role": "user", "content": "Observation: The workshop contains a battery and wires."},
            ],
            {
                "reason": "action_failure",
                "action": "open drawer",
                "observation": "Nothing happens.",
                "selected_skills_before": ["scienceworld-object-locator"],
            },
            remaining_steps=7,
        )

        self.assertIn("[Runtime Compiler Context]", runtime_task)
        self.assertIn("Trigger reason: action_failure", runtime_task)
        self.assertIn("Latest action: open drawer", runtime_task)
        self.assertIn("Nothing happens.", runtime_task)
        self.assertIn("scienceworld-object-locator", runtime_task)

    def test_runtime_recompile_task_context_includes_state_progress(self):
        module = SkillModule(
            skills_dir=str(ROOT_DIR / "experiments" / "src" / "skills" / "alfworld"),
            selection_strategy="dsc",
        )
        module.last_selected_skill_names = [
            "alfworld-object-picker",
            "alfworld-object-disposer",
        ]

        runtime_task = module.build_runtime_recompile_task(
            "Your task is to: put two spraybottle in garbagecan.",
            [],
            {
                "reason": "action_failure",
                "action": "take spraybottle 2 from sidetable 1",
                "observation": "Nothing happens.",
                "selected_skills_before": ["alfworld-object-picker"],
                "state_snapshot": {
                    "current_location": "sidetable 1",
                    "inventory": ["spraybottle 1"],
                    "visible_entities": ["spraybottle 2", "tissuebox 1"],
                    "visited_locations": ["sidetable 1", "garbagecan 1"],
                    "completed_transfers": [
                        {"object": "spraybottle 1", "destination": "garbagecan 1"},
                    ],
                },
            },
            remaining_steps=6,
        )

        self.assertIn("Progress summary:", runtime_task)
        self.assertIn("Inventory already holds 1/2 target object(s)", runtime_task)
        self.assertIn("Confirmed placements already completed: 1/2", runtime_task)
        self.assertIn("State snapshot:", runtime_task)
        self.assertIn("completed_transfers", runtime_task)

    def test_runtime_recompile_proxy_raises_on_failed_observation(self):
        class FakeEnv:
            def step(self, action):
                return "Nothing happens.", 0, False, {"score": 0}

        controller = RuntimeRecompileController(
            benchmark="scienceworld",
            enabled=True,
            selected_skill_names=["scienceworld-object-locator"],
            min_steps_between_recompiles=1,
            stagnation_threshold=2,
        )
        proxy = RuntimeRecompileEnvProxy(
            FakeEnv(),
            controller,
            lambda action, result: {
                "action": action,
                "observation": result[0],
                "task_done": result[2],
                "task_reward": result[3]["score"],
            },
            benchmark="scienceworld",
        )

        with self.assertRaises(RuntimeSkillRecompileRequested) as raised:
            proxy.step("open drawer")

        decision = raised.exception.decision
        self.assertEqual(decision.reason, "action_failure")
        self.assertEqual(decision.step_index, 1)
        self.assertEqual(decision.action, "open drawer")
        self.assertEqual(decision.state_snapshot.get("benchmark"), "scienceworld")

    def test_runtime_recompile_proxy_handles_singleton_tuples_from_alfworld(self):
        class FakeEnv:
            def __init__(self):
                self.actions = []

            def step(self, action):
                self.actions.append(action)
                return ("Nothing happens.",), (0,), (False,), {"won": [False]}

        controller = RuntimeRecompileController(
            benchmark="alfworld",
            enabled=True,
            selected_skill_names=["alfworld-object-locator"],
            min_steps_between_recompiles=1,
            stagnation_threshold=2,
        )
        proxy = RuntimeRecompileEnvProxy(
            FakeEnv(),
            controller,
            lambda action, result: {
                "action": action,
                "observation": result[0],
                "task_done": result[2],
                "task_reward": result[3]["won"],
            },
            benchmark="alfworld",
        )

        proxy.step(["open cabinet 1"])

        with self.assertRaises(RuntimeSkillRecompileRequested) as raised:
            proxy.step(["open cabinet 1"])

        decision = raised.exception.decision
        self.assertEqual(decision.reason, "action_failure")
        self.assertEqual(decision.action, "open cabinet 1")

    def test_runtime_recompile_proxy_repairs_missing_spacing_in_alfworld(self):
        class FakeEnv:
            def __init__(self):
                self.actions = []

            def step(self, action):
                self.actions.append(action)
                return (
                    ("You arrive at coffeetable 1. On the coffeetable 1, you see a remotecontrol 1.",),
                    (0,),
                    (False,),
                    {"won": [False]},
                )

        controller = RuntimeRecompileController(
            benchmark="alfworld",
            enabled=True,
            selected_skill_names=["alfworld-object-locator"],
            min_steps_between_recompiles=1,
            stagnation_threshold=2,
        )
        env = FakeEnv()
        proxy = RuntimeRecompileEnvProxy(
            env,
            controller,
            lambda action, result: {
                "action": action,
                "observation": result[0],
                "task_done": result[2],
                "task_reward": result[3]["won"],
            },
            benchmark="alfworld",
        )

        proxy.step(["go to coffeetable1"])

        self.assertEqual(env.actions, [["go to coffeetable 1"]])

    def test_runtime_recompile_proxy_repairs_missing_object_index_from_observation(self):
        class FakeEnv:
            def __init__(self):
                self.actions = []

            def step(self, action):
                self.actions.append(action)
                action_text = action[0]
                if action_text == "go to diningtable 1":
                    return (
                        ("You arrive at diningtable 1. On the diningtable 1, you see a mug 1, and a pen 2.",),
                        (0,),
                        (False,),
                        {"won": [False]},
                    )
                if action_text == "take mug 1 from diningtable 1":
                    return (
                        ("You pick up the mug 1 from the diningtable 1.",),
                        (0,),
                        (False,),
                        {"won": [False]},
                    )
                return (("Nothing happens.",), (0,), (False,), {"won": [False]})

        controller = RuntimeRecompileController(
            benchmark="alfworld",
            enabled=True,
            selected_skill_names=["alfworld-object-picker"],
            min_steps_between_recompiles=1,
            stagnation_threshold=2,
        )
        env = FakeEnv()
        proxy = RuntimeRecompileEnvProxy(
            env,
            controller,
            lambda action, result: {
                "action": action,
                "observation": result[0],
                "task_done": result[2],
                "task_reward": result[3]["won"],
            },
            benchmark="alfworld",
        )

        proxy.step(["go to diningtable 1"])
        proxy.step(["take mug from diningtable 1"])

        self.assertEqual(
            env.actions,
            [["go to diningtable 1"], ["take mug 1 from diningtable 1"]],
        )

    def test_runtime_recompile_controller_skips_when_budget_too_low(self):
        controller = RuntimeRecompileController(
            benchmark="alfworld",
            enabled=True,
            selected_skill_names=["alfworld-object-picker"],
            max_total_steps=6,
            min_steps_between_recompiles=1,
            min_remaining_steps_to_recompile=3,
            stagnation_threshold=2,
        )

        decision = None
        for step in range(4):
            decision = controller.record_step(
                action=f"attempt {step + 1}",
                observation="Nothing happens.",
                task_done=False,
                task_reward=0.0,
            )

        self.assertIsNone(decision)

    def test_alfworld_runtime_recompile_waits_for_repeated_weak_failures(self):
        controller = RuntimeRecompileController(
            benchmark="alfworld",
            enabled=True,
            selected_skill_names=["alfworld-object-picker"],
            min_steps_between_recompiles=1,
            stagnation_threshold=2,
        )

        first_decision = controller.record_step(
            action="open drawer 1",
            observation="Nothing happens.",
            task_done=False,
            task_reward=0.0,
        )
        second_decision = controller.record_step(
            action="open drawer 1",
            observation="Nothing happens.",
            task_done=False,
            task_reward=0.0,
        )

        self.assertIsNone(first_decision)
        self.assertIsNotNone(second_decision)
        self.assertEqual(second_decision.reason, "action_failure")

    def test_alfworld_runtime_recompile_event_includes_state_snapshot(self):
        class FakeEnv:
            def __init__(self):
                self.calls = 0

            def step(self, action):
                self.calls += 1
                if self.calls == 1:
                    return (
                        ("You arrive at sidetable 1. On the sidetable 1, you see a spraybottle 1.",),
                        (0,),
                        (False,),
                        {"won": [False]},
                    )
                return ("Nothing happens.",), (0,), (False,), {"won": [False]}

        controller = RuntimeRecompileController(
            benchmark="alfworld",
            enabled=True,
            selected_skill_names=["alfworld-object-picker"],
            min_steps_between_recompiles=1,
            stagnation_threshold=2,
        )
        proxy = RuntimeRecompileEnvProxy(
            FakeEnv(),
            controller,
            lambda action, result: {
                "action": action,
                "observation": result[0],
                "task_done": result[2],
                "task_reward": result[3]["won"],
            },
            benchmark="alfworld",
        )

        proxy.step(["go to sidetable 1"])
        proxy.step(["go to sidetable 1"])
        with self.assertRaises(RuntimeSkillRecompileRequested) as raised:
            proxy.step(["go to sidetable 1"])

        snapshot = raised.exception.decision.state_snapshot
        self.assertEqual(snapshot.get("benchmark"), "alfworld")
        self.assertIn("sidetable 1", snapshot.get("visited_locations", []))

    def test_execute_compiled_procedure_recompiles_and_updates_skills(self):
        class FakeSkillModule:
            def __init__(self):
                self.selection_strategy = "dsc"
                self.runtime_recompile_enabled = True
                self.runtime_recompile_max_rounds = 2
                self.runtime_recompile_min_interval_steps = 1
                self.runtime_recompile_stagnation_threshold = 2
                self.runtime_recompile_min_remaining_steps = 1
                self.runtime_recompile_trace_tail = 4
                self.runtime_recompile_count = 0
                self.runtime_recompile_events = []
                self.runtime_last_recompile_step = -999
                self.last_selected_skill_names = ["initial-skill"]

            def _infer_benchmark(self):
                return "scienceworld"

            def should_use_runtime_recompile(self):
                return True

            def can_runtime_recompile(self):
                return self.runtime_recompile_count < self.runtime_recompile_max_rounds

            def record_runtime_recompile(self, event):
                self.runtime_recompile_count += 1
                self.runtime_last_recompile_step = event["step_index"]
                self.runtime_recompile_events.append(dict(event))

            def build_runtime_recompile_task(self, task, messages, event, remaining_steps):
                return f"{task}\n[Runtime Compiler Context]\nReason: {event['reason']}"

            def retrieve_relevant_skills(self, task):
                if "[Runtime Compiler Context]" in task:
                    self.last_selected_skill_names = ["recovery-skill"]
                    return ["recovery-skill"]
                self.last_selected_skill_names = ["initial-skill"]
                return ["initial-skill"]

            def generate_overall_procedure(self, task, skill_names):
                return f"skills={skill_names}"

            def generate_overall_procedure_code(self, task, overall_procedure):
                if "[Runtime Compiler Context]" not in task:
                    return """
def overall_procedure_code(env, llm, model, parse_action, messages=[], max_steps=30):
    messages.append({"role": "assistant", "content": "Action: open stuck drawer"})
    observation, step_reward, task_done, info = env.step("open stuck drawer")
    messages.append({"role": "user", "content": f"Observation: {observation}"})
    return messages, task_done, info.get("score", 0), 1
"""
                return """
def overall_procedure_code(env, llm, model, parse_action, messages=[], max_steps=30):
    messages.append({"role": "assistant", "content": "Action: look around"})
    observation, step_reward, task_done, info = env.step("look around")
    messages.append({"role": "user", "content": f"Observation: {observation}"})
    messages.append({"role": "assistant", "content": "Action: open drawer"})
    observation, step_reward, task_done, info = env.step("open drawer")
    messages.append({"role": "user", "content": f"Observation: {observation}"})
    return messages, task_done, info.get("score", 0), 2
"""

        class FakeEnv:
            def __init__(self):
                self.actions = []

            def step(self, action):
                self.actions.append(action)
                if action == "open stuck drawer":
                    return "Nothing happens.", 0, False, {"score": 0}
                if action == "look around":
                    return "The drawer is closed but unlocked.", 0, False, {"score": 0}
                if action == "open drawer":
                    return "You open the drawer.", 1, True, {"score": 1}
                return "", 0, False, {"score": 0}

        skill_module = FakeSkillModule()
        env = FakeEnv()
        result = execute_compiled_procedure(
            env=env,
            llm=lambda *_args, **_kwargs: "",
            model="o4-mini",
            task_prompt="Open the drawer safely.",
            messages=[],
            max_steps=4,
            skill_module=skill_module,
            selected_skill_names=["initial-skill"],
            step_adapter=lambda action, raw_result: {
                "action": action,
                "observation": raw_result[0],
                "task_done": raw_result[2],
                "task_reward": raw_result[3]["score"],
            },
            invoke=lambda func, runtime_env, runtime_messages, remaining_steps: func(
                runtime_env,
                None,
                "o4-mini",
                None,
                runtime_messages,
                remaining_steps,
            ),
        )

        self.assertTrue(result["task_done"])
        self.assertEqual(result["task_reward"], 1)
        self.assertEqual(result["steps"], 3)
        self.assertEqual(result["skill_names"], ["recovery-skill"])
        self.assertEqual(skill_module.runtime_recompile_count, 1)
        self.assertEqual(env.actions, ["open stuck drawer", "look around", "open drawer"])

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
        self.assertIn("scienceworld-threshold-evaluator", selected)
        self.assertIn("scienceworld-conditional-focus-executor", selected)
        self.assertIn("scienceworld-object-locator", selected)

    def test_growth_quality_first_selection_includes_ambiguity_resolution(self):
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

        self.assertIn("scienceworld-ambiguous-action-resolution", selected)
        self.assertIn("scienceworld-planting-coordinator", selected)

    def test_growth_quality_first_selection_can_inject_missing_canonical_skills(self):
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
                "scienceworld-pot-preparer",
                "scienceworld-planting-coordinator",
                "scienceworld-liquid-filler",
                "controlled-waiting",
            ],
            [
                "scienceworld-room-navigator",
                "scienceworld-object-focuser",
                "scienceworld-pot-preparer",
                "scienceworld-planting-coordinator",
                "scienceworld-liquid-filler",
                "controlled-waiting",
            ],
        )

        self.assertIn("soil-extraction", selected)
        self.assertIn("scienceworld-growth-focuser", selected)
        self.assertIn("scienceworld-ambiguous-action-resolution", selected)

    def test_generic_quality_first_selection_preserves_reference_set(self):
        module = SkillModule(
            skills_dir=str(ROOT_DIR / "experiments" / "src" / "skills" / "scienceworld"),
            selection_strategy="dsc",
        )

        selected = module._select_quality_first_skill_names(
            "Your task is to find an animal. First, focus on the animal. Then, move it to the green box in the living room.",
            [
                "scienceworld-object-locator",
                "scienceworld-object-focuser",
                "scienceworld-container-relocator",
            ],
            [
                "scienceworld-object-locator",
                "scienceworld-object-focuser",
                "scienceworld-container-relocator",
                "scienceworld-living-entity-identifier",
                "scienceworld-target-identifier",
                "scienceworld-object-retriever",
            ],
        )

        self.assertEqual(
            selected,
            [
                "scienceworld-object-locator",
                "scienceworld-object-focuser",
                "scienceworld-container-relocator",
            ],
        )

    def test_compile_critic_pass_uses_stronger_model_and_refines_selection(self):
        module = SkillModule(
            skills_dir=str(ROOT_DIR / "experiments" / "src" / "skills" / "scienceworld"),
            selection_strategy="dsc",
            compiler_critic_enabled=True,
            compiler_critic_force=True,
            compiler_critic_model="gpt-5",
        )

        fake_compilation = self._fake_compilation(
            module,
            [
                "scienceworld-object-locator",
                "scienceworld-task-focuser",
                "scienceworld-temperature-measurer",
                "scienceworld-conditional-box-placer",
            ],
        )
        called_models = []

        def fake_llm(messages, is_string, model):
            called_models.append(model)
            if model == "gpt-5":
                return (
                    "<Compile_Critic_Decision>"
                    "{"
                    "\"task_family\": \"temperature\", "
                    "\"must_keep_skills\": [\"scienceworld-object-locator\", \"scienceworld-temperature-measurer\"], "
                    "\"preferred_skill_order\": ["
                    "\"scienceworld-object-locator\", "
                    "\"scienceworld-temperature-measurer\", "
                    "\"scienceworld-conditional-box-placer\""
                    "], "
                    "\"drop_skills\": [\"scienceworld-task-focuser\"], "
                    "\"reasoning_summary\": \"Keep the locator and measurement stack; the task focuser is redundant here.\""
                    "}"
                    "</Compile_Critic_Decision>"
                )
            return (
                "<Relevant_Skill_Names>"
                "[\"scienceworld-object-locator\", \"scienceworld-task-focuser\", "
                "\"scienceworld-temperature-measurer\", \"scienceworld-conditional-box-placer\"]"
                "</Relevant_Skill_Names>"
            )

        with patch.object(module, "_compile_task", return_value=fake_compilation):
            with patch("src.skill.get_llm_response", side_effect=fake_llm):
                selected = module.retrieve_relevant_skills(
                    "Measure the temperature of mercury, then place it in the correct box."
                )

        self.assertEqual(called_models[-1], "gpt-5")
        self.assertEqual(selected[0], "scienceworld-object-locator")
        self.assertIn("scienceworld-temperature-measurer", selected)
        self.assertIn("scienceworld-object-focuser", selected)
        self.assertEqual(
            module.last_compile_critic_decision["task_family"],
            "temperature",
        )

    def test_compile_critic_invalid_output_falls_back_to_quality_first_selection(self):
        module = SkillModule(
            skills_dir=str(ROOT_DIR / "experiments" / "src" / "skills" / "scienceworld"),
            selection_strategy="dsc",
            compiler_critic_enabled=True,
            compiler_critic_force=True,
            compiler_critic_model="gpt-5",
        )

        fake_compilation = self._fake_compilation(
            module,
            [
                "scienceworld-object-locator",
                "scienceworld-object-focuser",
                "scienceworld-conductivity-tester",
                "scienceworld-object-classifier",
            ],
        )

        with patch.object(module, "_compile_task", return_value=fake_compilation):
            with patch(
                "src.skill.get_llm_response",
                side_effect=[
                    (
                        "<Relevant_Skill_Names>"
                        "[\"scienceworld-object-locator\", \"scienceworld-object-focuser\", "
                        "\"scienceworld-conductivity-tester\", \"scienceworld-object-classifier\"]"
                        "</Relevant_Skill_Names>"
                    ),
                    (
                        "<Relevant_Skill_Names>"
                        "[\"scienceworld-object-locator\", \"scienceworld-object-focuser\", "
                        "\"scienceworld-conductivity-tester\", \"scienceworld-object-classifier\"]"
                        "</Relevant_Skill_Names>"
                    ),
                    "<Compile_Critic_Decision>not valid json</Compile_Critic_Decision>",
                ],
            ):
                selected = module.retrieve_relevant_skills(
                    "Determine if unknown substance S is electrically conductive."
                )

        self.assertEqual(
            selected,
            [
                "scienceworld-object-locator",
                "scienceworld-object-focuser",
                "scienceworld-conductivity-tester",
                "scienceworld-object-classifier",
            ],
        )
        self.assertIsNone(module.last_compile_critic_decision)

    def test_retrieve_relevant_skills_uses_reference_style_payload_for_temperature_tasks(self):
        module = SkillModule(
            skills_dir=str(ROOT_DIR / "experiments" / "src" / "skills" / "scienceworld"),
            selection_strategy="dsc",
        )

        fake_compilation = self._fake_compilation(
            module,
            [
                "scienceworld-room-navigator",
                "scienceworld-temperature-measurer",
                "scienceworld-conditional-focus-executor",
            ],
        )
        fake_compilation.metrics.coverage_score = 0.85
        fake_compilation.metrics.subgoal_count = 3
        fake_compilation.metrics.covered_subgoal_count = 3

        with patch.object(module, "_compile_task", return_value=fake_compilation):
            with patch(
                "src.skill.get_llm_response",
                side_effect=[
                    (
                        "<Relevant_Skill_Names>"
                        "[\"scienceworld-object-locator\", \"scienceworld-task-focuser\", "
                        "\"scienceworld-temperature-measurer\", \"scienceworld-conditional-box-placer\"]"
                        "</Relevant_Skill_Names>"
                    ),
                    (
                        "<Relevant_Skill_Names>"
                        "[\"scienceworld-room-navigator\", \"scienceworld-temperature-measurer\", "
                        "\"scienceworld-conditional-focus-executor\"]"
                        "</Relevant_Skill_Names>"
                    ),
                ],
            ):
                selected = module.retrieve_relevant_skills(
                    "Measure the temperature, then focus on the correct box."
                )

        self.assertEqual(
            selected[:3],
            [
                "scienceworld-room-navigator",
                "scienceworld-temperature-measurer",
                "scienceworld-conditional-focus-executor",
            ],
        )
        self.assertIn("scienceworld-object-locator", selected)
        self.assertIn("scienceworld-conditional-box-placer", selected)
        self.assertEqual(module.last_payload_strategy, "quality_first")

    def test_missing_skill_tag_no_longer_raises_index_error(self):
        module = SkillModule(
            skills_dir=str(ROOT_DIR / "experiments" / "src" / "skills" / "scienceworld"),
            selection_strategy="baseline",
        )

        with patch("src.skill.get_llm_response", return_value="<Analysis>no tags</Analysis>"):
            selected = module.retrieve_relevant_skills("Find a living thing.")

        self.assertEqual(selected, [])

    def test_baseline_retrieve_relevant_skills_resets_dsc_state_and_skips_compiler(self):
        module = SkillModule(
            skills_dir=str(ROOT_DIR / "experiments" / "src" / "skills" / "scienceworld"),
            selection_strategy="baseline",
        )
        module.last_compilation = self._fake_compilation(
            module,
            ["scienceworld-object-locator"],
        )
        module.last_payload_strategy = "quality_first"
        module.last_seed_skill_names = ["scienceworld-room-navigator"]
        module.last_quality_reference_skill_names = ["scienceworld-object-locator"]
        module.last_retrieval_warnings = ["stale warning"]

        with patch.object(module, "_compile_task", side_effect=AssertionError("baseline should not compile")):
            with patch(
                "src.skill.get_llm_response",
                return_value=(
                    "<Relevant_Skill_Names>"
                    "[\"scienceworld-object-locator\", \"scienceworld-object-focuser\"]"
                    "</Relevant_Skill_Names>"
                ),
            ):
                selected = module.retrieve_relevant_skills("Find a living thing.")

        self.assertEqual(
            selected,
            ["scienceworld-object-locator", "scienceworld-object-focuser"],
        )
        self.assertIsNone(module.last_compilation)
        self.assertEqual(module.last_payload_strategy, "full")
        self.assertEqual(module.last_seed_skill_names, [])
        self.assertEqual(module.last_quality_reference_skill_names, [])
        self.assertEqual(module.last_retrieval_warnings, [])

    def test_baseline_generate_overall_procedure_ignores_stale_compilation_state(self):
        module = SkillModule(
            skills_dir=str(ROOT_DIR / "experiments" / "src" / "skills" / "scienceworld"),
            selection_strategy="baseline",
        )
        module.last_compilation = self._fake_compilation(
            module,
            ["scienceworld-object-locator"],
        )
        module.last_payload_strategy = "quality_first"

        captured = {}

        def fake_llm(messages, is_string, model):
            captured["prompt"] = messages[1]["content"]
            return (
                "<Analysis>ok</Analysis>"
                "<Overall_Procedure>Use the baseline payload.</Overall_Procedure>"
            )

        with patch("src.skill.get_llm_response", side_effect=fake_llm):
            procedure = module.generate_overall_procedure(
                "Find a living thing.",
                ["scienceworld-object-locator"],
            )

        self.assertEqual(procedure, "Use the baseline payload.")
        self.assertIn("[File: SKILL.md]", captured["prompt"])
        self.assertNotIn("Dynamic Skill Compiler Summary", captured["prompt"])
        self.assertNotIn("=== Compiled Skill:", captured["prompt"])

    def test_skillnet_alias_maps_legacy_baseline_to_clean_skillnet_path(self):
        baseline_alias = SkillModule(
            skills_dir=str(ROOT_DIR / "experiments" / "src" / "skills" / "scienceworld"),
            selection_strategy="baseline",
        )
        direct_skillnet = SkillModule(
            skills_dir=str(ROOT_DIR / "experiments" / "src" / "skills" / "scienceworld"),
            selection_strategy="skillnet",
        )

        self.assertEqual(baseline_alias.selection_strategy, "skillnet")
        self.assertEqual(direct_skillnet.selection_strategy, "skillnet")


if __name__ == "__main__":
    unittest.main()
