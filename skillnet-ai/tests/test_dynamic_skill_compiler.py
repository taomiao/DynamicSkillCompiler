import json
import os
import tempfile
import unittest

from skillnet_ai.compiler import (
    CompilerConfig,
    GRAPH_PASS_PRESETS,
    DynamicSkillCompiler,
    EnvironmentGrounder,
    FragmentMatcher,
    InMemorySkillRetriever,
    LocalEnvironment,
    LocalSkillLibraryRetriever,
    QueryOptimizer,
    SkillAsset,
    SkillFragmentExtractor,
    TaskDecomposer,
)


class QueryOptimizerTest(unittest.TestCase):
    def test_optimizer_extracts_keywords_intents_and_constraints(self):
        optimizer = QueryOptimizer()

        plan = optimizer.optimize(
            "Build a local low token PDF analysis workflow for this workspace"
        )

        self.assertIn("build", plan.intents)
        self.assertIn("localize", plan.intents)
        self.assertIn("low_token", plan.constraints)
        self.assertIn("pdf", plan.required_capabilities)
        self.assertTrue(plan.keyword_query)
        self.assertGreaterEqual(len(plan.semantic_queries), 2)

    def test_optimizer_expands_domain_synonyms(self):
        optimizer = QueryOptimizer()

        plan = optimizer.optimize("cool and put the pan")

        self.assertIn("cool", plan.required_capabilities)
        expanded_terms = set(plan.required_capabilities) | set(plan.optional_capabilities)
        self.assertTrue({"place", "plac", "move"} & expanded_terms)
        self.assertIn("contain", expanded_terms)
        self.assertIn("monit", expanded_terms)

    def test_optimizer_filters_generic_prompt_words_and_keeps_task_terms(self):
        optimizer = QueryOptimizer()

        plan = optimizer.optimize(
            "Your task is to measure the melting point of mercury. "
            "First, focus on the thermometer. Next, focus on the mercury."
        )

        self.assertNotIn("your", plan.required_capabilities)
        self.assertNotIn("task", plan.required_capabilities)
        self.assertIn("measure", plan.required_capabilities)
        self.assertIn("melt", plan.required_capabilities)
        self.assertIn("mercury", plan.required_capabilities)
        self.assertIn("thermomet", plan.required_capabilities)
        self.assertIn("focu", plan.required_capabilities)

    def test_optimizer_aligns_phase_change_query_with_skill_capability_stems(self):
        optimizer = QueryOptimizer()

        plan = optimizer.optimize(
            "Your task is to boil lead. For compounds without a boiling point, combusting the substance is also acceptable. "
            "First, focus on the substance. Then, take actions that will cause it to change its state of matter."
        )

        self.assertNotIn("acceptable", plan.required_capabilities)
        self.assertNotIn("will", plan.required_capabilities)
        self.assertNotIn("actions", plan.required_capabilities)
        self.assertIn("boil", plan.required_capabilities)
        self.assertIn("combust", plan.required_capabilities)
        self.assertIn("focu", plan.required_capabilities)
        self.assertIn("substance", plan.required_capabilities)
        self.assertIn("contain", plan.optional_capabilities)
        self.assertIn("monit", plan.optional_capabilities)
        self.assertIn("transform", plan.intents)


class DynamicSkillCompilerTest(unittest.TestCase):
    def test_task_decomposition_and_fragment_grounding(self):
        query_plan = QueryOptimizer().optimize("cool the pan and put it on stoveburner")
        subgoals = TaskDecomposer().decompose(query_plan)
        self.assertEqual(len(subgoals), 2)
        self.assertIn("cool", subgoals[0].required_capabilities)
        self.assertIn("put", subgoals[1].required_capabilities)

        skill = SkillAsset(
            skill_id="alf-cool",
            name="alfworld-object-cooler",
            description="Cool an object and place it.",
            instructions=[
                "Execute the `cool {object} with {recep}` action.",
                "Execute the `move {object} to {recep}` action.",
            ],
        )
        extractor = SkillFragmentExtractor()
        fragments = extractor.extract([skill])["alf-cool"]
        matches = FragmentMatcher().match(subgoals[0], fragments)
        grounded = EnvironmentGrounder().ground_fragment(
            matches[0],
            subgoals[0],
            LocalEnvironment(cwd="/tmp", workspace_root="/tmp/ws", benchmark="alfworld"),
        )
        self.assertTrue(grounded.example_actions)
        self.assertIn("domain=alfworld", grounded.content)
        self.assertTrue(matches[0].action_schema)

    def test_task_decomposition_splits_multiphase_scienceworld_query(self):
        query_plan = QueryOptimizer().optimize(
            "Your task is to measure the temperature of unknown substance B. "
            "First, focus on the thermometer. Next, focus on the unknown substance B. "
            "If the temperature is above 50 degrees celsius, place it in the green box."
        )
        subgoals = TaskDecomposer().decompose(query_plan)

        self.assertGreaterEqual(len(subgoals), 4)
        descriptions = " ".join(subgoal.description for subgoal in subgoals)
        self.assertIn("thermometer", descriptions)
        self.assertIn("green box", descriptions)

    def test_fragment_extractor_prioritizes_late_action_lines(self):
        skill = SkillAsset(
            skill_id="transfer",
            name="scienceworld-container-transfer",
            description="Transfer a substance into a target container.",
            instructions=[
                "Purpose: move the substance to a safer container.",
                "This skill helps prepare a material for processing.",
                "Use it before heating or mixing steps.",
                "The task may require verification after transfer.",
                "Keep object names grounded in observations.",
                "Containers are already open in this environment.",
                "Command: move <SUBSTANCE> to <CONTAINER>",
                "Verify the transfer with look at <CONTAINER>.",
            ],
        )

        fragments = SkillFragmentExtractor(max_fragments_per_skill=4).extract([skill])["transfer"]
        actions = [fragment.action_schema for fragment in fragments if fragment.action_schema]

        self.assertIn("move <SUBSTANCE> to <CONTAINER>", actions)
        self.assertTrue(any("look at <CONTAINER>" in action for action in actions))

    def test_task_decomposition_adds_phase_change_support_capabilities(self):
        query_plan = QueryOptimizer().optimize(
            "Your task is to boil lead. First, focus on the substance. "
            "Then, take actions that will cause it to change its state of matter."
        )
        subgoals = TaskDecomposer().decompose(query_plan)

        self.assertGreaterEqual(len(subgoals), 3)
        thermal_subgoal = next(
            subgoal for subgoal in subgoals
            if "boil" in subgoal.required_capabilities or "transform" in subgoal.required_capabilities
        )
        self.assertIn("contain", thermal_subgoal.required_capabilities)
        self.assertIn("apparatu", thermal_subgoal.required_capabilities)
        self.assertIn("monit", thermal_subgoal.optional_capabilities)
        self.assertEqual(thermal_subgoal.environment_hints.get("domain"), "scienceworld")

    def test_compiler_prunes_redundant_skills_and_keeps_dependencies(self):
        skills = [
            SkillAsset(
                skill_id="runtime",
                name="python-runtime",
                description="Provide python execution in the local workspace.",
                capabilities={"python", "workspace", "execution"},
                token_cost=0.5,
                execution_cost=0.2,
                quality_scores={"executability": 0.9},
            ),
            SkillAsset(
                skill_id="parser-heavy",
                name="pdf-parser-heavy",
                description="Parse pdf documents and extract structured text for analysis.",
                capabilities={"pdf", "parse", "extract", "analysis"},
                dependencies={"runtime"},
                token_cost=15,
                execution_cost=5,
                latency_ms=900,
                quality_scores={"executability": 0.7, "maintainability": 0.6},
                instructions=["python parser.py --input {cwd}/report.pdf"],
            ),
            SkillAsset(
                skill_id="parser-lite",
                name="pdf-parser-lite",
                description="Efficient pdf parse and extract workflow for analysis.",
                capabilities={"pdf", "parse", "extract", "analysis", "efficient"},
                similar_to={"parser-heavy"},
                dependencies={"runtime"},
                token_cost=2,
                execution_cost=1,
                latency_ms=120,
                quality_scores={"executability": 0.85, "maintainability": 0.8},
                instructions=["python parser_lite.py --input {cwd}/report.pdf"],
            ),
            SkillAsset(
                skill_id="summarizer",
                name="summary-skill",
                description="Summarize extracted document findings into concise analysis.",
                capabilities={"summarize", "analysis", "document"},
                composes_with={"parser-lite"},
                token_cost=3,
                execution_cost=1,
                latency_ms=100,
                quality_scores={"executability": 0.9, "cost_awareness": 0.8},
                instructions=["python summarize.py --from {workspace_root}/outputs.json"],
            ),
            SkillAsset(
                skill_id="suite",
                name="document-suite",
                description="Large umbrella suite for general document automation.",
                capabilities={"document", "automation", "general"},
                contains={"parser-lite", "summarizer"},
                token_cost=20,
                execution_cost=8,
                latency_ms=1500,
                quality_scores={"executability": 0.6},
            ),
            SkillAsset(
                skill_id="web",
                name="web-scraper",
                description="Scrape web pages for unrelated browser tasks.",
                capabilities={"web", "scrape", "browser"},
                token_cost=8,
                execution_cost=4,
                latency_ms=400,
                quality_scores={"executability": 0.5},
            ),
        ]

        compiler = DynamicSkillCompiler(
            retriever=InMemorySkillRetriever(skills),
            config=CompilerConfig(min_relevance=0.28),
        )
        compiled = compiler.compile(
            query="Build an efficient local PDF analysis workflow with low token cost",
            environment=LocalEnvironment(
                cwd="/tmp/current",
                workspace_root="/tmp/workspace",
                python_bin="python3",
                shell="zsh",
                os_name="macos",
            ),
        )

        selected = {item.asset.skill_id for item in compiled.compiled_skills}
        self.assertIn("parser-lite", selected)
        self.assertIn("summarizer", selected)
        self.assertIn("runtime", selected)
        self.assertNotIn("parser-heavy", selected)
        self.assertNotIn("suite", selected)
        self.assertNotIn("web", selected)
        self.assertGreater(compiled.metrics.redundancy_reduction, 0.0)
        self.assertLess(
            compiled.metrics.estimated_token_cost_after,
            compiled.metrics.estimated_token_cost_before,
        )
        self.assertLess(
            compiled.execution_order.index("runtime"),
            compiled.execution_order.index("parser-lite"),
        )
        self.assertTrue(compiled.subgoals)
        self.assertTrue(any(item.selected_fragments for item in compiled.compiled_skills))
        self.assertGreaterEqual(compiled.metrics.fragment_count_after, 1)
        self.assertGreaterEqual(compiled.metrics.subgoal_count, 1)

        parser_skill = next(item for item in compiled.compiled_skills if item.asset.skill_id == "parser-lite")
        self.assertIn("python3 parser_lite.py --input /tmp/current/report.pdf", parser_skill.localized_instructions[0])

    def test_preserve_top_k_keeps_low_relevance_support_skill(self):
        skills = [
            SkillAsset(
                skill_id="core",
                name="core-pdf",
                description="Analyze pdf files.",
                capabilities={"pdf", "analyze"},
                token_cost=3,
                execution_cost=1,
            ),
            SkillAsset(
                skill_id="support",
                name="support-runtime",
                description="Runtime helper for execution.",
                capabilities={"runtime"},
                token_cost=1,
                execution_cost=0.5,
            ),
        ]
        compiler = DynamicSkillCompiler(
            retriever=InMemorySkillRetriever(skills),
            config=CompilerConfig(min_relevance=0.3, preserve_top_k=2),
        )
        compiled = compiler.compile("Analyze a pdf", environment=LocalEnvironment())
        selected = {item.asset.skill_id for item in compiled.compiled_skills}
        self.assertIn("core", selected)
        self.assertIn("support", selected)

    def test_similar_prune_margin_can_disable_aggressive_pruning(self):
        skills = [
            SkillAsset(
                skill_id="a",
                name="parser-a",
                description="Parse pdf extract text",
                capabilities={"pdf", "parse", "extract"},
                similar_to={"b"},
                token_cost=10,
                execution_cost=4,
                latency_ms=800,
                quality_scores={"executability": 0.40},
            ),
            SkillAsset(
                skill_id="b",
                name="parser-b",
                description="Parse pdf extract text",
                capabilities={"pdf", "parse", "extract"},
                similar_to={"a"},
                token_cost=1,
                execution_cost=0.5,
                latency_ms=80,
                quality_scores={"executability": 0.95},
            ),
        ]
        conservative = DynamicSkillCompiler(
            retriever=InMemorySkillRetriever(skills),
            config=CompilerConfig(
                min_relevance=0.1,
                preserve_top_k=2,
                similar_prune_margin=0.5,
            ),
        )
        compiled = conservative.compile("Parse a pdf efficiently", environment=LocalEnvironment())
        selected = {item.asset.skill_id for item in compiled.compiled_skills}
        self.assertEqual(selected, {"a", "b"})

    def test_graph_pass_sequence_can_disable_similarity_pruning(self):
        skills = [
            SkillAsset(
                skill_id="a",
                name="parser-a",
                description="Parse pdf extract text",
                capabilities={"pdf", "parse", "extract"},
                similar_to={"b"},
                token_cost=10,
                execution_cost=4,
                latency_ms=800,
                quality_scores={"executability": 0.40},
            ),
            SkillAsset(
                skill_id="b",
                name="parser-b",
                description="Parse pdf extract text",
                capabilities={"pdf", "parse", "extract"},
                similar_to={"a"},
                token_cost=1,
                execution_cost=0.5,
                latency_ms=80,
                quality_scores={"executability": 0.95},
            ),
        ]

        default_compiler = DynamicSkillCompiler(
            retriever=InMemorySkillRetriever(skills),
            config=CompilerConfig(
                min_relevance=0.1,
                preserve_top_k=1,
                graph_passes=GRAPH_PASS_PRESETS["legacy_default"],
            ),
        )
        default_compiled = default_compiler.compile(
            "Parse a pdf efficiently",
            environment=LocalEnvironment(),
        )

        custom_compiler = DynamicSkillCompiler(
            retriever=InMemorySkillRetriever(skills),
            config=CompilerConfig(
                min_relevance=0.1,
                preserve_top_k=1,
                graph_passes=(
                    "fallback_selection",
                ),
            ),
        )
        custom_compiled = custom_compiler.compile(
            "Parse a pdf efficiently",
            environment=LocalEnvironment(),
        )

        default_selected = {item.asset.skill_id for item in default_compiled.compiled_skills}
        custom_selected = {item.asset.skill_id for item in custom_compiled.compiled_skills}
        self.assertEqual(default_selected, {"b"})
        self.assertEqual(custom_selected, {"a", "b"})
        self.assertEqual(
            [trace.pass_name for trace in custom_compiled.pass_traces],
            ["fallback_selection"],
        )
        self.assertEqual(custom_compiled.pass_traces[0].added, ["a", "b"])
        self.assertTrue(
            any("Graph compiler pass sequence was overridden by config." in note for note in custom_compiled.notes)
        )

    def test_compiler_prefers_covering_subset_over_dense_candidate_pool(self):
        skills = [
            SkillAsset(
                skill_id="cooler",
                name="alfworld-object-cooler",
                description="Cool a pan or object before moving it.",
                capabilities={"cool", "pan", "temperature"},
                token_cost=4,
                execution_cost=1,
                quality_scores={"executability": 0.9},
            ),
            SkillAsset(
                skill_id="placer",
                name="alfworld-object-placer",
                description="Put an object in the stoveburner or another receptacle.",
                capabilities={"put", "move", "stoveburner", "pan"},
                token_cost=4,
                execution_cost=1,
                quality_scores={"executability": 0.9},
            ),
            SkillAsset(
                skill_id="cooler-heavy",
                name="alfworld-temperature-regulator",
                description="Cool objects using a slower generic temperature workflow.",
                capabilities={"cool", "temperature", "object"},
                similar_to={"cooler"},
                token_cost=9,
                execution_cost=3,
                quality_scores={"executability": 0.65},
            ),
            SkillAsset(
                skill_id="noise-search",
                name="alfworld-search-pattern-executor",
                description="Search unrelated drawers and cabinets.",
                capabilities={"search", "drawer", "cabinet"},
                token_cost=5,
                execution_cost=2,
                quality_scores={"executability": 0.6},
            ),
            SkillAsset(
                skill_id="noise-clean",
                name="alfworld-clean-object",
                description="Clean objects in sinks.",
                capabilities={"clean", "sink", "wash"},
                token_cost=5,
                execution_cost=2,
                quality_scores={"executability": 0.6},
            ),
        ]

        compiler = DynamicSkillCompiler(
            retriever=InMemorySkillRetriever(skills),
            config=CompilerConfig(min_relevance=0.22, preserve_top_k=1),
        )
        compiled = compiler.compile(
            "cool the pan and put it in stoveburner",
            environment=LocalEnvironment(benchmark="alfworld"),
        )

        selected = {item.asset.skill_id for item in compiled.compiled_skills}
        self.assertIn("cooler", selected)
        self.assertIn("placer", selected)
        self.assertNotIn("noise-search", selected)
        self.assertNotIn("noise-clean", selected)
        self.assertLess(len(selected), len(skills))
        self.assertGreaterEqual(compiled.metrics.covered_subgoal_count, 2)

    def test_local_skill_library_retriever_loads_relationships(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            alpha_dir = os.path.join(temp_dir, "alpha")
            beta_dir = os.path.join(temp_dir, "beta")
            os.makedirs(alpha_dir)
            os.makedirs(beta_dir)

            with open(os.path.join(alpha_dir, "SKILL.md"), "w", encoding="utf-8") as file:
                file.write(
                    "---\nname: alpha\ndescription: Alpha parses pdf files.\n---\n# Alpha\n"
                )
            with open(os.path.join(beta_dir, "SKILL.md"), "w", encoding="utf-8") as file:
                file.write("# Beta\nBeta summarizes parsed results.\n")
            with open(os.path.join(temp_dir, "relationships.json"), "w", encoding="utf-8") as file:
                json.dump(
                    [
                        {
                            "source": "beta",
                            "target": "alpha",
                            "type": "depend_on",
                            "reason": "beta needs parsed input",
                        }
                    ],
                    file,
                )

            retriever = LocalSkillLibraryRetriever(temp_dir)
            plan = QueryOptimizer().optimize("Summarize parsed pdf results")
            skills = {skill.skill_id: skill for skill in retriever.retrieve(plan)}

            self.assertIn("alpha", skills)
            self.assertIn("beta", skills)
            self.assertIn("alpha", skills["beta"].dependencies)
            self.assertGreater(skills["alpha"].token_cost, 0.0)

    def test_local_skill_library_retriever_extracts_instruction_lines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            alpha_dir = os.path.join(temp_dir, "alpha")
            os.makedirs(alpha_dir)

            with open(os.path.join(alpha_dir, "SKILL.md"), "w", encoding="utf-8") as file:
                file.write(
                    "---\n"
                    "name: alpha\n"
                    "description: Transfer a substance into a safe container.\n"
                    "---\n"
                    "# Alpha\n"
                    "1. Use `move lead to metal pot`.\n"
                    "2. Verify with `look at metal pot`.\n"
                )

            retriever = LocalSkillLibraryRetriever(temp_dir)
            plan = QueryOptimizer().optimize("Transfer lead into a safe container")
            skill = retriever.retrieve(plan)[0]

            self.assertIn("Transfer a substance into a safe container.", skill.instructions[0])
            self.assertTrue(any("move lead to metal pot" in item for item in skill.instructions))

    def test_compiler_keeps_measurement_threshold_focus_chain_when_query_matches(self):
        skills = [
            SkillAsset(
                skill_id="measure",
                name="scienceworld-temperature-measurer",
                description="Measure the temperature or melting point of mercury with a thermometer.",
                capabilities={"measure", "temperature", "melting", "point", "mercury", "thermometer"},
                token_cost=6,
                execution_cost=1,
            ),
            SkillAsset(
                skill_id="threshold",
                name="scienceworld-threshold-evaluator",
                description="Evaluate whether a measured value is above or below a threshold.",
                capabilities={"evaluate", "threshold", "above", "below", "measure", "value"},
                token_cost=4,
                execution_cost=1,
            ),
            SkillAsset(
                skill_id="focus",
                name="scienceworld-conditional-focus-executor",
                description="Focus on the orange box or yellow box based on the measured threshold result.",
                capabilities={"focus", "orange", "yellow", "box", "conditional", "threshold"},
                token_cost=4,
                execution_cost=1,
            ),
            SkillAsset(
                skill_id="noise",
                name="scienceworld-animal-identifier",
                description="Identify an animal in the outside area.",
                capabilities={"animal", "outside", "identify"},
                token_cost=4,
                execution_cost=1,
            ),
        ]

        compiler = DynamicSkillCompiler(
            retriever=InMemorySkillRetriever(skills),
            config=CompilerConfig(min_relevance=0.25),
        )
        compiled = compiler.compile(
            query=(
                "Measure the melting point of mercury, then focus on the orange box "
                "if above threshold, otherwise focus on the yellow box."
            ),
            environment=LocalEnvironment(),
        )

        selected = {item.asset.skill_id for item in compiled.compiled_skills}
        self.assertIn("measure", selected)
        self.assertIn("threshold", selected)
        self.assertIn("focus", selected)
        self.assertNotIn("noise", selected)

    def test_compiler_selects_phase_change_preparation_and_monitor_chain(self):
        skills = [
            SkillAsset(
                skill_id="focus",
                name="scienceworld-object-focuser",
                description="Focus on the target substance before a critical operation.",
                capabilities={"focu", "substance"},
                token_cost=3,
                execution_cost=1,
            ),
            SkillAsset(
                skill_id="prepare",
                name="scienceworld-substance-preparator",
                description="Move a substance into a suitable container before heating.",
                capabilities={"substance", "contain", "prepar", "heat"},
                token_cost=4,
                execution_cost=1,
            ),
            SkillAsset(
                skill_id="heat",
                name="scienceworld-heating-apparatus-setup",
                description="Place the prepared container onto a heater and activate it.",
                capabilities={"heat", "apparatu", "boil"},
                token_cost=4,
                execution_cost=1,
            ),
            SkillAsset(
                skill_id="monitor",
                name="scienceworld-process-monitor",
                description="Monitor the apparatus and examine the substance to detect state changes.",
                capabilities={"monit", "state", "chang", "substance"},
                token_cost=3,
                execution_cost=1,
            ),
            SkillAsset(
                skill_id="wait",
                name="scienceworld-controlled-waiting",
                description="Wait strategically during an active process before checking again.",
                capabilities={"wait", "process", "monit"},
                token_cost=2,
                execution_cost=1,
            ),
            SkillAsset(
                skill_id="noise",
                name="scienceworld-planting-coordinator",
                description="Place a seed into prepared soil and coordinate plant growth.",
                capabilities={"seed", "soil", "growth"},
                token_cost=3,
                execution_cost=1,
            ),
        ]

        compiler = DynamicSkillCompiler(
            retriever=InMemorySkillRetriever(skills),
            config=CompilerConfig(min_relevance=0.25, preserve_top_k=2),
        )
        compiled = compiler.compile(
            query=(
                "Your task is to boil lead. For compounds without a boiling point, combusting the substance is also acceptable. "
                "First, focus on the substance. Then, take actions that will cause it to change its state of matter."
            ),
            environment=LocalEnvironment(benchmark="scienceworld"),
        )

        selected = {item.asset.skill_id for item in compiled.compiled_skills}
        self.assertIn("focus", selected)
        self.assertIn("prepare", selected)
        self.assertIn("heat", selected)
        self.assertIn("monitor", selected)
        self.assertNotIn("noise", selected)
        self.assertGreaterEqual(compiled.metrics.coverage_score, 0.6)
        self.assertGreaterEqual(compiled.metrics.covered_subgoal_count, 3)
        self.assertTrue(
            any("Adaptive compiler config preserved extra workflow support" in note for note in compiled.notes)
        )

    def test_adaptive_workflow_bias_can_be_disabled_via_config(self):
        skills = [
            SkillAsset(
                skill_id="focus",
                name="scienceworld-object-focuser",
                description="Focus on the target substance before a critical operation.",
                capabilities={"focu", "substance"},
                token_cost=3,
                execution_cost=1,
            ),
            SkillAsset(
                skill_id="prepare",
                name="scienceworld-substance-preparator",
                description="Prepare the target substance before the main operation.",
                capabilities={"substance", "prepar", "workflow"},
                token_cost=4,
                execution_cost=1,
            ),
            SkillAsset(
                skill_id="heat",
                name="scienceworld-heating-apparatus-setup",
                description="Place the prepared container onto a heater and activate it.",
                capabilities={"heat", "apparatu", "boil"},
                token_cost=4,
                execution_cost=1,
            ),
            SkillAsset(
                skill_id="noise",
                name="scienceworld-room-explorer",
                description="Explore unrelated rooms to look for general objects.",
                capabilities={"room", "explor", "search"},
                token_cost=3,
                execution_cost=1,
            ),
        ]

        default_compiler = DynamicSkillCompiler(
            retriever=InMemorySkillRetriever(skills),
            config=CompilerConfig(min_relevance=0.25, preserve_top_k=2),
        )
        default_compiled = default_compiler.compile(
            query=(
                "Your task is to boil lead. First, focus on the substance. "
                "Then, take actions that will cause it to change its state of matter."
            ),
            environment=LocalEnvironment(benchmark="scienceworld"),
        )

        no_bias_compiler = DynamicSkillCompiler(
            retriever=InMemorySkillRetriever(skills),
            config=CompilerConfig(
                min_relevance=0.25,
                preserve_top_k=2,
                adaptive_workflow_bias=False,
            ),
        )
        no_bias_compiled = no_bias_compiler.compile(
            query=(
                "Your task is to boil lead. First, focus on the substance. "
                "Then, take actions that will cause it to change its state of matter."
            ),
            environment=LocalEnvironment(benchmark="scienceworld"),
        )

        default_selected = {item.asset.skill_id for item in default_compiled.compiled_skills}
        no_bias_selected = {item.asset.skill_id for item in no_bias_compiled.compiled_skills}
        self.assertIn("prepare", default_selected)
        self.assertNotIn("prepare", no_bias_selected)
        self.assertTrue(
            any("Adaptive compiler config preserved extra workflow support" in note for note in default_compiled.notes)
        )
        self.assertFalse(
            any("Adaptive compiler config preserved extra workflow support" in note for note in no_bias_compiled.notes)
        )

    def test_prune_overlapping_support_removes_redundant_inspector(self):
        skills = [
            SkillAsset(
                skill_id="transfer",
                name="scienceworld-container-transfer",
                description="Move a substance into a heat-safe container for heating.",
                capabilities={"move", "substance", "contain", "heat", "lead"},
                token_cost=3,
                execution_cost=1,
            ),
            SkillAsset(
                skill_id="setup",
                name="scienceworld-heating-apparatus-setup",
                description="Move the prepared container to a heater and activate it.",
                capabilities={"heat", "apparatu", "activate", "contain"},
                token_cost=3,
                execution_cost=1,
            ),
            SkillAsset(
                skill_id="monitor",
                name="scienceworld-process-monitor",
                description="Look at the apparatus and examine the substance to monitor state changes.",
                capabilities={"monitor", "look", "examine", "state", "substance"},
                token_cost=2,
                execution_cost=1,
            ),
            SkillAsset(
                skill_id="inspector",
                name="scienceworld-container-inspector",
                description="Look at a container to inspect its contents and verify state.",
                capabilities={"inspect", "look", "contain", "verify", "state"},
                token_cost=2,
                execution_cost=1,
            ),
            SkillAsset(
                skill_id="focus",
                name="scienceworld-object-focuser",
                description="Focus on the target substance before manipulation.",
                capabilities={"focus", "substance"},
                token_cost=2,
                execution_cost=1,
            ),
        ]

        compiler = DynamicSkillCompiler(
            retriever=InMemorySkillRetriever(skills),
            config=CompilerConfig(
                min_relevance=0.1,
                graph_passes=("fallback_selection", "prune_overlapping_support"),
            ),
        )
        compiled = compiler.compile(
            query=(
                "Your task is to boil lead. First, focus on the substance. "
                "Then, take actions that will cause it to change its state of matter."
            ),
            environment=LocalEnvironment(benchmark="scienceworld"),
        )

        selected = {item.asset.skill_id for item in compiled.compiled_skills}
        self.assertIn("transfer", selected)
        self.assertIn("monitor", selected)
        self.assertIn("focus", selected)
        self.assertNotIn("inspector", selected)
        self.assertEqual(
            compiled.dropped_skills["inspector"],
            "support_redundant_with_action_drivers",
        )

    def test_prune_topic_drift_removes_off_query_skills(self):
        skills = [
            SkillAsset(
                skill_id="transfer",
                name="scienceworld-container-transfer",
                description="Move lead into a heat-safe container before heating.",
                capabilities={"move", "lead", "substance", "contain", "heat"},
                token_cost=3,
                execution_cost=1,
            ),
            SkillAsset(
                skill_id="setup",
                name="scienceworld-heating-apparatus-setup",
                description="Move the prepared container to a heater and activate it.",
                capabilities={"heat", "activate", "apparatu", "contain"},
                token_cost=3,
                execution_cost=1,
            ),
            SkillAsset(
                skill_id="monitor",
                name="scienceworld-process-monitor",
                description="Examine the heated substance to monitor its phase change.",
                capabilities={"monitor", "examine", "substance", "heat", "phase"},
                token_cost=2,
                execution_cost=1,
            ),
            SkillAsset(
                skill_id="focus",
                name="scienceworld-object-focuser",
                description="Focus on the target substance before manipulation.",
                capabilities={"focus", "substance"},
                token_cost=2,
                execution_cost=1,
            ),
            SkillAsset(
                skill_id="cooler",
                name="scienceworld-substance-cooler",
                description="Cool a substance inside a container and verify the new state.",
                capabilities={"cool", "contain", "substance", "verify", "state"},
                token_cost=2,
                execution_cost=1,
            ),
            SkillAsset(
                skill_id="growth",
                name="scienceworld-growth-focuser",
                description="Focus on a seed and monitor biological growth changes over time.",
                capabilities={"focus", "growth", "monitor", "seed", "time"},
                token_cost=2,
                execution_cost=1,
            ),
        ]

        compiler = DynamicSkillCompiler(
            retriever=InMemorySkillRetriever(skills),
            config=CompilerConfig(
                min_relevance=0.1,
                graph_passes=("fallback_selection", "prune_topic_drift"),
            ),
        )
        compiled = compiler.compile(
            query=(
                "Your task is to boil lead. First, focus on the substance. "
                "Then, take actions that will cause it to change its state of matter."
            ),
            environment=LocalEnvironment(benchmark="scienceworld"),
        )

        selected = {item.asset.skill_id for item in compiled.compiled_skills}
        self.assertIn("transfer", selected)
        self.assertIn("setup", selected)
        self.assertIn("monitor", selected)
        self.assertIn("focus", selected)
        self.assertNotIn("cooler", selected)
        self.assertNotIn("growth", selected)
        self.assertEqual(compiled.dropped_skills["cooler"], "topic_drift_low_anchor_overlap")
        self.assertEqual(compiled.dropped_skills["growth"], "topic_drift_low_anchor_overlap")

    def test_default_pass_trace_records_selection_diff(self):
        skills = [
            SkillAsset(
                skill_id="a",
                name="parser-a",
                description="Parse pdf extract text",
                capabilities={"pdf", "parse", "extract"},
                similar_to={"b"},
                token_cost=10,
                execution_cost=4,
                latency_ms=800,
                quality_scores={"executability": 0.40},
            ),
            SkillAsset(
                skill_id="b",
                name="parser-b",
                description="Parse pdf extract text",
                capabilities={"pdf", "parse", "extract"},
                similar_to={"a"},
                token_cost=1,
                execution_cost=0.5,
                latency_ms=80,
                quality_scores={"executability": 0.95},
            ),
        ]

        compiler = DynamicSkillCompiler(
            retriever=InMemorySkillRetriever(skills),
            config=CompilerConfig(min_relevance=0.1, preserve_top_k=1),
        )
        compiled = compiler.compile(
            "Parse a pdf efficiently",
            environment=LocalEnvironment(),
        )

        first_trace = compiled.pass_traces[0]
        self.assertEqual(first_trace.pass_name, "select_covering_skills")
        self.assertEqual(first_trace.added, ["b"])
        self.assertEqual([trace.pass_name for trace in compiled.pass_traces[:3]], [
            "select_covering_skills",
            "fallback_selection",
            "add_dependencies",
        ])
        self.assertEqual(compiled.dropped_skills["a"], "relevance_below_threshold")


if __name__ == "__main__":
    unittest.main()
