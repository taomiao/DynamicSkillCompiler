import json
import os
import tempfile
import unittest

from skillnet_ai.compiler import (
    CompilerConfig,
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

        self.assertIn("cooler", plan.required_capabilities)
        expanded_terms = set(plan.required_capabilities) | set(plan.optional_capabilities)
        self.assertTrue({"place", "placer", "move"} & expanded_terms)

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
        self.assertIn("thermometer", plan.required_capabilities)
        self.assertIn("focus", plan.required_capabilities)


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
                token_cost=2,
                execution_cost=1,
                quality_scores={"executability": 0.75},
            ),
            SkillAsset(
                skill_id="b",
                name="parser-b",
                description="Parse pdf extract text efficiently",
                capabilities={"pdf", "parse", "extract", "efficient"},
                similar_to={"a"},
                token_cost=2.2,
                execution_cost=1,
                quality_scores={"executability": 0.77},
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


if __name__ == "__main__":
    unittest.main()
