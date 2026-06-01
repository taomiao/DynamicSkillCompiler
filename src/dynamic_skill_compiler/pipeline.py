from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional

from dynamic_skill_compiler.decompose import TaskDecomposer
from dynamic_skill_compiler.fragments import FragmentMatcher, SkillFragmentExtractor
from dynamic_skill_compiler.graph import (
    DEFAULT_GRAPH_PASSES,
    GRAPH_PASS_PRESETS,
    SkillGraphBuilder,
    SkillGraphCompiler,
    SkillUtilityScorer,
)
from dynamic_skill_compiler.grounding import EnvironmentGrounder
from dynamic_skill_compiler.models import CompiledSkillPackage, LocalEnvironment, SkillAsset, SkillFragment, Subgoal
from dynamic_skill_compiler.query import QueryOptimizer
from dynamic_skill_compiler.retriever import SkillRetriever


@dataclass
class CompilerConfig:
    min_relevance: float = 0.25
    similarity_threshold: float = 0.5
    preserve_top_k: int = 2
    similar_prune_margin: float = 0.08
    keep_parent_if_better_by: float = 0.05
    coverage_weight: float = 0.55
    quality_weight: float = 0.20
    cost_weight: float = 0.15
    latency_weight: float = 0.10
    adaptive_workflow_bias: bool = True
    graph_passes: tuple[str, ...] = DEFAULT_GRAPH_PASSES
    max_selected_skills: int = 0
    max_support_skills: int = 0
    action_driver_bonus: float = 0.0
    support_skill_penalty: float = 0.0
    coverage_floor: float = 0.0
    compile_stage: str = "initial"
    profile_name: str = "default"


@dataclass
class DynamicSkillCompiler:
    retriever: SkillRetriever
    config: CompilerConfig = field(default_factory=CompilerConfig)
    query_optimizer: QueryOptimizer = field(default_factory=QueryOptimizer)
    decomposer: TaskDecomposer = field(default_factory=TaskDecomposer)
    fragment_extractor: SkillFragmentExtractor = field(default_factory=SkillFragmentExtractor)
    fragment_matcher: FragmentMatcher = field(default_factory=FragmentMatcher)
    grounder: EnvironmentGrounder = field(default_factory=EnvironmentGrounder)
    # Optional SemanticSoftMatcher.  When provided, capability scoring uses
    # soft cosine-similarity matching instead of pure keyword intersection.
    soft_matcher: object = None  # SemanticSoftMatcher | None

    def compile(
        self,
        query: str,
        environment: Optional[LocalEnvironment] = None,
    ) -> CompiledSkillPackage:
        env = environment or LocalEnvironment()
        query_plan = self.query_optimizer.optimize(query)
        subgoals = self.decomposer.decompose(query_plan)
        effective_config = self._effective_config(query_plan, subgoals, env)
        candidates = self.retriever.retrieve(query_plan)
        matched_fragments = self._compile_fragments(candidates, subgoals, env)

        builder = SkillGraphBuilder(similarity_threshold=effective_config.similarity_threshold)
        graph = builder.build(candidates)

        compiler = SkillGraphCompiler(
            scorer=SkillUtilityScorer(
                coverage_weight=effective_config.coverage_weight,
                quality_weight=effective_config.quality_weight,
                cost_weight=effective_config.cost_weight,
                latency_weight=effective_config.latency_weight,
                soft_matcher=self.soft_matcher,
            ),
            min_relevance=effective_config.min_relevance,
            preserve_top_k=effective_config.preserve_top_k,
            similar_prune_margin=effective_config.similar_prune_margin,
            keep_parent_if_better_by=effective_config.keep_parent_if_better_by,
            pass_sequence=effective_config.graph_passes,
            max_selected_skills=effective_config.max_selected_skills,
            max_support_skills=effective_config.max_support_skills,
            action_driver_bonus=effective_config.action_driver_bonus,
            support_skill_penalty=effective_config.support_skill_penalty,
            coverage_floor=effective_config.coverage_floor,
        )
        compiled = compiler.compile(
            graph=graph,
            query_plan=query_plan,
            environment=env,
            subgoals=subgoals,
            matched_fragments=matched_fragments,
        )
        if effective_config != self.config:
            compiled.notes.append(
                "Adaptive compiler config adjusted selection profile "
                f"(min_relevance={effective_config.min_relevance:.2f}, "
                f"preserve_top_k={effective_config.preserve_top_k}, "
                f"similar_prune_margin={effective_config.similar_prune_margin:.2f}, "
                f"max_selected={effective_config.max_selected_skills}, "
                f"max_support={effective_config.max_support_skills}, "
                f"coverage_floor={effective_config.coverage_floor:.2f}, "
                f"profile={effective_config.profile_name}, "
                f"stage={effective_config.compile_stage})."
            )
        return compiled

    @staticmethod
    def summarize(compiled: CompiledSkillPackage) -> dict:
        return {
            "query": compiled.query_plan.raw_query,
            "subgoals": [subgoal.description for subgoal in compiled.subgoals],
            "selected_skills": [item.asset.name for item in compiled.compiled_skills],
            "execution_order": compiled.execution_order,
            "coverage_score": compiled.metrics.coverage_score,
            "token_savings": compiled.metrics.estimated_token_cost_before
            - compiled.metrics.estimated_token_cost_after,
            "graph_pass_sequence": [trace.pass_name for trace in compiled.pass_traces],
            "pass_trace": [
                {
                    "pass_name": trace.pass_name,
                    "added": trace.added,
                    "removed": trace.removed,
                    "dropped_delta": trace.dropped_delta,
                }
                for trace in compiled.pass_traces
            ],
            "dropped_skills": compiled.dropped_skills,
            "notes": compiled.notes,
        }

    def _compile_fragments(
        self,
        candidates: List[SkillAsset],
        subgoals: List[Subgoal],
        environment: LocalEnvironment,
    ) -> dict[str, List[SkillFragment]]:
        extracted = self.fragment_extractor.extract(candidates)
        selected: dict[str, List[SkillFragment]] = {}
        for subgoal in subgoals:
            for skill in candidates:
                fragments = extracted.get(skill.skill_id, [])
                matches = self.fragment_matcher.match(subgoal, fragments)
                if not matches:
                    continue
                grounded = [
                    self.grounder.ground_fragment(fragment, subgoal, environment)
                    for fragment in matches[:2]
                ]
                selected.setdefault(skill.skill_id, [])
                existing_ids = {fragment.fragment_id for fragment in selected[skill.skill_id]}
                for fragment in grounded:
                    if fragment.fragment_id not in existing_ids:
                        selected[skill.skill_id].append(fragment)
                        existing_ids.add(fragment.fragment_id)
        return selected

    def _effective_config(
        self,
        query_plan,
        subgoals: List[Subgoal],
        environment: LocalEnvironment,
    ) -> CompilerConfig:
        effective = replace(self.config)
        if not effective.adaptive_workflow_bias:
            return effective
        signals = self._task_signals(query_plan, subgoals)
        is_runtime_recompile = effective.compile_stage == "runtime_recompile"
        profile_name = self._task_profile(query_plan, subgoals, signals)
        effective.profile_name = profile_name
        adaptive_passes_allowed = tuple(self.config.graph_passes or DEFAULT_GRAPH_PASSES) == DEFAULT_GRAPH_PASSES

        if profile_name == "search":
            if adaptive_passes_allowed:
                effective.graph_passes = GRAPH_PASS_PRESETS["slim_default"]
            effective.min_relevance = max(
                effective.min_relevance,
                0.24 if not is_runtime_recompile else 0.20,
            )
            effective.preserve_top_k = max(
                effective.preserve_top_k,
                2 if not is_runtime_recompile else 3,
            )
            effective.similar_prune_margin = max(effective.similar_prune_margin, 0.09)
            effective.max_selected_skills = self._tighten_budget(
                effective.max_selected_skills,
                5 if not is_runtime_recompile else 6,
            )
            effective.max_support_skills = self._tighten_budget(
                effective.max_support_skills,
                2 if not is_runtime_recompile else 3,
            )
            effective.action_driver_bonus = max(
                effective.action_driver_bonus,
                0.08 if not is_runtime_recompile else 0.05,
            )
            effective.support_skill_penalty = max(
                effective.support_skill_penalty,
                0.05 if not is_runtime_recompile else 0.02,
            )
            effective.coverage_floor = max(
                effective.coverage_floor,
                0.18 if not is_runtime_recompile else 0.22,
            )
            return effective

        if profile_name == "discovery":
            if adaptive_passes_allowed:
                effective.graph_passes = GRAPH_PASS_PRESETS["coverage_first"]
            effective.min_relevance = min(
                effective.min_relevance,
                0.20 if not is_runtime_recompile else 0.16,
            )
            effective.preserve_top_k = max(
                effective.preserve_top_k,
                4 if not is_runtime_recompile else 5,
            )
            effective.max_selected_skills = self._tighten_budget(
                effective.max_selected_skills,
                6 if not is_runtime_recompile else 7,
            )
            effective.max_support_skills = self._tighten_budget(
                effective.max_support_skills,
                3 if not is_runtime_recompile else 4,
            )
            effective.action_driver_bonus = max(
                effective.action_driver_bonus,
                0.04 if not is_runtime_recompile else 0.02,
            )
            if effective.support_skill_penalty > 0:
                effective.support_skill_penalty = min(
                    effective.support_skill_penalty,
                    0.015 if not is_runtime_recompile else 0.01,
                )
            effective.coverage_floor = max(
                effective.coverage_floor,
                0.34 if not is_runtime_recompile else 0.40,
            )
            return effective

        if profile_name == "workflow":
            if adaptive_passes_allowed:
                effective.graph_passes = GRAPH_PASS_PRESETS["coverage_first"]
            effective.min_relevance = min(
                effective.min_relevance,
                0.18 if not is_runtime_recompile else 0.15,
            )
            effective.preserve_top_k = max(
                effective.preserve_top_k,
                4 if not is_runtime_recompile else 5,
            )
            effective.similar_prune_margin = max(effective.similar_prune_margin, 0.10)
            effective.action_driver_bonus = max(
                effective.action_driver_bonus,
                0.03 if not is_runtime_recompile else 0.02,
            )
            if effective.support_skill_penalty > 0:
                effective.support_skill_penalty = min(
                    effective.support_skill_penalty,
                    0.02 if not is_runtime_recompile else 0.01,
                )
            effective.coverage_floor = max(
                effective.coverage_floor,
                0.50 if not is_runtime_recompile else 0.55,
            )
            return effective

        if profile_name == "physical":
            if adaptive_passes_allowed:
                effective.graph_passes = GRAPH_PASS_PRESETS["slim_default"]
            effective.min_relevance = max(
                effective.min_relevance,
                0.22 if not is_runtime_recompile else 0.18,
            )
            effective.preserve_top_k = max(
                effective.preserve_top_k,
                2 if not is_runtime_recompile else 3,
            )
            effective.max_selected_skills = self._tighten_budget(
                effective.max_selected_skills,
                5 if not is_runtime_recompile else 6,
            )
            effective.max_support_skills = self._tighten_budget(
                effective.max_support_skills,
                2 if not is_runtime_recompile else 3,
            )
            effective.action_driver_bonus = max(
                effective.action_driver_bonus,
                0.05 if not is_runtime_recompile else 0.03,
            )
            effective.support_skill_penalty = max(
                effective.support_skill_penalty,
                0.03 if not is_runtime_recompile else 0.015,
            )
            effective.coverage_floor = max(
                effective.coverage_floor,
                0.24 if not is_runtime_recompile else 0.28,
            )
            return effective

        if adaptive_passes_allowed:
            effective.graph_passes = GRAPH_PASS_PRESETS["default"]
        effective.min_relevance = max(
            effective.min_relevance,
            0.22 if not is_runtime_recompile else 0.18,
        )
        effective.preserve_top_k = max(
            effective.preserve_top_k,
            2 if not is_runtime_recompile else 3,
        )
        effective.max_selected_skills = self._tighten_budget(
            effective.max_selected_skills,
            5 if not is_runtime_recompile else 6,
        )
        effective.max_support_skills = self._tighten_budget(
            effective.max_support_skills,
            2 if not is_runtime_recompile else 3,
        )
        effective.action_driver_bonus = max(
            effective.action_driver_bonus,
            0.04 if not is_runtime_recompile else 0.02,
        )
        effective.support_skill_penalty = max(
            effective.support_skill_penalty,
            0.03 if not is_runtime_recompile else 0.015,
        )
        effective.coverage_floor = max(
            effective.coverage_floor,
            0.24 if not is_runtime_recompile else 0.28,
        )
        return effective

    def _task_profile(
        self,
        query_plan,
        subgoals: List[Subgoal],
        signals: dict,
    ) -> str:
        if signals["search_score"] >= max(0.42, signals["workflow_score"] + 0.12):
            return "search"
        if signals["discovery_score"] >= max(0.45, signals["physical_score"] + 0.08):
            return "discovery"
        if signals["workflow_score"] >= 0.55:
            return "workflow"
        if signals["physical_score"] >= 0.35:
            return "physical"
        return "balanced"

    def _task_signals(
        self,
        query_plan,
        subgoals: List[Subgoal],
    ) -> dict:
        raw_query = query_plan.raw_query.lower()
        token_pool = set(query_plan.required_capabilities) | set(query_plan.optional_capabilities)
        search_markers = (
            "instruction:",
            "price lower than",
            "dollars",
            "under ",
            "less than",
            "color:",
            "size:",
            "search",
            "product",
            "buy",
            "purchase",
        )
        phase_markers = ("first", "next", "then", "before", "after", "otherwise", "if ", "while")
        discovery_markers = (
            "find a(",
            "find a ",
            "find an ",
            "find the ",
            "locate ",
            "identify ",
            "which ",
            "what ",
            "shortest",
            "longest",
            "largest",
            "smallest",
        )
        search_marker_hits = sum(1 for marker in search_markers if marker in raw_query)
        discovery_marker_hits = sum(1 for marker in discovery_markers if marker in raw_query)
        phase_hits = sum(1 for marker in phase_markers if marker in raw_query)
        constraint_hits = (
            len(query_plan.constraints)
            + sum(
                1
                for marker in ("price", "dollar", "color:", "size:", "under", "lower than", "less than")
                if marker in raw_query
            )
        )
        search_token_hits = len(
            token_pool
            & {
                "buy",
                "purchas",
                "search",
                "product",
                "page",
                "price",
                "size",
                "color",
                "variant",
                "option",
                "checkout",
            }
        )
        workflow_structural_hits = len(
            token_pool
            & {
                "workflow",
                "apparatu",
                "contain",
                "monit",
                "threshold",
                "circuit",
                "conduct",
                "temperature",
                "transform",
                "phase",
            }
        )
        physical_hits = len(
            token_pool
            & {
                "open",
                "pick",
                "take",
                "move",
                "place",
                "put",
                "heat",
                "cool",
                "clean",
                "go",
                "navigat",
                "focus",
                "retriev",
            }
        )
        discovery_token_hits = len(
            token_pool
            & {
                "find",
                "locat",
                "identif",
                "thing",
                "animal",
                "plant",
                "object",
                "living",
                "compare",
                "lifespan",
                "short",
                "long",
                "select",
            }
        )
        comparative_hits = sum(
            1
            for marker in ("shortest", "longest", "largest", "smallest", "oldest", "youngest")
            if marker in raw_query
        )
        search_score = min(
            1.0,
            0.14 * search_marker_hits
            + 0.08 * constraint_hits
            + 0.05 * search_token_hits,
        )
        discovery_score = min(
            1.0,
            0.10 * discovery_marker_hits
            + 0.06 * discovery_token_hits
            + 0.15 * comparative_hits
            + 0.12 * int(bool({"find", "locate", "identify"} & set(query_plan.intents)))
            + 0.10 * int(self._needs_discovery_support_bias(query_plan, subgoals))
        )
        workflow_score = min(
            1.0,
            0.08 * len(subgoals)
            + 0.10 * phase_hits
            + 0.20 * int(self._needs_workflow_support_bias(query_plan, subgoals))
            + 0.05 * workflow_structural_hits,
        )
        physical_score = min(
            1.0,
            0.10 * physical_hits
            + 0.08 * len(
                set(query_plan.intents) & {"retrieve", "transform", "evaluate"}
            ),
        )
        return {
            "search_score": search_score,
            "discovery_score": discovery_score,
            "workflow_score": workflow_score,
            "physical_score": physical_score,
            "constraint_density": constraint_hits,
        }

    def _tighten_budget(self, current: int, target: int) -> int:
        if current <= 0:
            return target
        return min(current, target)

    def _needs_workflow_support_bias(self, query_plan, subgoals: List[Subgoal]) -> bool:
        intents = set(query_plan.intents)
        capability_signals = set(query_plan.required_capabilities) | set(query_plan.optional_capabilities)
        phase_markers = ("first", "next", "then", "before", "after", "otherwise", "if", "while")
        structural_terms = {
            "contain",
            "apparatu",
            "monit",
            "threshold",
            "condit",
            "branch",
            "boil",
            "melt",
            "freez",
            "heat",
            "cool",
            "transform",
        }
        multi_phase = len(subgoals) >= 3 or any(marker in query_plan.raw_query.lower() for marker in phase_markers)
        structure_heavy = bool({"build", "transform", "evaluate"} & intents) or (
            len(capability_signals & structural_terms) >= 3
        )
        return multi_phase and structure_heavy

    def _needs_discovery_support_bias(self, query_plan, subgoals: List[Subgoal]) -> bool:
        raw_query = query_plan.raw_query.lower()
        intents = set(query_plan.intents)
        capability_signals = set(query_plan.required_capabilities) | set(query_plan.optional_capabilities)
        abstract_targets = {
            "thing",
            "animal",
            "plant",
            "object",
            "living",
            "nonliving",
            "lifespan",
            "select",
            "identif",
        }
        comparative_markers = ("shortest", "longest", "largest", "smallest", "oldest", "youngest")
        return (
            bool({"find", "locate", "identify"} & intents)
            and (
                len(capability_signals & abstract_targets) >= 2
                or any(marker in raw_query for marker in comparative_markers)
                or len(subgoals) >= 2
            )
        )
