from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional

from skillnet_ai.compiler.decompose import TaskDecomposer
from skillnet_ai.compiler.fragments import FragmentMatcher, SkillFragmentExtractor
from skillnet_ai.compiler.graph import (
    DEFAULT_GRAPH_PASSES,
    SkillGraphBuilder,
    SkillGraphCompiler,
    SkillUtilityScorer,
)
from skillnet_ai.compiler.grounding import EnvironmentGrounder
from skillnet_ai.compiler.models import CompiledSkillPackage, LocalEnvironment, SkillAsset, SkillFragment, Subgoal
from skillnet_ai.compiler.query import QueryOptimizer
from skillnet_ai.compiler.retriever import SkillRetriever


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


@dataclass
class DynamicSkillCompiler:
    retriever: SkillRetriever
    config: CompilerConfig = field(default_factory=CompilerConfig)
    query_optimizer: QueryOptimizer = field(default_factory=QueryOptimizer)
    decomposer: TaskDecomposer = field(default_factory=TaskDecomposer)
    fragment_extractor: SkillFragmentExtractor = field(default_factory=SkillFragmentExtractor)
    fragment_matcher: FragmentMatcher = field(default_factory=FragmentMatcher)
    grounder: EnvironmentGrounder = field(default_factory=EnvironmentGrounder)

    def compile(
        self,
        query: str,
        environment: Optional[LocalEnvironment] = None,
    ) -> CompiledSkillPackage:
        env = environment or LocalEnvironment()
        query_plan = self.query_optimizer.optimize(query)
        subgoals = self.decomposer.decompose(query_plan)
        effective_config = self._effective_config(query_plan, subgoals)
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
            ),
            min_relevance=effective_config.min_relevance,
            preserve_top_k=effective_config.preserve_top_k,
            similar_prune_margin=effective_config.similar_prune_margin,
            keep_parent_if_better_by=effective_config.keep_parent_if_better_by,
            pass_sequence=effective_config.graph_passes,
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
                "Adaptive compiler config preserved extra workflow support "
                f"(min_relevance={effective_config.min_relevance:.2f}, "
                f"preserve_top_k={effective_config.preserve_top_k}, "
                f"similar_prune_margin={effective_config.similar_prune_margin:.2f})."
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
    ) -> CompilerConfig:
        effective = replace(self.config)
        if not effective.adaptive_workflow_bias:
            return effective
        if not self._needs_workflow_support_bias(query_plan, subgoals):
            return effective

        effective.min_relevance = min(effective.min_relevance, 0.15)
        effective.preserve_top_k = max(effective.preserve_top_k, 3)
        effective.similar_prune_margin = max(effective.similar_prune_margin, 0.12)
        return effective

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
