from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

from skillnet_ai.compiler.models import (
    CompilationMetrics,
    CompiledSkill,
    CompiledSkillPackage,
    CompilerPassTrace,
    LocalEnvironment,
    QueryPlan,
    SkillAsset,
    SkillFragment,
    SkillGraph,
    SkillRelation,
    Subgoal,
)


SIMILARITY_THRESHOLD = 0.5
ACTION_DRIVER_KEYWORDS = frozenset(
    {
        "activate",
        "build",
        "clean",
        "close",
        "connect",
        "cool",
        "deactivate",
        "disconnect",
        "fetch",
        "heat",
        "mix",
        "move",
        "open",
        "pick",
        "pick_up",
        "place",
        "plant",
        "pour",
        "prepare",
        "put",
        "retrieve",
        "setup",
        "teleport",
        "transfer",
        "transport",
        "turn",
        "use",
    }
)
SUPPORT_KEYWORDS = frozenset(
    {
        "archiv",
        "classif",
        "confirm",
        "evaluat",
        "examine",
        "focus",
        "identif",
        "inspect",
        "locat",
        "look",
        "monitor",
        "observ",
        "read",
        "scan",
        "track",
        "verify",
        "wait",
    }
)
ACTION_INTENT_KEYWORDS = frozenset(
    {
        "activate",
        "boil",
        "build",
        "clean",
        "combust",
        "connect",
        "cool",
        "heat",
        "mix",
        "move",
        "place",
        "put",
        "transform",
        "transport",
        "use",
    }
)
QUERY_ANCHOR_STOPWORDS = frozenset(
    {
        "action",
        "actions",
        "after",
        "agent",
        "before",
        "cause",
        "change",
        "changes",
        "contain",
        "content",
        "contents",
        "current",
        "environment",
        "first",
        "focus",
        "focu",
        "goal",
        "its",
        "item",
        "items",
        "locate",
        "look",
        "make",
        "matter",
        "move",
        "next",
        "object",
        "objects",
        "phase",
        "process",
        "result",
        "state",
        "step",
        "steps",
        "take",
        "that",
        "task",
        "tasks",
        "the",
        "then",
        "thing",
        "things",
        "this",
        "those",
        "these",
        "through",
        "use",
        "using",
        "verify",
        "when",
        "will",
        "with",
        "transform",
        "your",
    }
)
LEGACY_DEFAULT_GRAPH_PASSES: Tuple[str, ...] = (
    "select_covering_skills",
    "fallback_selection",
    "add_dependencies",
    "repair_coverage",
    "add_dependencies",
    "augment_compositional_support",
    "add_dependencies",
    "prune_similar",
    "prune_broad_containers",
    "prune_topic_drift",
    "prune_overlapping_support",
    "trim_low_contribution",
)
SLIM_GRAPH_PASSES: Tuple[str, ...] = (
    "select_covering_skills",
    "fallback_selection",
    "repair_coverage",
    "augment_compositional_support",
    "add_dependencies",
    "prune_topic_drift",
    "prune_overlapping_support",
    "trim_low_contribution",
)
DEFAULT_GRAPH_PASSES: Tuple[str, ...] = LEGACY_DEFAULT_GRAPH_PASSES
GRAPH_PASS_PRESETS: Dict[str, Tuple[str, ...]] = {
    "default": DEFAULT_GRAPH_PASSES,
    "legacy_default": LEGACY_DEFAULT_GRAPH_PASSES,
    "full": LEGACY_DEFAULT_GRAPH_PASSES,
    "slim_default": SLIM_GRAPH_PASSES,
    "lean": SLIM_GRAPH_PASSES,
    "coverage_first": (
        "select_covering_skills",
        "fallback_selection",
        "repair_coverage",
        "augment_compositional_support",
        "add_dependencies",
        "prune_topic_drift",
        "prune_overlapping_support",
    ),
    "conservative": (
        "select_covering_skills",
        "fallback_selection",
        "repair_coverage",
        "augment_compositional_support",
        "add_dependencies",
        "prune_broad_containers",
        "prune_topic_drift",
        "prune_overlapping_support",
        "trim_low_contribution",
    ),
    "minimal": (
        "select_covering_skills",
        "fallback_selection",
        "repair_coverage",
    ),
}
SUPPORTED_GRAPH_PASSES = frozenset(LEGACY_DEFAULT_GRAPH_PASSES)


@dataclass
class SkillGraphBuilder:
    similarity_threshold: float = SIMILARITY_THRESHOLD

    def build(self, skills: Iterable[SkillAsset]) -> SkillGraph:
        skill_map = {skill.skill_id: skill for skill in skills}
        relations: List[SkillRelation] = []
        explicit_relation_count = 0

        for skill in skill_map.values():
            for dependency in skill.dependencies:
                if dependency in skill_map:
                    explicit_relation_count += 1
                    relations.append(
                        SkillRelation(
                            source=skill.skill_id,
                            target=dependency,
                            relation_type="depend_on",
                            reason="declared dependency",
                        )
                    )
            for child in skill.contains:
                if child in skill_map:
                    explicit_relation_count += 1
                    relations.append(
                        SkillRelation(
                            source=child,
                            target=skill.skill_id,
                            relation_type="belong_to",
                            reason="declared containment",
                        )
                    )
            for peer in skill.composes_with:
                if peer in skill_map:
                    explicit_relation_count += 1
                    relations.append(
                        SkillRelation(
                            source=skill.skill_id,
                            target=peer,
                            relation_type="compose_with",
                            reason="declared composition",
                        )
                    )
            for peer in skill.similar_to:
                if peer in skill_map:
                    explicit_relation_count += 1
                    relations.append(
                        SkillRelation(
                            source=skill.skill_id,
                            target=peer,
                            relation_type="similar_to",
                            reason="declared similarity",
                        )
                    )

        # Keep declared relationships authoritative, but still infer similarity so the compiler
        # can collapse near-duplicates even when the hand-authored graph is only partial.
        relations.extend(self._infer_similarity(skill_map))
        if explicit_relation_count == 0:
            relations.extend(self._infer_composition(skill_map))
        return SkillGraph(skills=skill_map, relations=self._dedupe(relations))

    def _infer_similarity(self, skills: Dict[str, SkillAsset]) -> List[SkillRelation]:
        skill_list = list(skills.values())
        relations: List[SkillRelation] = []
        for index, left in enumerate(skill_list):
            left_caps = left.normalized_capabilities()
            for right in skill_list[index + 1:]:
                right_caps = right.normalized_capabilities()
                if not left_caps or not right_caps:
                    continue
                overlap = len(left_caps & right_caps)
                union = len(left_caps | right_caps)
                similarity = overlap / union if union else 0.0
                family_bonus = self._family_bonus(left, right)
                adjusted_similarity = similarity + family_bonus
                if similarity >= self.similarity_threshold or (
                    overlap >= 2 and adjusted_similarity >= self.similarity_threshold
                ):
                    reason = f"inferred similarity={similarity:.2f}"
                    relations.append(
                        SkillRelation(
                            source=left.skill_id,
                            target=right.skill_id,
                            relation_type="similar_to",
                            weight=max(similarity, adjusted_similarity),
                            reason=reason,
                        )
                    )
                    relations.append(
                        SkillRelation(
                            source=right.skill_id,
                            target=left.skill_id,
                            relation_type="similar_to",
                            weight=max(similarity, adjusted_similarity),
                            reason=reason,
                        )
                    )
        return relations

    def _infer_composition(self, skills: Dict[str, SkillAsset]) -> List[SkillRelation]:
        skill_list = list(skills.values())
        relations: List[SkillRelation] = []
        for index, left in enumerate(skill_list):
            left_caps = left.normalized_capabilities()
            left_name_tokens = set(left.name.split("-"))
            for right in skill_list[index + 1:]:
                right_caps = right.normalized_capabilities()
                right_name_tokens = set(right.name.split("-"))
                shared_caps = left_caps & right_caps
                shared_name_tokens = (left_name_tokens & right_name_tokens) - {"scienceworld", "alfworld", "webshop"}
                if len(shared_caps) >= 3 or (len(shared_caps) >= 2 and shared_name_tokens):
                    reason = (
                        f"inferred composition via shared capabilities={len(shared_caps)} "
                        f"and shared_name_tokens={sorted(shared_name_tokens)}"
                    )
                    relations.append(
                        SkillRelation(
                            source=left.skill_id,
                            target=right.skill_id,
                            relation_type="compose_with",
                            weight=min(1.0, 0.3 + 0.1 * len(shared_caps)),
                            reason=reason,
                        )
                    )
                    relations.append(
                        SkillRelation(
                            source=right.skill_id,
                            target=left.skill_id,
                            relation_type="compose_with",
                            weight=min(1.0, 0.3 + 0.1 * len(shared_caps)),
                            reason=reason,
                        )
                    )
        return relations

    def _family_bonus(self, left: SkillAsset, right: SkillAsset) -> float:
        left_tokens = [token for token in left.name.split("-") if token]
        right_tokens = [token for token in right.name.split("-") if token]
        if not left_tokens or not right_tokens:
            return 0.0
        if left_tokens[0] == right_tokens[0]:
            shared_tail = set(left_tokens[1:]) & set(right_tokens[1:])
            if shared_tail:
                return 0.15
        return 0.0

    def _dedupe(self, relations: List[SkillRelation]) -> List[SkillRelation]:
        seen: Set[Tuple[str, str, str]] = set()
        deduped: List[SkillRelation] = []
        for relation in relations:
            key = (relation.source, relation.target, relation.relation_type)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(relation)
        return deduped


@dataclass
class SkillUtilityScorer:
    coverage_weight: float = 0.55
    quality_weight: float = 0.20
    cost_weight: float = 0.15
    latency_weight: float = 0.10

    def score(self, skill: SkillAsset, query_plan: QueryPlan) -> float:
        skill_caps = skill.normalized_capabilities()
        required = query_plan.required_capabilities
        optional = query_plan.optional_capabilities
        coverage = self._coverage(skill_caps, required, optional)
        quality = skill.mean_quality()
        cost = 1.0 / (1.0 + skill.token_cost + skill.execution_cost)
        latency = 1.0 / (1.0 + skill.latency_ms / 1000.0)
        return (
            self.coverage_weight * coverage
            + self.quality_weight * quality
            + self.cost_weight * cost
            + self.latency_weight * latency
        )

    def _coverage(self, capabilities: Set[str], required: Set[str], optional: Set[str]) -> float:
        if not required and not optional:
            return 0.5
        required_hit = len(required & capabilities) / max(len(required), 1)
        optional_hit = len(optional & capabilities) / max(len(optional), 1) if optional else 0.0
        specificity = len(required & capabilities) / max(len(capabilities), 1) if capabilities else 0.0
        return min(1.0, required_hit * 0.65 + optional_hit * 0.15 + specificity * 0.20)


@dataclass
class SkillGraphCompiler:
    scorer: SkillUtilityScorer
    min_relevance: float = 0.25
    preserve_top_k: int = 0
    similar_prune_margin: float = 0.08
    keep_parent_if_better_by: float = 0.05
    pass_sequence: Tuple[str, ...] = DEFAULT_GRAPH_PASSES

    def compile(
        self,
        graph: SkillGraph,
        query_plan: QueryPlan,
        environment: LocalEnvironment,
        subgoals: Optional[List[Subgoal]] = None,
        matched_fragments: Optional[Dict[str, List[SkillFragment]]] = None,
    ) -> CompiledSkillPackage:
        subgoals = subgoals or []
        matched_fragments = matched_fragments or {}
        scores = {
            skill_id: self.scorer.score(skill, query_plan)
            for skill_id, skill in graph.skills.items()
        }
        ranked = sorted(scores, key=lambda skill_id: scores[skill_id], reverse=True)
        pass_sequence = self._validated_pass_sequence()
        dropped: Dict[str, str] = {}
        protected: Set[str] = set(ranked[: self.preserve_top_k]) if self.preserve_top_k > 0 else set()
        subgoal_matches = self._subgoal_matches(graph, subgoals, matched_fragments)
        pass_traces: List[CompilerPassTrace] = []
        selected: Set[str] = set()
        selection_passes = {"select_covering_skills", "fallback_selection"}
        for pass_name in pass_sequence:
            before_selected = set(selected)
            before_dropped = dict(dropped)
            if pass_name == "select_covering_skills":
                selected = self._select_covering_skills(
                    graph=graph,
                    query_plan=query_plan,
                    subgoals=subgoals,
                    scores=scores,
                    protected=protected,
                    subgoal_matches=subgoal_matches,
                )
            elif pass_name == "fallback_selection":
                if not selected:
                    selected = {
                        skill_id for skill_id, score in scores.items() if score >= self.min_relevance
                    }
            elif pass_name == "add_dependencies":
                selected = self._add_dependencies(graph, selected)
            elif pass_name == "repair_coverage":
                selected = self._repair_coverage(
                    graph,
                    selected,
                    query_plan,
                    subgoals,
                    scores,
                    subgoal_matches,
                )
            elif pass_name == "augment_compositional_support":
                selected = self._augment_compositional_support(
                    graph,
                    selected,
                    query_plan,
                    subgoals,
                    scores,
                    subgoal_matches,
                )
            elif pass_name == "prune_similar":
                selected = self._prune_similar(
                    graph,
                    selected,
                    scores,
                    dropped,
                    query_plan,
                    subgoals,
                    subgoal_matches,
                )
            elif pass_name == "prune_broad_containers":
                selected = self._prune_broad_containers(
                    graph,
                    selected,
                    scores,
                    dropped,
                    query_plan,
                    subgoals,
                    subgoal_matches,
                )
            elif pass_name == "prune_topic_drift":
                selected = self._prune_topic_drift(
                    graph,
                    selected,
                    scores,
                    dropped,
                    query_plan,
                    subgoals,
                    subgoal_matches,
                )
            elif pass_name == "prune_overlapping_support":
                selected = self._prune_overlapping_support(
                    graph,
                    selected,
                    scores,
                    dropped,
                    query_plan,
                    subgoals,
                    subgoal_matches,
                )
            elif pass_name == "trim_low_contribution":
                selected = self._trim_low_contribution(
                    graph,
                    selected,
                    scores,
                    dropped,
                    protected,
                    query_plan,
                    subgoals,
                    subgoal_matches,
                )

            if pass_name in selection_passes:
                selected |= protected
            protected &= selected
            pass_traces.append(
                CompilerPassTrace(
                    pass_name=pass_name,
                    before_selected=sorted(before_selected),
                    after_selected=sorted(selected),
                    added=sorted(selected - before_selected),
                    removed=sorted(before_selected - selected),
                    dropped_delta={
                        skill_id: reason
                        for skill_id, reason in dropped.items()
                        if before_dropped.get(skill_id) != reason
                    },
                )
            )

        selected |= protected
        if not selected and ranked:
            selected.add(ranked[0])
            dropped.pop(ranked[0], None)

        for skill_id in graph.skills:
            if skill_id not in selected and skill_id not in dropped:
                dropped[skill_id] = "relevance_below_threshold"

        compiled_skills = [
            CompiledSkill(
                asset=graph.skills[skill_id],
                selected_fragments=matched_fragments.get(skill_id, []),
                assigned_subgoals=self._assigned_subgoals(
                    skill_id,
                    matched_fragments,
                    subgoals,
                    subgoal_matches,
                ),
                localized_instructions=self._localize_instructions(
                    graph.skills[skill_id],
                    environment,
                    matched_fragments.get(skill_id, []),
                ),
                utility_score=scores[skill_id],
                selected_reason=self._build_reason(graph.skills[skill_id], query_plan, scores[skill_id]),
            )
            for skill_id in self._sort_selected(graph, selected, scores)
        ]

        compiled_graph = SkillGraph(
            skills={skill_id: graph.skills[skill_id] for skill_id in selected},
            relations=[
                relation
                for relation in graph.relations
                if relation.source in selected and relation.target in selected
            ],
        )
        execution_order = self._execution_order(compiled_graph, scores)
        metrics = self._metrics(graph, compiled_graph, query_plan)
        fragment_count_after = sum(len(item.selected_fragments) for item in compiled_skills)
        fragment_token_cost_after = sum(
            fragment.token_cost
            for item in compiled_skills
            for fragment in item.selected_fragments
        )
        covered_subgoals = {
            subgoal_id
            for item in compiled_skills
            for subgoal_id in item.assigned_subgoals
        }
        metrics.subgoal_count = len(subgoals)
        metrics.covered_subgoal_count = len(covered_subgoals)
        metrics.fragment_count_after = fragment_count_after
        metrics.fragment_token_cost_after = fragment_token_cost_after
        selected_capabilities: Set[str] = set()
        for item in compiled_skills:
            selected_capabilities |= item.asset.normalized_capabilities()
        metrics.coverage_score = self._coverage_score(
            query_plan=query_plan,
            subgoals=subgoals,
            covered_subgoals=covered_subgoals,
            selected_capabilities=selected_capabilities,
        )

        notes = [
            "Selection prioritized marginal subgoal coverage before utility-only pruning.",
            "Similar skills were merged only when they did not carry unique task coverage.",
            "Dependencies were reintroduced after pruning to preserve executability.",
            "Instructions were localized against the current environment.",
            f"Graph pass sequence: {' -> '.join(pass_sequence)}.",
        ]
        if pass_sequence != DEFAULT_GRAPH_PASSES:
            notes.append("Graph compiler pass sequence was overridden by config.")
        return CompiledSkillPackage(
            query_plan=query_plan,
            subgoals=subgoals,
            graph=compiled_graph,
            compiled_skills=compiled_skills,
            execution_order=execution_order,
            metrics=metrics,
            dropped_skills=dropped,
            notes=notes,
            pass_traces=pass_traces,
        )

    def _select_covering_skills(
        self,
        graph: SkillGraph,
        query_plan: QueryPlan,
        subgoals: List[Subgoal],
        scores: Dict[str, float],
        protected: Set[str],
        subgoal_matches: Dict[str, Dict[str, float]],
    ) -> Set[str]:
        selected: Set[str] = set()
        covered_subgoals: Set[str] = set()
        covered_required: Set[str] = set()
        selected_capabilities: Set[str] = set()
        all_subgoal_ids = {subgoal.subgoal_id for subgoal in subgoals}
        required_pool = self._required_capability_pool(query_plan, subgoals)
        candidate_pool = {
            skill_id for skill_id, score in scores.items() if score >= self.min_relevance
        } | protected
        if not candidate_pool:
            candidate_pool = set(graph.skills)

        for subgoal in sorted(subgoals, key=lambda item: item.priority):
            if subgoal.subgoal_id in covered_subgoals:
                continue
            best_skill = None
            best_value = 0.0
            for skill_id in candidate_pool:
                subgoal_score = subgoal_matches.get(skill_id, {}).get(subgoal.subgoal_id, 0.0)
                if subgoal_score <= 0:
                    continue
                skill_caps = graph.skills[skill_id].normalized_capabilities()
                required_gain = len((skill_caps & subgoal.required_capabilities) - covered_required) / max(
                    len(subgoal.required_capabilities),
                    1,
                )
                structural_bonus = self._structural_bonus(graph, skill_id, selected)
                value = (
                    0.55 * subgoal_score
                    + 0.20 * required_gain
                    + 0.20 * scores[skill_id]
                    + 0.05 * structural_bonus
                )
                if value > best_value:
                    best_value = value
                    best_skill = skill_id
            if best_skill is None:
                continue
            selected.add(best_skill)
            covered_subgoals |= self._covered_subgoals(best_skill, subgoal_matches)
            skill_caps = graph.skills[best_skill].normalized_capabilities()
            covered_required |= skill_caps & required_pool
            selected_capabilities |= skill_caps

        threshold = max(0.18, self.min_relevance * 0.65)
        while True:
            uncovered_subgoals = all_subgoal_ids - covered_subgoals
            uncovered_required = required_pool - covered_required
            best_skill = None
            best_gain = threshold
            for skill_id in graph.skills:
                if skill_id in selected:
                    continue
                gain = self._marginal_gain(
                    graph=graph,
                    skill_id=skill_id,
                    query_plan=query_plan,
                    subgoals=subgoals,
                    selected=selected,
                    selected_capabilities=selected_capabilities,
                    uncovered_required=uncovered_required,
                    uncovered_subgoals=uncovered_subgoals,
                    required_pool=required_pool,
                    scores=scores,
                    subgoal_matches=subgoal_matches,
                )
                if gain > best_gain:
                    best_gain = gain
                    best_skill = skill_id
            if best_skill is None:
                break
            selected.add(best_skill)
            covered_subgoals |= self._covered_subgoals(best_skill, subgoal_matches)
            skill_caps = graph.skills[best_skill].normalized_capabilities()
            covered_required |= skill_caps & required_pool
            selected_capabilities |= skill_caps

            if not uncovered_subgoals and not uncovered_required:
                threshold = max(0.22, self.min_relevance * 0.8)

        return selected

    def _repair_coverage(
        self,
        graph: SkillGraph,
        selected: Set[str],
        query_plan: QueryPlan,
        subgoals: List[Subgoal],
        scores: Dict[str, float],
        subgoal_matches: Dict[str, Dict[str, float]],
    ) -> Set[str]:
        repaired = set(selected)
        all_subgoal_ids = {subgoal.subgoal_id for subgoal in subgoals}
        required_pool = self._required_capability_pool(query_plan, subgoals)

        while True:
            covered_subgoals = self._selected_subgoals(repaired, subgoal_matches)
            uncovered_subgoals = all_subgoal_ids - covered_subgoals
            if not uncovered_subgoals:
                break
            best_skill = None
            best_value = 0.0
            for skill_id in graph.skills:
                if skill_id in repaired:
                    continue
                new_subgoals = self._covered_subgoals(skill_id, subgoal_matches) & uncovered_subgoals
                if not new_subgoals:
                    continue
                value = (
                    0.75 * sum(subgoal_matches[skill_id][subgoal_id] for subgoal_id in new_subgoals)
                    / max(len(uncovered_subgoals), 1)
                    + 0.25 * scores[skill_id]
                    + self._action_bias(query_plan, graph.skills[skill_id])
                    + self._topic_alignment_bias(query_plan, subgoals, graph.skills[skill_id])
                )
                if value > best_value:
                    best_value = value
                    best_skill = skill_id
            if best_skill is None:
                break
            repaired.add(best_skill)

        covered_capabilities = self._selected_capabilities(graph, repaired)
        uncovered_required = required_pool - covered_capabilities
        while uncovered_required:
            best_skill = None
            best_value = 0.0
            for skill_id in graph.skills:
                if skill_id in repaired:
                    continue
                skill_caps = graph.skills[skill_id].normalized_capabilities()
                new_required = skill_caps & uncovered_required
                if not new_required:
                    continue
                value = (
                    0.70 * len(new_required) / max(len(uncovered_required), 1)
                    + 0.30 * scores[skill_id]
                    + self._action_bias(query_plan, graph.skills[skill_id])
                    + self._topic_alignment_bias(query_plan, subgoals, graph.skills[skill_id])
                )
                if value > best_value:
                    best_value = value
                    best_skill = skill_id
            if best_skill is None or best_value < max(0.15, self.min_relevance * 0.5):
                break
            repaired.add(best_skill)
            covered_capabilities |= graph.skills[best_skill].normalized_capabilities()
            uncovered_required = required_pool - covered_capabilities
        return repaired

    def _augment_compositional_support(
        self,
        graph: SkillGraph,
        selected: Set[str],
        query_plan: QueryPlan,
        subgoals: List[Subgoal],
        scores: Dict[str, float],
        subgoal_matches: Dict[str, Dict[str, float]],
    ) -> Set[str]:
        retained = set(selected)
        workflow_task = bool({"build", "transform", "evaluate"} & set(query_plan.intents)) or (
            "workflow" in query_plan.required_capabilities
        )
        required_pool = self._required_capability_pool(query_plan, subgoals)
        changed = True
        while changed:
            changed = False
            covered_capabilities = self._selected_capabilities(graph, retained)
            covered_subgoals = self._selected_subgoals(retained, subgoal_matches)
            for relation in graph.relations:
                if relation.relation_type != "compose_with":
                    continue
                candidate = None
                if relation.source in retained and relation.target not in retained:
                    candidate = relation.target
                elif relation.target in retained and relation.source not in retained:
                    candidate = relation.source
                if candidate is None:
                    continue
                if scores[candidate] < max(0.12, self.min_relevance * 0.5):
                    continue
                candidate_caps = graph.skills[candidate].normalized_capabilities()
                new_required = (candidate_caps & required_pool) - (
                    covered_capabilities & required_pool
                )
                new_optional = (candidate_caps & query_plan.optional_capabilities) - (
                    covered_capabilities & query_plan.optional_capabilities
                )
                new_subgoals = self._covered_subgoals(candidate, subgoal_matches) - covered_subgoals
                declared_composition = relation.reason == "declared composition"
                if (
                    new_required
                    or new_subgoals
                    or len(new_optional) >= 1
                    or (declared_composition and workflow_task)
                ):
                    retained.add(candidate)
                    changed = True
                    break
        return retained

    def _prune_similar(
        self,
        graph: SkillGraph,
        selected: Set[str],
        scores: Dict[str, float],
        dropped: Dict[str, str],
        query_plan: QueryPlan,
        subgoals: List[Subgoal],
        subgoal_matches: Dict[str, Dict[str, float]],
    ) -> Set[str]:
        retained = set(selected)
        for relation in graph.relations:
            if relation.relation_type != "similar_to":
                continue
            if relation.source not in retained or relation.target not in retained:
                continue
            source = relation.source
            target = relation.target
            source_score = scores[source]
            target_score = scores[target]
            if abs(source_score - target_score) < self.similar_prune_margin:
                continue
            loser = target if source_score >= target_score else source
            winner = source if loser == target else target
            if self._has_unique_contribution(
                graph,
                loser,
                retained,
                query_plan,
                subgoals,
                subgoal_matches,
            ):
                continue
            if self._is_dependency_anchor(graph, retained, loser):
                continue
            if loser in retained:
                retained.remove(loser)
                dropped[loser] = f"dominated_by_similar_skill:{winner}"
        return retained

    def _prune_broad_containers(
        self,
        graph: SkillGraph,
        selected: Set[str],
        scores: Dict[str, float],
        dropped: Dict[str, str],
        query_plan: QueryPlan,
        subgoals: List[Subgoal],
        subgoal_matches: Dict[str, Dict[str, float]],
    ) -> Set[str]:
        retained = set(selected)
        for relation in graph.relations:
            if relation.relation_type != "belong_to":
                continue
            child = relation.source
            parent = relation.target
            if child not in retained or parent not in retained:
                continue
            if scores[child] < scores[parent] + self.keep_parent_if_better_by:
                continue
            if self._has_unique_contribution(
                graph,
                parent,
                retained,
                query_plan,
                subgoals,
                subgoal_matches,
            ):
                continue
            if self._is_dependency_anchor(graph, retained, parent):
                continue
            if child in retained and parent in retained:
                retained.remove(parent)
                dropped[parent] = f"replaced_by_more_specific_child:{child}"
        return retained

    def _prune_overlapping_support(
        self,
        graph: SkillGraph,
        selected: Set[str],
        scores: Dict[str, float],
        dropped: Dict[str, str],
        query_plan: QueryPlan,
        subgoals: List[Subgoal],
        subgoal_matches: Dict[str, Dict[str, float]],
    ) -> Set[str]:
        if not self._is_action_heavy_query(query_plan):
            return set(selected)

        retained = set(selected)
        required_pool = self._required_capability_pool(query_plan, subgoals)
        changed = True
        while changed:
            changed = False
            for skill_id in sorted(
                retained,
                key=lambda item: (
                    -self._skill_role_profile(graph.skills[item])["support_score"],
                    -int(self._skill_role_profile(graph.skills[item])["is_verifier"]),
                    scores[item],
                ),
            ):
                profile = self._skill_role_profile(graph.skills[skill_id])
                if profile["driver_score"] > 0:
                    continue
                if not (profile["support_score"] > 0 or profile["is_monitor"] or profile["is_focus"]):
                    continue
                if self._is_dependency_anchor(graph, retained, skill_id):
                    continue
                if self._is_compositional_support(graph, retained, skill_id, query_plan):
                    continue
                if profile["is_focus"] and self._query_explicitly_requires_focus(query_plan):
                    continue
                if profile["is_monitor"] and not self._has_other_monitoring_skill(
                    graph,
                    retained,
                    skill_id,
                ):
                    continue
                if not self._has_action_driver_peer(
                    graph,
                    retained,
                    skill_id,
                    subgoal_matches,
                ):
                    continue

                others = retained - {skill_id}
                other_subgoals = self._selected_subgoals(others, subgoal_matches)
                if (
                    self._covered_subgoals(skill_id, subgoal_matches) - other_subgoals
                    and (
                        profile["is_focus"]
                        or (profile["is_monitor"] and not self._has_other_monitoring_skill(
                            graph,
                            retained,
                            skill_id,
                        ))
                    )
                ):
                    continue
                other_required = self._selected_capabilities(graph, others) & required_pool
                skill_required = graph.skills[skill_id].normalized_capabilities() & required_pool
                if skill_required - other_required:
                    continue

                retained.remove(skill_id)
                dropped[skill_id] = "support_redundant_with_action_drivers"
                changed = True
                break
        return retained

    def _prune_topic_drift(
        self,
        graph: SkillGraph,
        selected: Set[str],
        scores: Dict[str, float],
        dropped: Dict[str, str],
        query_plan: QueryPlan,
        subgoals: List[Subgoal],
        subgoal_matches: Dict[str, Dict[str, float]],
    ) -> Set[str]:
        if not self._is_action_heavy_query(query_plan):
            return set(selected)
        anchors = self._query_anchor_terms(query_plan, subgoals)
        if len(anchors) < 2:
            return set(selected)

        retained = set(selected)
        changed = True
        while changed:
            changed = False
            for skill_id in sorted(
                retained,
                key=lambda item: (
                    self._topic_alignment_score(graph.skills[item], anchors),
                    scores[item],
                ),
            ):
                skill = graph.skills[skill_id]
                profile = self._skill_role_profile(skill)
                anchor_hits = self._skill_anchor_hits(skill, anchors)
                alignment = self._topic_alignment_score(skill, anchors)

                if alignment >= 0.40 or len(anchor_hits) >= 3:
                    continue
                if self._is_dependency_anchor(graph, retained, skill_id) and (
                    alignment >= 0.16 or anchor_hits
                ):
                    continue
                if self._is_compositional_support(graph, retained, skill_id, query_plan) and (
                    alignment >= 0.16 or anchor_hits
                ):
                    continue
                if (
                    profile["is_focus"]
                    and self._query_explicitly_requires_focus(query_plan)
                    and anchor_hits
                ):
                    continue
                if (
                    self._needs_workflow_support(query_plan, subgoals)
                    and anchor_hits
                    and (
                        "workflow" in skill.normalized_capabilities()
                        or "prepar" in skill.normalized_capabilities()
                        or "prepare" in self._skill_text_tokens(skill)
                    )
                ):
                    continue

                others = retained - {skill_id}
                if not others:
                    continue

                other_anchor_hits: Set[str] = set()
                for other in others:
                    other_anchor_hits |= self._skill_anchor_hits(graph.skills[other], anchors)
                if anchor_hits - other_anchor_hits:
                    continue

                unique_contribution = self._has_unique_contribution(
                    graph,
                    skill_id,
                    retained,
                    query_plan,
                    subgoals,
                    subgoal_matches,
                )
                if unique_contribution and not self._has_stronger_aligned_peer(
                    graph,
                    others,
                    skill_id,
                    scores,
                    anchors,
                    subgoal_matches,
                ):
                    continue

                retained.remove(skill_id)
                dropped[skill_id] = "topic_drift_low_anchor_overlap"
                changed = True
                break
        return retained

    def _add_dependencies(self, graph: SkillGraph, selected: Set[str]) -> Set[str]:
        resolved = set(selected)
        changed = True
        while changed:
            changed = False
            for relation in graph.relations:
                if relation.relation_type != "depend_on":
                    continue
                if relation.source in resolved and relation.target not in resolved:
                    resolved.add(relation.target)
                    changed = True
        return resolved

    def _trim_low_contribution(
        self,
        graph: SkillGraph,
        selected: Set[str],
        scores: Dict[str, float],
        dropped: Dict[str, str],
        protected: Set[str],
        query_plan: QueryPlan,
        subgoals: List[Subgoal],
        subgoal_matches: Dict[str, Dict[str, float]],
    ) -> Set[str]:
        retained = set(selected)
        all_subgoal_ids = {subgoal.subgoal_id for subgoal in subgoals}
        required_pool = self._required_capability_pool(query_plan, subgoals)
        changed = True
        while changed:
            changed = False
            covered_subgoals = self._selected_subgoals(retained, subgoal_matches)
            covered_capabilities = self._selected_capabilities(graph, retained)
            for skill_id in sorted(retained, key=lambda item: scores[item]):
                if skill_id in protected or len(retained) <= 1:
                    continue
                if self._is_dependency_anchor(graph, retained, skill_id):
                    continue
                if self._is_compositional_support(graph, retained, skill_id, query_plan):
                    continue
                if self._has_unique_contribution(
                    graph,
                    skill_id,
                    retained,
                    query_plan,
                    subgoals,
                    subgoal_matches,
                ):
                    continue
                others = retained - {skill_id}
                skill_subgoals = self._covered_subgoals(skill_id, subgoal_matches)
                other_subgoals = self._selected_subgoals(others, subgoal_matches)
                other_caps = self._selected_capabilities(graph, others) & required_pool
                if all_subgoal_ids and other_subgoals != covered_subgoals and skill_subgoals:
                    continue
                if required_pool and other_caps != (
                    required_pool & covered_capabilities
                ):
                    continue
                if scores[skill_id] >= max(0.45, self.min_relevance + 0.12):
                    continue
                retained.remove(skill_id)
                dropped[skill_id] = "low_marginal_contribution"
                changed = True
                break
        return retained

    def _subgoal_matches(
        self,
        graph: SkillGraph,
        subgoals: List[Subgoal],
        matched_fragments: Dict[str, List[SkillFragment]],
    ) -> Dict[str, Dict[str, float]]:
        matches: Dict[str, Dict[str, float]] = {skill_id: {} for skill_id in graph.skills}
        if not subgoals:
            return matches

        for skill_id, skill in graph.skills.items():
            skill_caps = skill.normalized_capabilities()
            fragments = matched_fragments.get(skill_id, [])
            for subgoal in subgoals:
                capability_score = self._capability_overlap(
                    capabilities=skill_caps,
                    required=subgoal.required_capabilities,
                    optional=subgoal.optional_capabilities,
                )
                fragment_score = 0.0
                for fragment in fragments:
                    fragment_score = max(
                        fragment_score,
                        self._fragment_subgoal_score(subgoal, fragment),
                    )
                combined = max(fragment_score, capability_score * 0.9)
                if combined >= 0.16:
                    matches[skill_id][subgoal.subgoal_id] = combined
        return matches

    def _marginal_gain(
        self,
        graph: SkillGraph,
        skill_id: str,
        query_plan: QueryPlan,
        subgoals: List[Subgoal],
        selected: Set[str],
        selected_capabilities: Set[str],
        uncovered_required: Set[str],
        uncovered_subgoals: Set[str],
        required_pool: Set[str],
        scores: Dict[str, float],
        subgoal_matches: Dict[str, Dict[str, float]],
    ) -> float:
        skill_caps = graph.skills[skill_id].normalized_capabilities()
        new_required = skill_caps & uncovered_required
        skill_subgoals = self._covered_subgoals(skill_id, subgoal_matches)
        new_subgoals = skill_subgoals & uncovered_subgoals
        overlap_ratio = len(skill_caps & selected_capabilities) / max(len(skill_caps), 1)
        structural_bonus = self._structural_bonus(graph, skill_id, selected)
        subgoal_gain = 0.0
        if uncovered_subgoals:
            subgoal_gain = sum(
                subgoal_matches[skill_id][subgoal_id]
                for subgoal_id in new_subgoals
            ) / max(len(uncovered_subgoals), 1)
        required_gain = len(new_required) / max(len(required_pool), 1) if required_pool else 0.0
        new_subgoal_ratio = len(new_subgoals) / max(len(uncovered_subgoals), 1) if uncovered_subgoals else 0.0
        return (
            0.40 * subgoal_gain
            + 0.25 * new_subgoal_ratio
            + 0.18 * required_gain
            + 0.12 * scores[skill_id]
            + 0.10 * structural_bonus
            + self._action_bias(query_plan, graph.skills[skill_id])
            + self._topic_alignment_bias(query_plan, subgoals, graph.skills[skill_id])
            - 0.15 * overlap_ratio
        )

    def _capability_overlap(
        self,
        capabilities: Set[str],
        required: Set[str],
        optional: Set[str],
    ) -> float:
        if not required and not optional:
            return 0.0
        required_hit = len(required & capabilities) / max(len(required), 1)
        optional_hit = len(optional & capabilities) / max(len(optional), 1) if optional else 0.0
        specificity = len(required & capabilities) / max(len(capabilities), 1) if capabilities else 0.0
        return min(1.0, required_hit * 0.65 + optional_hit * 0.15 + specificity * 0.20)

    def _fragment_subgoal_score(self, subgoal: Subgoal, fragment: SkillFragment) -> float:
        required_hit = len(subgoal.required_capabilities & fragment.capabilities) / max(
            len(subgoal.required_capabilities),
            1,
        )
        optional_hit = len(subgoal.optional_capabilities & fragment.capabilities) / max(
            len(subgoal.optional_capabilities),
            1,
        ) if subgoal.optional_capabilities else 0.0
        action_bonus = 0.15 if fragment.example_actions else 0.0
        return min(1.0, 0.70 * required_hit + 0.15 * optional_hit + action_bonus)

    def _covered_subgoals(
        self,
        skill_id: str,
        subgoal_matches: Dict[str, Dict[str, float]],
    ) -> Set[str]:
        return {
            subgoal_id
            for subgoal_id, score in subgoal_matches.get(skill_id, {}).items()
            if score >= 0.16
        }

    def _selected_subgoals(
        self,
        selected: Set[str],
        subgoal_matches: Dict[str, Dict[str, float]],
    ) -> Set[str]:
        covered: Set[str] = set()
        for skill_id in selected:
            covered |= self._covered_subgoals(skill_id, subgoal_matches)
        return covered

    def _selected_capabilities(self, graph: SkillGraph, selected: Set[str]) -> Set[str]:
        capabilities: Set[str] = set()
        for skill_id in selected:
            capabilities |= graph.skills[skill_id].normalized_capabilities()
        return capabilities

    def _structural_bonus(
        self,
        graph: SkillGraph,
        skill_id: str,
        selected: Set[str],
    ) -> float:
        if not selected:
            return 0.0
        bonus = 0.0
        for relation in graph.relations:
            if relation.source == skill_id and relation.target in selected:
                if relation.relation_type == "depend_on":
                    bonus += 0.5
                elif relation.relation_type == "compose_with":
                    bonus += 0.3
            if relation.target == skill_id and relation.source in selected:
                if relation.relation_type == "depend_on":
                    bonus += 0.6
                elif relation.relation_type == "compose_with":
                    bonus += 0.25
        return min(1.0, bonus)

    def _action_bias(self, query_plan: QueryPlan, skill: SkillAsset) -> float:
        if not self._is_action_heavy_query(query_plan):
            return 0.0
        profile = self._skill_role_profile(skill)
        if profile["driver_score"] > profile["support_score"]:
            return 0.06
        if profile["driver_score"] == 0 and profile["support_score"] > 0:
            return -0.04
        return 0.0

    def _topic_alignment_bias(
        self,
        query_plan: QueryPlan,
        subgoals: List[Subgoal],
        skill: SkillAsset,
    ) -> float:
        anchors = self._query_anchor_terms(query_plan, subgoals)
        if len(anchors) < 2:
            return 0.0
        alignment = self._topic_alignment_score(skill, anchors)
        if alignment >= 0.42:
            return 0.06
        if alignment >= 0.30:
            return 0.03
        if alignment < 0.16:
            return -0.08
        if alignment < 0.24:
            return -0.04
        return 0.0

    def _is_action_heavy_query(self, query_plan: QueryPlan) -> bool:
        intents = set(query_plan.intents)
        if {"build", "transform"} & intents:
            return True
        return bool(query_plan.required_capabilities & ACTION_INTENT_KEYWORDS)

    def _query_explicitly_requires_focus(self, query_plan: QueryPlan) -> bool:
        return "focu" in query_plan.required_capabilities or "focus" in query_plan.required_capabilities

    def _needs_workflow_support(
        self,
        query_plan: QueryPlan,
        subgoals: List[Subgoal],
    ) -> bool:
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

    def _skill_role_profile(self, skill: SkillAsset) -> Dict[str, object]:
        text = " ".join(
            [skill.name, skill.description, *list(skill.instructions or [])]
        ).lower().replace("-", " ").replace("_", " ")
        tokens = self._skill_text_tokens(skill)
        normalized_tokens = {
            token
            for token in skill.normalized_capabilities()
            if token
        }
        combined = tokens | normalized_tokens
        driver_score = sum(
            1
            for keyword in ACTION_DRIVER_KEYWORDS
            if keyword in combined or keyword in text
        )
        support_score = sum(
            1
            for keyword in SUPPORT_KEYWORDS
            if keyword in combined or keyword in text
        )
        return {
            "driver_score": driver_score,
            "support_score": support_score,
            "is_focus": "focus" in combined or "focu" in combined,
            "is_monitor": any(
                keyword in combined or keyword in text
                for keyword in ("monitor", "inspect", "look at", "examine", "observ")
            ),
            "is_verifier": any(
                keyword in combined or keyword in text
                for keyword in ("inspect", "verify", "confirm", "check")
            ),
        }

    def _skill_text_tokens(self, skill: SkillAsset) -> Set[str]:
        text = " ".join(
            [skill.name, skill.description, *list(skill.instructions or [])]
        ).lower().replace("-", " ").replace("_", " ")
        return {token for token in re.split(r"[^a-z0-9]+", text) if token}

    def _query_anchor_terms(
        self,
        query_plan: QueryPlan,
        subgoals: List[Subgoal],
    ) -> Set[str]:
        raw_tokens = {
            token
            for token in re.split(r"[^a-z0-9]+", query_plan.raw_query.lower())
            if len(token) > 2 and token not in QUERY_ANCHOR_STOPWORDS
        }
        required_tokens = {
            token
            for token in self._required_capability_pool(query_plan, subgoals)
            if len(token) > 2 and token not in QUERY_ANCHOR_STOPWORDS
        }
        return raw_tokens | required_tokens

    def _anchor_match(self, token: str, anchor: str) -> bool:
        if token == anchor:
            return True
        prefix = min(len(token), len(anchor), 6)
        if prefix < 4:
            return False
        return token.startswith(anchor[:prefix]) or anchor.startswith(token[:prefix])

    def _skill_anchor_hits(
        self,
        skill: SkillAsset,
        anchors: Set[str],
    ) -> Set[str]:
        skill_tokens = self._skill_text_tokens(skill) | skill.normalized_capabilities()
        hits: Set[str] = set()
        for anchor in anchors:
            if any(self._anchor_match(token, anchor) for token in skill_tokens):
                hits.add(anchor)
        return hits

    def _topic_alignment_score(
        self,
        skill: SkillAsset,
        anchors: Set[str],
    ) -> float:
        if not anchors:
            return 1.0
        skill_tokens = {
            token
            for token in (self._skill_text_tokens(skill) | skill.normalized_capabilities())
            if token not in QUERY_ANCHOR_STOPWORDS
        }
        if not skill_tokens:
            return 0.0
        hits = self._skill_anchor_hits(skill, anchors)
        recall = len(hits) / max(len(anchors), 1)
        precision = len(hits) / max(len(skill_tokens), 1)
        return min(1.0, 0.65 * recall + 0.35 * min(1.0, precision * 6.0))

    def _has_other_monitoring_skill(
        self,
        graph: SkillGraph,
        selected: Set[str],
        skill_id: str,
    ) -> bool:
        return any(
            other != skill_id and self._skill_role_profile(graph.skills[other])["is_monitor"]
            for other in selected
        )

    def _has_action_driver_peer(
        self,
        graph: SkillGraph,
        selected: Set[str],
        skill_id: str,
        subgoal_matches: Dict[str, Dict[str, float]],
    ) -> bool:
        skill_subgoals = self._covered_subgoals(skill_id, subgoal_matches)
        for other in selected:
            if other == skill_id:
                continue
            profile = self._skill_role_profile(graph.skills[other])
            if profile["driver_score"] <= 0:
                continue
            other_subgoals = self._covered_subgoals(other, subgoal_matches)
            if skill_subgoals & other_subgoals or not skill_subgoals:
                return True
        return False

    def _has_stronger_aligned_peer(
        self,
        graph: SkillGraph,
        others: Set[str],
        skill_id: str,
        scores: Dict[str, float],
        anchors: Set[str],
        subgoal_matches: Dict[str, Dict[str, float]],
    ) -> bool:
        skill_alignment = self._topic_alignment_score(graph.skills[skill_id], anchors)
        skill_subgoals = self._covered_subgoals(skill_id, subgoal_matches)
        for other in others:
            if scores[other] + 0.02 < scores[skill_id]:
                continue
            other_alignment = self._topic_alignment_score(graph.skills[other], anchors)
            if other_alignment < skill_alignment + 0.10:
                continue
            other_subgoals = self._covered_subgoals(other, subgoal_matches)
            if not skill_subgoals or skill_subgoals <= other_subgoals:
                return True
        return False

    def _is_dependency_anchor(
        self,
        graph: SkillGraph,
        selected: Set[str],
        skill_id: str,
    ) -> bool:
        for relation in graph.relations:
            if relation.relation_type != "depend_on":
                continue
            if relation.source in selected and relation.target == skill_id:
                return True
        return False

    def _is_compositional_support(
        self,
        graph: SkillGraph,
        selected: Set[str],
        skill_id: str,
        query_plan: QueryPlan,
    ) -> bool:
        workflow_task = bool({"build", "transform", "evaluate"} & set(query_plan.intents)) or (
            "workflow" in query_plan.required_capabilities
        )
        if not workflow_task:
            return False
        for relation in graph.relations:
            if relation.relation_type != "compose_with" or relation.reason != "declared composition":
                continue
            if relation.source == skill_id and relation.target in selected:
                return True
            if relation.target == skill_id and relation.source in selected:
                return True
        return False

    def _has_unique_contribution(
        self,
        graph: SkillGraph,
        skill_id: str,
        selected: Set[str],
        query_plan: QueryPlan,
        subgoals: Optional[List[Subgoal]],
        subgoal_matches: Dict[str, Dict[str, float]],
    ) -> bool:
        others = set(selected) - {skill_id}
        if not others:
            return True

        other_subgoals = self._selected_subgoals(others, subgoal_matches)
        if self._covered_subgoals(skill_id, subgoal_matches) - other_subgoals:
            return True

        required_pool = self._required_capability_pool(query_plan, subgoals or [])
        skill_required = graph.skills[skill_id].normalized_capabilities() & required_pool
        other_required = self._selected_capabilities(graph, others) & required_pool
        return bool(skill_required - other_required)

    def _coverage_score(
        self,
        query_plan: QueryPlan,
        subgoals: List[Subgoal],
        covered_subgoals: Set[str],
        selected_capabilities: Set[str],
    ) -> float:
        required = self._required_capability_pool(query_plan, subgoals)
        capability_coverage = len(required & selected_capabilities) / max(len(required), 1) if required else 1.0
        if not subgoals:
            return capability_coverage
        subgoal_coverage = len(covered_subgoals) / max(len(subgoals), 1)
        return min(1.0, 0.55 * subgoal_coverage + 0.45 * capability_coverage)

    def _required_capability_pool(
        self,
        query_plan: QueryPlan,
        subgoals: Optional[List[Subgoal]] = None,
    ) -> Set[str]:
        required = set(query_plan.required_capabilities)
        for subgoal in subgoals or []:
            required |= set(subgoal.required_capabilities)
        return required

    def _validated_pass_sequence(self) -> Tuple[str, ...]:
        pass_sequence = tuple(self.pass_sequence or DEFAULT_GRAPH_PASSES)
        unknown = [pass_name for pass_name in pass_sequence if pass_name not in SUPPORTED_GRAPH_PASSES]
        if unknown:
            raise ValueError(f"Unsupported graph compiler passes: {unknown}")
        return pass_sequence

    def _sort_selected(
        self,
        graph: SkillGraph,
        selected: Set[str],
        scores: Dict[str, float],
    ) -> List[str]:
        return sorted(selected, key=lambda skill_id: scores[skill_id], reverse=True)

    def _execution_order(self, graph: SkillGraph, scores: Dict[str, float]) -> List[str]:
        pending = set(graph.skills)
        order: List[str] = []
        while pending:
            ready = [
                skill_id
                for skill_id in pending
                if all(
                    relation.target not in pending
                    for relation in graph.relations
                    if relation.source == skill_id and relation.relation_type == "depend_on"
                )
            ]
            if not ready:
                order.extend(sorted(pending, key=lambda skill_id: scores[skill_id], reverse=True))
                break
            ready.sort(key=lambda skill_id: scores[skill_id], reverse=True)
            order.extend(ready)
            pending -= set(ready)
        return order

    def _build_reason(self, skill: SkillAsset, query_plan: QueryPlan, score: float) -> str:
        matched = sorted(skill.normalized_capabilities() & query_plan.required_capabilities)
        if matched:
            return f"matched required capabilities {', '.join(matched[:5])}; utility={score:.3f}"
        return f"selected for structural support; utility={score:.3f}"

    def _localize_instructions(
        self,
        skill: SkillAsset,
        environment: LocalEnvironment,
        fragments: List[SkillFragment],
    ) -> List[str]:
        localized: List[str] = []
        actionable_fragments = [
            fragment.content
            for fragment in fragments
            if fragment.example_actions or len(fragment.content.split()) >= 4
        ]
        source_instructions = skill.instructions or [skill.description]
        merged_instructions = list(dict.fromkeys(source_instructions + actionable_fragments))
        source_instructions = merged_instructions or source_instructions
        for instruction in source_instructions:
            line = instruction.replace("{cwd}", environment.cwd)
            line = line.replace("{workspace_root}", environment.workspace_root)
            line = line.replace("python ", f"{environment.python_bin} ")
            if "~/workspace" in line:
                line = line.replace("~/workspace", environment.workspace_root)
            localized.append(line)
        return localized

    def _assigned_subgoals(
        self,
        skill_id: str,
        matched_fragments: Dict[str, List[SkillFragment]],
        subgoals: List[Subgoal],
        subgoal_matches: Dict[str, Dict[str, float]],
    ) -> List[str]:
        matched = [
            subgoal.subgoal_id
            for subgoal in subgoals
            if subgoal.subgoal_id in subgoal_matches.get(skill_id, {})
        ]
        if matched:
            return matched

        fragments = matched_fragments.get(skill_id, [])
        fragment_caps = set()
        for fragment in fragments:
            fragment_caps |= fragment.capabilities
        assigned = []
        for subgoal in subgoals:
            if subgoal.required_capabilities & fragment_caps:
                assigned.append(subgoal.subgoal_id)
        return assigned

    def _metrics(
        self,
        original_graph: SkillGraph,
        compiled_graph: SkillGraph,
        query_plan: QueryPlan,
    ) -> CompilationMetrics:
        original_token_cost = sum(skill.token_cost for skill in original_graph.skills.values())
        compiled_token_cost = sum(skill.token_cost for skill in compiled_graph.skills.values())
        original_execution_cost = sum(skill.execution_cost for skill in original_graph.skills.values())
        compiled_execution_cost = sum(skill.execution_cost for skill in compiled_graph.skills.values())
        selected_capabilities: Set[str] = set()
        fragment_count_before = 0
        for skill in compiled_graph.skills.values():
            selected_capabilities |= skill.normalized_capabilities()
        for skill in original_graph.skills.values():
            fragment_count_before += max(1, len(skill.instructions) or 1)

        required = query_plan.required_capabilities
        coverage = len(required & selected_capabilities) / max(len(required), 1)
        redundancy = 1.0 - (len(compiled_graph.skills) / max(len(original_graph.skills), 1))

        return CompilationMetrics(
            candidate_count=len(original_graph.skills),
            selected_count=len(compiled_graph.skills),
            edge_count_before=len(original_graph.relations),
            edge_count_after=len(compiled_graph.relations),
            estimated_token_cost_before=original_token_cost,
            estimated_token_cost_after=compiled_token_cost,
            estimated_execution_cost_before=original_execution_cost,
            estimated_execution_cost_after=compiled_execution_cost,
            coverage_score=coverage,
            redundancy_reduction=max(0.0, redundancy),
            fragment_count_before=fragment_count_before,
        )
