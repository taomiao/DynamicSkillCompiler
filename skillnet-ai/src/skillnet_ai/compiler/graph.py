from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from skillnet_ai.compiler.models import (
    CompilationMetrics,
    CompiledSkill,
    CompiledSkillPackage,
    LocalEnvironment,
    QueryPlan,
    SkillAsset,
    SkillFragment,
    SkillGraph,
    SkillRelation,
    Subgoal,
)


SIMILARITY_THRESHOLD = 0.5


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

        # Prefer SkillNet-style explicit relationships (for example from relationships.json).
        # Only fall back to heuristic inference when the local library has no declared graph.
        if explicit_relation_count == 0:
            inferred = self._infer_similarity(skill_map)
            relations.extend(inferred)
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
        return min(1.0, required_hit * 0.8 + optional_hit * 0.2)


@dataclass
class SkillGraphCompiler:
    scorer: SkillUtilityScorer
    min_relevance: float = 0.25
    preserve_top_k: int = 0
    similar_prune_margin: float = 0.08
    keep_parent_if_better_by: float = 0.05

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
        dropped: Dict[str, str] = {}
        ranked = sorted(scores, key=lambda skill_id: scores[skill_id], reverse=True)
        selected: Set[str] = {
            skill_id for skill_id, score in scores.items() if score >= self.min_relevance
        }
        protected: Set[str] = set()
        if self.preserve_top_k > 0:
            protected = set(ranked[: self.preserve_top_k])
            selected |= protected

        for skill_id in graph.skills:
            if skill_id not in selected:
                dropped[skill_id] = "relevance_below_threshold"

        selected = self._prune_similar(graph, selected, scores, dropped)
        selected = self._prune_broad_containers(graph, selected, scores, dropped)
        selected = self._add_dependencies(graph, selected)
        selected = self._trim_isolated(graph, selected, scores, dropped, protected, query_plan)

        compiled_skills = [
            CompiledSkill(
                asset=graph.skills[skill_id],
                selected_fragments=matched_fragments.get(skill_id, []),
                assigned_subgoals=self._assigned_subgoals(skill_id, matched_fragments, subgoals),
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

        notes = [
            "Similar skills were merged by utility score.",
            "Dependencies were reintroduced after pruning to preserve executability.",
            "Instructions were localized against the current environment.",
        ]
        return CompiledSkillPackage(
            query_plan=query_plan,
            subgoals=subgoals,
            graph=compiled_graph,
            compiled_skills=compiled_skills,
            execution_order=execution_order,
            metrics=metrics,
            dropped_skills=dropped,
            notes=notes,
        )

    def _prune_similar(
        self,
        graph: SkillGraph,
        selected: Set[str],
        scores: Dict[str, float],
        dropped: Dict[str, str],
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
    ) -> Set[str]:
        retained = set(selected)
        for relation in graph.relations:
            if relation.relation_type != "belong_to":
                continue
            child = relation.source
            parent = relation.target
            if child in retained and parent in retained and scores[child] >= scores[parent] + self.keep_parent_if_better_by:
                retained.remove(parent)
                dropped[parent] = f"replaced_by_more_specific_child:{child}"
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

    def _trim_isolated(
        self,
        graph: SkillGraph,
        selected: Set[str],
        scores: Dict[str, float],
        dropped: Dict[str, str],
        protected: Set[str],
        query_plan: QueryPlan,
    ) -> Set[str]:
        retained = set(selected)
        for skill_id in list(selected):
            if skill_id in protected:
                continue
            matched_required = (
                graph.skills[skill_id].normalized_capabilities() & query_plan.required_capabilities
            )
            if matched_required:
                continue
            neighbors = {
                relation.target
                for relation in graph.relations
                if relation.source == skill_id and relation.target in selected
            } | {
                relation.source
                for relation in graph.relations
                if relation.target == skill_id and relation.source in selected
            }
            if neighbors:
                continue
            if scores[skill_id] < 0.35 and len(selected) > 1:
                retained.remove(skill_id)
                dropped[skill_id] = "isolated_low_value_skill"
        return retained

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
    ) -> List[str]:
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
