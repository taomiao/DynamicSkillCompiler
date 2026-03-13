from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List

from skillnet_ai.compiler.models import SkillAsset, SkillFragment, Subgoal


LINE_SPLIT = re.compile(r"[.\n]")
ACTION_QUOTED = re.compile(r"`([^`]+)`")
ACTION_PREFIXES = (
    "go to ",
    "take ",
    "move ",
    "open ",
    "close ",
    "use ",
    "clean ",
    "heat ",
    "cool ",
    "focus on ",
    "activate ",
    "pour ",
    "mix ",
    "measure ",
    "search ",
    "click ",
    "select ",
    "buy now",
    "add to cart",
    "filter ",
    "sort ",
)


@dataclass
class SkillFragmentExtractor:
    max_fragments_per_skill: int = 6

    def extract(self, skills: Iterable[SkillAsset]) -> Dict[str, List[SkillFragment]]:
        fragment_map: Dict[str, List[SkillFragment]] = {}
        for skill in skills:
            fragments = self._extract_for_skill(skill)
            fragment_map[skill.skill_id] = fragments
        return fragment_map

    def _extract_for_skill(self, skill: SkillAsset) -> List[SkillFragment]:
        lines: List[str] = []
        for instruction in skill.instructions:
            for line in LINE_SPLIT.split(instruction):
                cleaned = line.strip(" -*\t")
                if cleaned:
                    lines.append(cleaned)
        if not lines:
            lines.append(skill.description)

        fragments: List[SkillFragment] = []
        for index, line in enumerate(lines[: self.max_fragments_per_skill]):
            capabilities = {
                token.strip(".,:;()[]{}").lower()
                for token in line.split()
                if len(token.strip(".,:;()[]{}")) > 2
            }
            example_actions = self._extract_actions(line)
            preconditions = self._extract_preconditions(line)
            postconditions = self._extract_postconditions(line)
            action_schema = self._extract_action_schema(line, example_actions)
            fragments.append(
                SkillFragment(
                    fragment_id=f"{skill.skill_id}::frag_{index + 1}",
                    skill_id=skill.skill_id,
                    title=f"{skill.name} fragment {index + 1}",
                    content=line,
                    capabilities=capabilities or skill.normalized_capabilities(),
                    action_schema=action_schema,
                    preconditions=preconditions,
                    postconditions=postconditions,
                    example_actions=example_actions,
                    token_cost=max(1.0, len(line.split()) / 20.0),
                    metadata={"source_skill": skill.name},
                )
            )
        return fragments

    def _extract_actions(self, line: str) -> List[str]:
        matches = ACTION_QUOTED.findall(line)
        if matches:
            return matches
        if "action:" in line.lower():
            return [line.split(":", 1)[1].strip()]
        lowered = line.lower()
        for prefix in ACTION_PREFIXES:
            idx = lowered.find(prefix)
            if idx != -1:
                return [line[idx:].strip()]
        return []

    def _extract_preconditions(self, line: str) -> List[str]:
        lowered = line.lower()
        if lowered.startswith(("if ", "ensure ", "verify ", "when ", "requires ")):
            return [line]
        if "before " in lowered or "must " in lowered:
            return [line]
        return []

    def _extract_postconditions(self, line: str) -> List[str]:
        lowered = line.lower()
        if lowered.startswith(("output", "result", "return", "the agent receives", "the specified object is")):
            return [line]
        if "output" in lowered or "confirm" in lowered or "transferred" in lowered:
            return [line]
        return []

    def _extract_action_schema(self, line: str, example_actions: List[str]) -> str | None:
        if example_actions:
            return example_actions[0]
        lowered = line.lower()
        for prefix in ACTION_PREFIXES:
            if prefix in lowered:
                start = lowered.find(prefix)
                return line[start:].strip()
        return None


@dataclass
class FragmentMatcher:
    subgoal_weight: float = 0.75
    optional_weight: float = 0.25

    def match(self, subgoal: Subgoal, fragments: List[SkillFragment]) -> List[SkillFragment]:
        ranked = sorted(
            fragments,
            key=lambda fragment: self.score(subgoal, fragment),
            reverse=True,
        )
        return [fragment for fragment in ranked if self.score(subgoal, fragment) > 0]

    def score(self, subgoal: Subgoal, fragment: SkillFragment) -> float:
        required = subgoal.required_capabilities
        optional = subgoal.optional_capabilities
        frag_caps = fragment.capabilities
        required_hit = len(required & frag_caps) / max(len(required), 1)
        optional_hit = len(optional & frag_caps) / max(len(optional), 1) if optional else 0.0
        action_bonus = 0.15 if fragment.example_actions else 0.0
        return self.subgoal_weight * required_hit + self.optional_weight * optional_hit + action_bonus
