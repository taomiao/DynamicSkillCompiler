from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List

from dynamic_skill_compiler.models import SkillAsset, SkillFragment, Subgoal


LINE_SPLIT = re.compile(r"[.\n]")
ACTION_QUOTED = re.compile(r"`([^`]+)`")
ACTION_PREFIXES = (
    "teleport to ",
    "pick up ",
    "go to ",
    "take ",
    "move ",
    "open ",
    "close ",
    "use ",
    "look at ",
    "examine ",
    "turn on ",
    "turn off ",
    "deactivate ",
    "wait1",
    "wait",
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
LOW_SIGNAL_PREFIXES = (
    "purpose",
    "when to use",
    "notes",
    "key parameters",
    "key considerations",
    "integration with other skills",
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
        indexed_lines: List[tuple[int, str]] = []
        for instruction in skill.instructions:
            for line in LINE_SPLIT.split(instruction):
                cleaned = line.strip(" -*\t")
                if cleaned:
                    indexed_lines.append((len(indexed_lines), cleaned))
        lines = [line for _, line in indexed_lines]
        if not lines:
            lines.append(skill.description)
            indexed_lines = [(0, skill.description)]

        selected_lines = self._select_fragment_lines(indexed_lines)

        fragments: List[SkillFragment] = []
        for index, line in enumerate(selected_lines):
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

    def _select_fragment_lines(self, indexed_lines: List[tuple[int, str]]) -> List[str]:
        if len(indexed_lines) <= self.max_fragments_per_skill:
            return [line for _, line in indexed_lines]

        ranked = sorted(
            indexed_lines,
            key=lambda item: (
                self._line_priority(item[1]),
                -item[0],
            ),
            reverse=True,
        )
        chosen = sorted(
            ranked[: self.max_fragments_per_skill],
            key=lambda item: item[0],
        )
        return [line for _, line in chosen]

    def _line_priority(self, line: str) -> int:
        lowered = line.lower()
        priority = 0
        if self._extract_actions(line):
            priority += 8
        if self._extract_preconditions(line):
            priority += 3
        if self._extract_postconditions(line):
            priority += 2
        if any(marker in lowered for marker in ("command", "action pattern", "execute", "retry", "verify")):
            priority += 2
        if any(lowered.startswith(prefix) for prefix in LOW_SIGNAL_PREFIXES):
            priority -= 2
        return priority

    def _extract_actions(self, line: str) -> List[str]:
        matches = ACTION_QUOTED.findall(line)
        if matches:
            return matches
        lowered = line.lower()
        for marker in ("action:", "command:", "action pattern:", "example:"):
            if marker in lowered:
                candidate = line[lowered.find(marker) + len(marker):].strip()
                if candidate:
                    return [candidate]
        if "action:" in lowered:
            return [line.split(":", 1)[1].strip()]
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
