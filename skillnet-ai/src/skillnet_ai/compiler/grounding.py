from __future__ import annotations

from dataclasses import dataclass
from typing import List

from skillnet_ai.compiler.models import LocalEnvironment, SkillFragment, Subgoal


DOMAIN_ACTIONS = {}


@dataclass
class EnvironmentGrounder:
    def ground_fragment(
        self,
        fragment: SkillFragment,
        subgoal: Subgoal,
        environment: LocalEnvironment,
    ) -> SkillFragment:
        grounded_actions = self._ground_actions(fragment.example_actions, environment)
        grounded_content = self._ground_text(fragment.content, environment, subgoal)
        return SkillFragment(
            fragment_id=fragment.fragment_id,
            skill_id=fragment.skill_id,
            title=fragment.title,
            content=grounded_content,
            capabilities=set(fragment.capabilities),
            preconditions=list(fragment.preconditions),
            postconditions=list(fragment.postconditions),
            example_actions=grounded_actions,
            token_cost=fragment.token_cost,
            metadata={**fragment.metadata, "grounded_for": environment.benchmark or "generic"},
        )

    def _ground_actions(self, actions: List[str], environment: LocalEnvironment) -> List[str]:
        if not actions:
            return []
        benchmark = environment.benchmark or "generic"
        allowed = DOMAIN_ACTIONS.get(benchmark, set())
        if not allowed:
            return actions
        grounded = []
        for action in actions:
            if any(action.lower().startswith(prefix) for prefix in allowed):
                grounded.append(action)
        return grounded or actions[:1]

    def _ground_text(self, text: str, environment: LocalEnvironment, subgoal: Subgoal) -> str:
        grounded = text.replace("{cwd}", environment.cwd)
        grounded = grounded.replace("{workspace_root}", environment.workspace_root)
        if environment.object_vocabulary:
            grounded += f" [env_objects={', '.join(sorted(list(environment.object_vocabulary)[:5]))}]"
        if subgoal.environment_hints.get("domain"):
            grounded += f" [domain={subgoal.environment_hints['domain']}]"
        return grounded
