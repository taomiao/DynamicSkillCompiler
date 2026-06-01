"""Smoke tests for the generic runtime execution guard."""

import sys

sys.path.insert(0, "src")

from runtime_recompile import (
    RuntimeRecompileController,
    RuntimeRecompileEnvProxy,
    RuntimeSkillRecompileRequested,
    build_runtime_protocol_state,
    classify_runtime_failure,
    infer_runtime_protocol_hints,
)
from prompt_generator import generate_overall_procedure_code_prompt
from prompt_generator import generate_overall_procedure_prompt


def _adapter(action, result):
    observation, reward, done, _info = result
    return {
        "action": action[0] if isinstance(action, list) else action,
        "observation": observation,
        "task_reward": reward,
        "task_done": done,
    }


class _Env:
    def __init__(self):
        self.actions = []

    def step(self, action):
        self.actions.append(action)
        return "Nothing happens.", 0.0, False, {}


class _OkEnv(_Env):
    def step(self, action):
        self.actions.append(action)
        return "You selected an option.", 0.0, False, {}


def test_classifies_skill_name_as_action():
    result = classify_runtime_failure(
        action="use object-locator",
        selected_skill_names=["object-locator"],
    )
    assert result["failure_type"] == "skill_as_action"


def test_classifies_script_tool_as_action():
    result = classify_runtime_failure(
        action="parse_query.py",
        selected_skill_names=["query-parser"],
    )
    assert result["failure_type"] == "skill_as_action"


def test_runtime_guard_blocks_skill_action_before_env_step():
    env = _Env()
    controller = RuntimeRecompileController(
        benchmark="generic",
        enabled=True,
        selected_skill_names=["object-locator"],
        max_total_steps=10,
        min_steps_between_recompiles=1,
    )
    proxy = RuntimeRecompileEnvProxy(env, controller, _adapter, benchmark="generic")

    try:
        proxy.step(["use object-locator"])
    except RuntimeSkillRecompileRequested as exc:
        decision = exc.decision.to_dict()
        assert decision["failure_type"] == "skill_as_action"
        assert decision["step_index"] == 1
    else:
        raise AssertionError("Expected RuntimeSkillRecompileRequested")

    assert env.actions == []


def test_placeholder_action_gets_one_soft_hint_before_recompile():
    env = _Env()
    controller = RuntimeRecompileController(
        benchmark="generic",
        enabled=True,
        selected_skill_names=[],
        max_total_steps=10,
        min_steps_between_recompiles=1,
    )
    proxy = RuntimeRecompileEnvProxy(env, controller, _adapter, benchmark="generic")
    proxy._update_context("", "A state with one available final action.")

    observation, reward, done, info = proxy.step(["none"])
    assert "Runtime guard hint" in observation
    assert "Continue with one concrete legal environment action" in observation
    assert reward == 0.0
    assert done is False
    assert info["runtime_guard_hint"] is True
    assert env.actions == []

    try:
        proxy.step(["none"])
    except RuntimeSkillRecompileRequested as exc:
        decision = exc.decision.to_dict()
        assert decision["failure_type"] == "invalid_action"
        assert decision["reason"] == "empty_or_placeholder_action"
    else:
        raise AssertionError("Expected repeated placeholder to trigger recompile")

    assert env.actions == []


def test_dsc_generation_prompt_contains_generic_execution_contract():
    procedure_messages = generate_overall_procedure_prompt(
        "Find or transform the target.",
        "Example procedure.",
        "Skill contents.",
    )
    procedure_text = procedure_messages[-1]["content"]
    assert "Hard-vs-Soft Best Effort" in procedure_text
    assert "Single-Step Action Contract" in procedure_text
    assert "Latest-State Authority" in procedure_text
    assert "No Keyword-Checklist Rejection" in procedure_text

    prompt_messages = generate_overall_procedure_code_prompt(
        "Find or transform the target.",
        "Use available evidence and proceed safely.",
        "def overall_procedure_code(env, llm, model, parse_action, messages=[], max_steps=30):\n    pass",
    )
    prompt_text = prompt_messages[-1]["content"]
    assert "Single-Step Action Contract" in prompt_text
    assert "Latest-State Authority" in prompt_text
    assert "Hard-vs-Soft Best Effort" in prompt_text
    assert "No Keyword-Checklist Rejection" in prompt_text
    assert "execute exactly one legal" in prompt_text


def test_observation_failure_gets_precondition_type():
    controller = RuntimeRecompileController(
        benchmark="generic",
        enabled=True,
        selected_skill_names=[],
        max_total_steps=10,
        min_steps_between_recompiles=1,
        stagnation_threshold=1,
    )
    decision = controller.record_step(
        action="take apple 1 from table 1",
        observation="Nothing happens.",
        task_done=False,
        task_reward=0.0,
    )
    assert decision is not None
    assert decision.failure_type == "precondition_missing"


def test_search_failure_gets_search_protocol():
    result = classify_runtime_failure(
        action="search[waterproof hiking shoes size 10 blue]",
        observation="No results found.",
    )
    assert result["failure_type"] == "search_exhausted"

    hints = infer_runtime_protocol_hints(
        "Find waterproof hiking shoes in size 10.",
        {
            "failure_type": result["failure_type"],
            "action": "search[waterproof hiking shoes size 10 blue]",
            "observation": "No results found.",
        },
    )
    assert any("candidate queue" in hint for hint in hints)
    assert any("over-constrained" in hint for hint in hints)


def test_runtime_protocol_state_preserves_hard_constraints_and_candidates():
    state = build_runtime_protocol_state(
        "Find an item with color: purple, and size: x-large, and price lower than 140 dollars",
        {
            "failure_type": "stagnation",
            "action": "search[purple camera optical zoom]",
            "trace_tail": [
                {"action": "search[purple camera optical zoom]", "observation": "Page 1"},
                {"action": "look around", "observation": "Candidate list"},
            ],
            "state_snapshot": {
                "products": [
                    {
                        "id": "candidate-1",
                        "title": "Purple Digital Camera with Optical Zoom",
                        "price": "$100.0",
                    }
                ],
            },
        },
    )

    hard_constraints = state["constraints"]["hard"]
    assert "color: purple" in hard_constraints
    assert "size: x-large" in hard_constraints
    assert "price/value limit <= 140" in hard_constraints
    assert "color: purple" in state["evidence_stages"]["detail_page"]
    assert any("select/apply color: purple" in item for item in state["evidence_stages"]["final_commit"])
    assert state["phase"] == "inspect_or_commit_best_candidate"
    assert any("candidate-1" in candidate for candidate in state["candidates"])
    assert any("Preserve hard constraints" in item for item in state["next_policy"])


def test_transform_failure_gets_transform_protocol():
    result = classify_runtime_failure(
        action="cool potato 1 with fridge 1",
        observation="Nothing happens.",
    )
    assert result["failure_type"] == "transformation_precondition_missing"

    hints = infer_runtime_protocol_hints(
        "Put a cool potato in the fridge.",
        {
            "failure_type": result["failure_type"],
            "action": "cool potato 1 with fridge 1",
            "observation": "Nothing happens.",
            "state_snapshot": {"inventory": ["potato 1"]},
        },
    )
    assert any("four-step protocol" in hint for hint in hints)
    assert any("with {tool}" in hint for hint in hints)
    assert any("one-at-a-time" in hint for hint in hints)


def test_proxy_resolves_ambiguous_followup_before_env_step():
    env = _OkEnv()
    controller = RuntimeRecompileController(
        benchmark="generic",
        enabled=True,
        selected_skill_names=[],
        max_total_steps=10,
        min_steps_between_recompiles=1,
    )
    proxy = RuntimeRecompileEnvProxy(env, controller, _adapter, benchmark="generic")
    proxy._update_context(
        "",
        (
            "Ambiguous request: Please enter the number for the action you intended "
            "(or blank to cancel):\n"
            "0:\tfocus on lemon seed (in flower pot 1, in greenhouse)\n"
            "1:\tfocus on lemon seed (in flower pot 2, in greenhouse)"
        ),
    )

    proxy.step("focus on lemon seed")
    assert env.actions == ["0"]


def test_proxy_blocks_orphan_ambiguous_index():
    env = _Env()
    controller = RuntimeRecompileController(
        benchmark="generic",
        enabled=True,
        selected_skill_names=[],
        max_total_steps=10,
        min_steps_between_recompiles=1,
    )
    proxy = RuntimeRecompileEnvProxy(env, controller, _adapter, benchmark="generic")
    proxy._update_context("", "This room is called the greenhouse. In it, you see a seed jar.")

    observation, reward, done, info = proxy.step("0")
    assert "numeric index is only legal" in observation
    assert reward == 0.0
    assert done is False
    assert info["runtime_guard_hint"] is True
    assert env.actions == []


def test_proxy_does_not_force_unrelated_action_into_ambiguous_index():
    env = _OkEnv()
    controller = RuntimeRecompileController(
        benchmark="generic",
        enabled=True,
        selected_skill_names=[],
        max_total_steps=10,
        min_steps_between_recompiles=1,
    )
    proxy = RuntimeRecompileEnvProxy(env, controller, _adapter, benchmark="generic")
    proxy._update_context(
        "",
        (
            "Ambiguous request: Please enter the number for the action you intended "
            "(or blank to cancel):\n"
            "0:\tinspect sample vial (in inventory)\n"
            "1:\tinspect sample vial (on shelf)"
        ),
    )

    observation, _reward, _done, info = proxy.step("look around")
    assert "waiting for a numeric index" in observation
    assert info["runtime_guard_hint"] is True
    assert env.actions == []


def test_proxy_uses_identity_ledger_for_same_name_ambiguity():
    env = _OkEnv()
    controller = RuntimeRecompileController(
        benchmark="generic",
        enabled=True,
        selected_skill_names=[],
        max_total_steps=10,
        min_steps_between_recompiles=1,
    )
    proxy = RuntimeRecompileEnvProxy(env, controller, _adapter, benchmark="generic")
    proxy._update_context(
        "",
        (
            "a storage shelf. On the storage shelf is: "
            "a sample vial (containing blue reagent), a sample vial (containing red reagent), "
            "a sample vial (containing yellow reagent)."
        ),
    )
    proxy._update_context(
        "",
        (
            "Ambiguous request: Please enter the number for the action you intended "
            "(or blank to cancel):\n"
            "0:\tpick up sample vial (on storage shelf)\n"
            "1:\tpick up sample vial (on storage shelf)\n"
            "2:\tpick up sample vial (on storage shelf)"
        ),
    )

    proxy.step("pick up sample vial (containing red reagent)")
    assert env.actions == ["1"]


def test_proxy_corrects_numeric_choice_from_pending_ambiguous_intent():
    env = _OkEnv()
    controller = RuntimeRecompileController(
        benchmark="generic",
        enabled=True,
        selected_skill_names=[],
        max_total_steps=10,
        min_steps_between_recompiles=1,
    )
    proxy = RuntimeRecompileEnvProxy(env, controller, _adapter, benchmark="generic")
    proxy._update_context(
        "",
        (
            "This work area contains: "
            "a sample vial (containing red reagent), a sample vial (containing blue reagent)."
        ),
    )
    proxy._last_raw_action_intent = "pick up sample vial (containing blue reagent)"
    proxy._update_context(
        "pick up sample vial",
        (
            "Ambiguous request: Please enter the number for the action you intended "
            "(or blank to cancel):\n"
            "0:\tpick up sample vial (in work area)\n"
            "1:\tpick up sample vial (in work area)"
        ),
    )

    proxy.step("0")
    assert env.actions == ["1"]


def test_runtime_protocol_state_includes_state_ledger():
    state = build_runtime_protocol_state(
        "Measure the target and place it in the correct box.",
        {
            "failure_type": "reward_plateau",
            "action": "look around",
            "observation": "You see two boxes.",
            "state_snapshot": {
                "benchmark": "generic",
                "completed_actions": [
                    {"action": "pick up thermometer", "observation": "You move the thermometer to inventory."}
                ],
                "focused_targets": ["unknown substance b"],
                "last_measurements": [
                    {"action": "use thermometer on unknown substance b", "value": "20"}
                ],
                "ambiguous_options": [{"index": "0", "text": "focus on seed"}],
            },
        },
    )

    assert state["completed_actions"]
    assert state["focused_targets"] == ["unknown substance b"]
    assert state["last_measurements"][0]["value"] == "20"
    assert state["ambiguous_options"][0]["index"] == "0"
    assert any("state ledger" in item for item in state["next_policy"])


if __name__ == "__main__":
    test_classifies_skill_name_as_action()
    test_classifies_script_tool_as_action()
    test_runtime_guard_blocks_skill_action_before_env_step()
    test_placeholder_action_gets_one_soft_hint_before_recompile()
    test_dsc_generation_prompt_contains_generic_execution_contract()
    test_observation_failure_gets_precondition_type()
    test_search_failure_gets_search_protocol()
    test_runtime_protocol_state_preserves_hard_constraints_and_candidates()
    test_transform_failure_gets_transform_protocol()
    test_proxy_resolves_ambiguous_followup_before_env_step()
    test_proxy_blocks_orphan_ambiguous_index()
    test_proxy_does_not_force_unrelated_action_into_ambiguous_index()
    test_proxy_uses_identity_ledger_for_same_name_ambiguity()
    test_proxy_corrects_numeric_choice_from_pending_ambiguous_intent()
    test_runtime_protocol_state_includes_state_ledger()
    print("runtime execution guard tests passed")
