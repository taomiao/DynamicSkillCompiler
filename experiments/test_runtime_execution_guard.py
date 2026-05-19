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


def test_classifies_skill_name_as_action():
    result = classify_runtime_failure(
        action="use alfworld-object-locator",
        selected_skill_names=["alfworld-object-locator"],
    )
    assert result["failure_type"] == "skill_as_action"


def test_classifies_script_tool_as_action():
    result = classify_runtime_failure(
        action="parse_query.py",
        benchmark="webshop",
        selected_skill_names=["webshop-query-parser"],
    )
    assert result["failure_type"] == "skill_as_action"


def test_runtime_guard_blocks_skill_action_before_env_step():
    env = _Env()
    controller = RuntimeRecompileController(
        benchmark="alfworld",
        enabled=True,
        selected_skill_names=["alfworld-object-locator"],
        max_total_steps=10,
        min_steps_between_recompiles=1,
    )
    proxy = RuntimeRecompileEnvProxy(env, controller, _adapter, benchmark="alfworld")

    try:
        proxy.step(["use alfworld-object-locator"])
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
        benchmark="webshop",
        enabled=True,
        selected_skill_names=[],
        max_total_steps=10,
        min_steps_between_recompiles=1,
    )
    proxy = RuntimeRecompileEnvProxy(env, controller, _adapter, benchmark="webshop")
    proxy._update_context("", "Product page [SEP] Buy Now")

    observation, reward, done, info = proxy.step(["none"])
    assert "Runtime guard hint" in observation
    assert "click[Buy Now]" in observation
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


def test_webshop_repairs_title_click_to_asin():
    env = _Env()
    controller = RuntimeRecompileController(
        benchmark="webshop",
        enabled=True,
        selected_skill_names=[],
        max_total_steps=10,
    )
    proxy = RuntimeRecompileEnvProxy(env, controller, _adapter, benchmark="webshop")
    proxy._update_context(
        "",
        "Page 1 [SEP] B09PVNLVRW [SEP] Women's V-Neck Rompers Printed Jumpsuit Long Sleeve Homewear [SEP] $17.4 [SEP] Next >",
    )
    repaired = proxy._repair_action(
        "click[Women's V-Neck Rompers Printed Jumpsuit Long Sleeve Homewear]"
    )
    assert repaired == "click[B09PVNLVRW]"
    assert proxy._repair_action("click[next]") == "click[Next >]"


def test_observation_failure_gets_precondition_type():
    controller = RuntimeRecompileController(
        benchmark="alfworld",
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
        benchmark="webshop",
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
        "Find a product with color: purple, and size: x-large, and price lower than 140 dollars",
        {
            "failure_type": "stagnation",
            "action": "click[Next >]",
            "trace_tail": [
                {"action": "search[purple camera optical zoom]", "observation": "Page 1"},
                {"action": "click[Next >]", "observation": "Page 2"},
            ],
            "state_snapshot": {
                "benchmark": "webshop",
                "webshop_products": [
                    {
                        "asin": "B004HO58MA",
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
    assert any("B004HO58MA" in candidate for candidate in state["candidates"])
    assert any("Preserve hard constraints" in item for item in state["next_policy"])


def test_webshop_search_gets_commitment_protocol():
    hints = infer_runtime_protocol_hints(
        "WebShop [SEP] Instruction: Find a black jumpsuit under 60 [SEP] Search",
        {
            "failure_type": "exploration_without_commit",
            "action": "click[Next >]",
            "observation": "Page 3 (Total results: 50) [SEP] B123 [SEP] Black Romper [SEP] $19.99",
            "trace_tail": [
                {"action": "click[Next >]", "observation": "Page 2"},
                {"action": "click[Next >]", "observation": "Page 3"},
            ],
        },
    )
    assert any("candidate queue" in hint for hint in hints)
    assert any("best-so-far candidate" in hint for hint in hints)
    assert not any("1-2 result pages" in hint for hint in hints)


def test_exploration_commit_guard_is_progress_based_not_page_based():
    controller = RuntimeRecompileController(
        benchmark="webshop",
        enabled=True,
        selected_skill_names=[],
        max_total_steps=20,
        min_steps_between_recompiles=1,
        exploration_commit_enabled=True,
        exploration_commit_threshold=3,
    )
    product_snapshot = {
        "webshop_products": [
            {"asin": "B09PVNLVRW", "title": "Black Romper", "price": "$19.99"}
        ]
    }

    assert controller.record_step(
        action="search[black romper under 60]",
        observation="Page 1 [SEP] B09PVNLVRW [SEP] Black Romper [SEP] $19.99",
        task_done=False,
        task_reward=0.0,
        state_snapshot=product_snapshot,
    ) is None
    assert controller.record_step(
        action="click[Next >]",
        observation="Page 2 [SEP] B09PVNLVRW [SEP] Black Romper [SEP] $19.99",
        task_done=False,
        task_reward=0.0,
        state_snapshot=product_snapshot,
    ) is None
    decision = controller.record_step(
        action="search[black jumpsuit under 60]",
        observation="Page 1 [SEP] B09PVNLVRW [SEP] Black Romper [SEP] $19.99",
        task_done=False,
        task_reward=0.0,
        state_snapshot=product_snapshot,
    )
    assert decision is not None
    assert decision.reason == "exploration_without_commit"
    assert decision.failure_type == "exploration_without_commit"

    controller = RuntimeRecompileController(
        benchmark="webshop",
        enabled=True,
        selected_skill_names=[],
        max_total_steps=20,
        min_steps_between_recompiles=1,
        exploration_commit_enabled=True,
        exploration_commit_threshold=3,
    )
    assert controller.record_step(
        action="search[black romper under 60]",
        observation="Page 1 [SEP] B09PVNLVRW [SEP] Black Romper [SEP] $19.99",
        task_done=False,
        task_reward=0.0,
        state_snapshot=product_snapshot,
    ) is None
    assert controller.record_step(
        action="click[B09PVNLVRW]",
        observation="Product page [SEP] Buy Now",
        task_done=False,
        task_reward=0.0,
        state_snapshot={},
    ) is None
    assert controller.horizontal_exploration_count == 0


def test_silent_distinct_commit_actions_do_not_trigger_stagnation():
    controller = RuntimeRecompileController(
        benchmark="webshop",
        enabled=True,
        selected_skill_names=[],
        max_total_steps=20,
        min_steps_between_recompiles=1,
        stagnation_threshold=1,
    )
    product_page = {
        "benchmark": "webshop",
        "webshop_products": [],
    }
    observation = "Product page [SEP] color [SEP] black [SEP] size [SEP] x-large [SEP] Buy Now"

    assert controller.record_step(
        action="click[black]",
        observation=observation,
        task_done=False,
        task_reward=0.0,
        state_snapshot=product_page,
    ) is None
    assert controller.record_step(
        action="click[x-large]",
        observation=observation,
        task_done=False,
        task_reward=0.0,
        state_snapshot=product_page,
    ) is None

    decision = controller.record_step(
        action="click[x-large]",
        observation=observation,
        task_done=False,
        task_reward=0.0,
        state_snapshot=product_page,
    )
    assert decision is not None
    assert decision.reason == "action_loop"


def test_transform_failure_gets_transform_protocol():
    result = classify_runtime_failure(
        action="cool potato 1 with fridge 1",
        observation="Nothing happens.",
        benchmark="alfworld",
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


if __name__ == "__main__":
    test_classifies_skill_name_as_action()
    test_classifies_script_tool_as_action()
    test_runtime_guard_blocks_skill_action_before_env_step()
    test_placeholder_action_gets_one_soft_hint_before_recompile()
    test_webshop_repairs_title_click_to_asin()
    test_observation_failure_gets_precondition_type()
    test_search_failure_gets_search_protocol()
    test_runtime_protocol_state_preserves_hard_constraints_and_candidates()
    test_webshop_search_gets_commitment_protocol()
    test_exploration_commit_guard_is_progress_based_not_page_based()
    test_silent_distinct_commit_actions_do_not_trigger_stagnation()
    test_transform_failure_gets_transform_protocol()
    print("runtime execution guard tests passed")
