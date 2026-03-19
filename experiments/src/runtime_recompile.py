from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence


FAILURE_PATTERNS = {
    "generic": (
        "nothing happens",
        "can't",
        "cannot",
        "not possible",
        "not valid",
        "unknown action",
        "no known action",
        "don't understand",
        "not sure what you are referring",
    ),
    "scienceworld": (
        "nothing happens",
        "can't",
        "cannot",
        "not possible",
        "not valid",
        "unknown action",
        "no known action",
        "not sure what you are referring",
        "you don't see",
    ),
    "alfworld": (
        "nothing happens",
        "can't",
        "cannot",
        "not possible",
        "not valid",
        "you are not carrying",
        "you can't see any such thing",
        "there is no",
    ),
    "webshop": (
        "invalid action",
        "not available",
        "not found",
        "no results",
        "error",
    ),
}

LOW_CONFIDENCE_FAILURE_PATTERNS = {
    "alfworld": (
        "nothing happens",
    ),
}


AMBIGUOUS_MARKERS = (
    "ambiguous request",
    "please enter the number",
)

ALFWORLD_VISIBLE_ENTITY_PATTERN = re.compile(r"\b(?:a|an) ([a-z]+(?: [a-z]+)*) (\d+)\b")
ALFWORLD_ARRIVAL_PATTERN = re.compile(r"you arrive at ([^.]+)\.", re.IGNORECASE)
ALFWORLD_PICKUP_PATTERN = re.compile(
    r"you pick up the ([a-z]+(?: [a-z]+)*) (\d+) from",
    re.IGNORECASE,
)
ALFWORLD_MOVE_PATTERN = re.compile(
    r"you move the ([a-z]+(?: [a-z]+)*) (\d+) to the ([^.]+)\.",
    re.IGNORECASE,
)
ALFWORLD_OPEN_PATTERN = re.compile(r"you open the ([^.]+)\.", re.IGNORECASE)
ALFWORLD_CLOSE_PATTERN = re.compile(r"you close the ([^.]+)\.", re.IGNORECASE)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _unwrap_singleton(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _unwrap_singleton(value[0])
    if hasattr(value, "shape") and hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _coerce_bool(value: Any) -> bool:
    return bool(_unwrap_singleton(value))


def _coerce_float(value: Any) -> float:
    scalar = _unwrap_singleton(value)
    try:
        return float(scalar or 0.0)
    except Exception:
        return 0.0


def _coerce_text(value: Any) -> str:
    return str(_unwrap_singleton(value) or "")


def _extract_action_text(action: Any) -> str:
    scalar = _unwrap_singleton(action)
    if isinstance(scalar, str):
        return scalar.strip()
    return _coerce_text(scalar).strip()


def _replace_action_text(action: Any, new_text: str) -> Any:
    if isinstance(action, list):
        updated = list(action)
        if updated:
            updated[0] = new_text
        else:
            updated = [new_text]
        return updated
    if isinstance(action, tuple):
        updated = list(action)
        if updated:
            updated[0] = new_text
        else:
            updated = [new_text]
        return tuple(updated)
    return new_text


def _space_indexed_tokens(text: str) -> str:
    return re.sub(r"\b([a-z]+)(\d+)\b", r"\1 \2", text, flags=re.IGNORECASE)


def _strip_articles(text: str) -> str:
    return re.sub(r"\b(?:the|a|an)\b", "", text, flags=re.IGNORECASE)


def _clean_action_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", _strip_articles(_space_indexed_tokens(text))).strip().lower()


@dataclass
class RuntimeRecompileDecision:
    reason: str
    step_index: int
    action: str
    observation: str
    selected_skills_before: List[str] = field(default_factory=list)
    steps_consumed: int = 0
    repeated_observation_count: int = 0
    repeated_action_count: int = 0
    task_done: bool = False
    task_reward: float = 0.0
    trace_tail: List[Dict[str, Any]] = field(default_factory=list)
    state_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reason": self.reason,
            "step_index": self.step_index,
            "action": self.action,
            "observation": self.observation,
            "selected_skills_before": list(self.selected_skills_before),
            "steps_consumed": self.steps_consumed,
            "repeated_observation_count": self.repeated_observation_count,
            "repeated_action_count": self.repeated_action_count,
            "task_done": self.task_done,
            "task_reward": self.task_reward,
            "trace_tail": list(self.trace_tail),
            "state_snapshot": dict(self.state_snapshot),
        }


class RuntimeSkillRecompileRequested(RuntimeError):
    def __init__(self, decision: RuntimeRecompileDecision):
        self.decision = decision
        super().__init__(
            f"Runtime skill recompile requested at step {decision.step_index}: {decision.reason}"
        )


class RuntimeRecompileController:
    def __init__(
        self,
        *,
        benchmark: str,
        enabled: bool,
        selected_skill_names: Optional[Sequence[str]] = None,
        max_total_steps: Optional[int] = None,
        total_steps_before_attempt: int = 0,
        last_recompile_step: int = -999,
        min_steps_between_recompiles: int = 2,
        stagnation_threshold: int = 2,
        min_remaining_steps_to_recompile: int = 1,
        trace_tail: int = 6,
    ):
        self.benchmark = benchmark or "generic"
        self.enabled = bool(enabled)
        self.selected_skill_names = list(selected_skill_names or [])
        self.max_total_steps = int(max_total_steps) if max_total_steps is not None else None
        self.total_steps_before_attempt = int(total_steps_before_attempt)
        self.last_recompile_step = int(last_recompile_step)
        self.min_steps_between_recompiles = max(1, int(min_steps_between_recompiles))
        self.stagnation_threshold = max(2, int(stagnation_threshold))
        self.min_remaining_steps_to_recompile = max(
            1,
            int(min_remaining_steps_to_recompile),
        )
        self.trace = deque(maxlen=max(2, int(trace_tail)))
        self.steps_consumed = 0
        self._last_observation_key = ""
        self._last_action_key = ""
        self.repeated_observation_count = 0
        self.repeated_action_count = 0

    def record_step(
        self,
        *,
        action: str,
        observation: str,
        task_done: bool,
        task_reward: float,
    ) -> Optional[RuntimeRecompileDecision]:
        self.steps_consumed += 1
        action_text = str(action or "").strip()
        observation_text = str(observation or "").strip()
        action_key = _normalize_text(action_text)
        observation_key = _normalize_text(observation_text)

        if observation_key and observation_key == self._last_observation_key:
            self.repeated_observation_count += 1
        else:
            self.repeated_observation_count = 1 if observation_key else 0

        if action_key and action_key == self._last_action_key and observation_key == self._last_observation_key:
            self.repeated_action_count += 1
        else:
            self.repeated_action_count = 1 if action_key else 0

        self._last_action_key = action_key
        self._last_observation_key = observation_key
        self.trace.append(
            {
                "step": self.total_steps_before_attempt + self.steps_consumed,
                "action": action_text,
                "observation": observation_text[:600],
            }
        )

        if not self.enabled or task_done:
            return None

        reason = self._detect_reason(observation_key)
        if reason is None:
            return None

        step_index = self.total_steps_before_attempt + self.steps_consumed
        return RuntimeRecompileDecision(
            reason=reason,
            step_index=step_index,
            action=action_text,
            observation=observation_text[:1200],
            selected_skills_before=list(self.selected_skill_names),
            steps_consumed=self.steps_consumed,
            repeated_observation_count=self.repeated_observation_count,
            repeated_action_count=self.repeated_action_count,
            task_done=bool(task_done),
            task_reward=float(task_reward or 0.0),
            trace_tail=list(self.trace),
        )

    def _detect_reason(self, observation_key: str) -> Optional[str]:
        current_step = self.total_steps_before_attempt + self.steps_consumed
        if current_step - self.last_recompile_step < self.min_steps_between_recompiles:
            return None
        if self.max_total_steps is not None:
            remaining_steps = self.max_total_steps - current_step
            if remaining_steps < self.min_remaining_steps_to_recompile:
                return None

        if any(marker in observation_key for marker in AMBIGUOUS_MARKERS):
            if self.repeated_observation_count >= self.stagnation_threshold:
                return "ambiguity_loop"
            return None

        failure_patterns = FAILURE_PATTERNS.get(self.benchmark, FAILURE_PATTERNS["generic"])
        if any(pattern in observation_key for pattern in failure_patterns):
            low_confidence_patterns = LOW_CONFIDENCE_FAILURE_PATTERNS.get(self.benchmark, ())
            if any(pattern in observation_key for pattern in low_confidence_patterns):
                if (
                    self.repeated_observation_count < self.stagnation_threshold
                    and self.repeated_action_count < self.stagnation_threshold
                ):
                    return None
            return "action_failure"

        if self.repeated_observation_count >= self.stagnation_threshold:
            return "stagnation"

        if self.repeated_action_count >= self.stagnation_threshold:
            return "action_loop"

        return None


class RuntimeRecompileEnvProxy:
    def __init__(
        self,
        env: Any,
        controller: RuntimeRecompileController,
        step_adapter: Callable[[Any, Any], Dict[str, Any]],
        benchmark: str = "generic",
    ):
        self._env = env
        self._controller = controller
        self._step_adapter = step_adapter
        self._benchmark = benchmark or "generic"
        self._latest_observation_text = ""
        self._current_location = ""
        self._latest_visible_entities: List[str] = []
        self._carried_objects: List[str] = []
        self._visited_locations: List[str] = []
        self._open_receptacles: List[str] = []
        self._completed_transfers: List[Dict[str, str]] = []
        self._completed_transforms: List[Dict[str, str]] = []

    def _repair_action(self, action_text: str) -> str:
        if self._benchmark != "alfworld":
            return action_text
        repaired = _clean_action_phrase(action_text)
        if not repaired:
            return action_text

        pick_up_match = re.match(r"^pick up (.+)$", repaired)
        if pick_up_match and self._current_location:
            obj_ref = self._repair_object_reference(pick_up_match.group(1))
            return f"take {obj_ref} from {self._current_location}"

        take_match = re.match(r"^take (.+?) from (.+)$", repaired)
        if take_match:
            obj_ref = self._repair_object_reference(take_match.group(1))
            recep_ref = _clean_action_phrase(take_match.group(2))
            return f"take {obj_ref} from {recep_ref}"

        move_match = re.match(r"^move (.+?) to (.+)$", repaired)
        if move_match:
            obj_ref = self._repair_object_reference(move_match.group(1))
            recep_ref = _clean_action_phrase(move_match.group(2))
            return f"move {obj_ref} to {recep_ref}"

        put_match = re.match(r"^put (.+?) (?:on|in|into) (.+)$", repaired)
        if put_match:
            obj_ref = self._repair_object_reference(put_match.group(1))
            recep_ref = _clean_action_phrase(put_match.group(2))
            return f"move {obj_ref} to {recep_ref}"

        tool_match = re.match(r"^(clean|heat|cool) (.+?) with (.+)$", repaired)
        if tool_match:
            verb, obj_ref, recep_ref = tool_match.groups()
            return f"{verb} {self._repair_object_reference(obj_ref)} with {_clean_action_phrase(recep_ref)}"

        if repaired.startswith("move to "):
            return f"go to {repaired[8:].strip()}"

        return repaired

    def _repair_object_reference(self, obj_ref: str) -> str:
        cleaned = _clean_action_phrase(obj_ref)
        if not cleaned:
            return cleaned
        if re.search(r"\b\d+\b", cleaned):
            return cleaned
        carried_matches = [entity for entity in self._carried_objects if entity.startswith(f"{cleaned} ")]
        carried_matches = list(dict.fromkeys(carried_matches))
        if len(carried_matches) == 1:
            return carried_matches[0]
        matches = [entity for entity in self._latest_visible_entities if entity.startswith(f"{cleaned} ")]
        deduped_matches = list(dict.fromkeys(matches))
        if len(deduped_matches) == 1:
            return deduped_matches[0]
        return cleaned

    def _remember_location(self, location: str):
        cleaned = _clean_action_phrase(location)
        if cleaned and cleaned not in self._visited_locations:
            self._visited_locations.append(cleaned)

    def _remember_open_receptacle(self, recep: str):
        cleaned = _clean_action_phrase(recep)
        if cleaned and cleaned not in self._open_receptacles:
            self._open_receptacles.append(cleaned)

    def _forget_open_receptacle(self, recep: str):
        cleaned = _clean_action_phrase(recep)
        self._open_receptacles = [item for item in self._open_receptacles if item != cleaned]

    def _remember_carried_object(self, obj_ref: str):
        cleaned = _clean_action_phrase(obj_ref)
        if cleaned and cleaned not in self._carried_objects:
            self._carried_objects.append(cleaned)

    def _forget_carried_object(self, obj_ref: str):
        cleaned = _clean_action_phrase(obj_ref)
        self._carried_objects = [item for item in self._carried_objects if item != cleaned]

    def _remember_transfer(self, obj_ref: str, destination: str):
        obj_clean = _clean_action_phrase(obj_ref)
        dest_clean = _clean_action_phrase(destination)
        if not obj_clean or not dest_clean:
            return
        self._completed_transfers.append({"object": obj_clean, "destination": dest_clean})

    def _remember_transform(self, verb: str, obj_ref: str, target: str):
        obj_clean = _clean_action_phrase(obj_ref)
        target_clean = _clean_action_phrase(target)
        if not obj_clean:
            return
        self._completed_transforms.append(
            {
                "verb": verb,
                "object": obj_clean,
                "target": target_clean,
            }
        )

    def _build_state_snapshot(self) -> Dict[str, Any]:
        return {
            "benchmark": self._benchmark,
            "current_location": self._current_location,
            "latest_observation": self._latest_observation_text[:600],
            "visible_entities": list(self._latest_visible_entities),
            "inventory": list(self._carried_objects),
            "visited_locations": list(self._visited_locations),
            "open_receptacles": list(self._open_receptacles),
            "completed_transfers": list(self._completed_transfers[-6:]),
            "completed_transforms": list(self._completed_transforms[-6:]),
        }

    def _update_state_from_action(self, action_text: str, observation_text: str):
        if self._benchmark != "alfworld":
            return
        cleaned_action = _clean_action_phrase(action_text)
        observation_key = _normalize_text(observation_text)
        if not cleaned_action or any(
            pattern in observation_key for pattern in FAILURE_PATTERNS.get("alfworld", ())
        ):
            return

        open_match = re.match(r"^open (.+)$", cleaned_action)
        if open_match:
            self._remember_open_receptacle(open_match.group(1))

        close_match = re.match(r"^close (.+)$", cleaned_action)
        if close_match:
            self._forget_open_receptacle(close_match.group(1))

        take_match = re.match(r"^take (.+?) from (.+)$", cleaned_action)
        if take_match:
            self._remember_carried_object(take_match.group(1))

        move_match = re.match(r"^move (.+?) to (.+)$", cleaned_action)
        if move_match:
            self._forget_carried_object(move_match.group(1))
            self._remember_transfer(move_match.group(1), move_match.group(2))

        transform_match = re.match(r"^(clean|heat|cool) (.+?) with (.+)$", cleaned_action)
        if transform_match:
            verb, obj_ref, target = transform_match.groups()
            self._remember_transform(verb, obj_ref, target)

    def _update_context(self, action_text: str, observation_text: str):
        normalized_observation = _coerce_text(observation_text)
        if not normalized_observation:
            return
        self._latest_observation_text = normalized_observation
        arrival_match = ALFWORLD_ARRIVAL_PATTERN.search(normalized_observation)
        if arrival_match:
            self._current_location = _clean_action_phrase(arrival_match.group(1))
            self._remember_location(self._current_location)
        self._latest_visible_entities = [
            f"{name.strip().lower()} {number}"
            for name, number in ALFWORLD_VISIBLE_ENTITY_PATTERN.findall(normalized_observation.lower())
        ]
        open_match = ALFWORLD_OPEN_PATTERN.search(normalized_observation)
        if open_match:
            self._remember_open_receptacle(open_match.group(1))
        close_match = ALFWORLD_CLOSE_PATTERN.search(normalized_observation)
        if close_match:
            self._forget_open_receptacle(close_match.group(1))
        pickup_match = ALFWORLD_PICKUP_PATTERN.search(normalized_observation)
        if pickup_match:
            self._remember_carried_object(
                f"{pickup_match.group(1).strip().lower()} {pickup_match.group(2)}"
            )
        move_match = ALFWORLD_MOVE_PATTERN.search(normalized_observation)
        if move_match:
            moved_object = f"{move_match.group(1).strip().lower()} {move_match.group(2)}"
            destination = move_match.group(3)
            self._forget_carried_object(moved_object)
            self._remember_transfer(moved_object, destination)

    def step(self, action: Any):
        original_action_text = _extract_action_text(action)
        repaired_action_text = self._repair_action(original_action_text)
        action_payload = action
        if repaired_action_text and repaired_action_text != original_action_text:
            action_payload = _replace_action_text(action, repaired_action_text)

        result = self._env.step(action_payload)
        snapshot = self._step_adapter(action_payload, result)
        executed_action = _coerce_text(snapshot.get("action", repaired_action_text or original_action_text))
        observation_text = _coerce_text(snapshot.get("observation", ""))
        self._update_context(executed_action, observation_text)
        self._update_state_from_action(executed_action, observation_text)
        decision = self._controller.record_step(
            action=executed_action,
            observation=observation_text,
            task_done=_coerce_bool(snapshot.get("task_done", False)),
            task_reward=_coerce_float(snapshot.get("task_reward", 0.0)),
        )
        if decision is not None:
            decision.state_snapshot = self._build_state_snapshot()
            raise RuntimeSkillRecompileRequested(decision)
        return result

    def __getattr__(self, name: str):
        return getattr(self._env, name)


def append_observation_message(messages: List[Dict[str, Any]], observation: str):
    observation_text = str(observation or "").strip()
    if not observation_text:
        return
    content = f"Observation: {observation_text}"
    if messages and messages[-1].get("role") == "user" and messages[-1].get("content") == content:
        return
    messages.append({"role": "user", "content": content})


def execute_compiled_procedure(
    *,
    env: Any,
    llm: Callable[..., Any],
    model: str,
    task_prompt: str,
    messages: List[Dict[str, Any]],
    max_steps: int,
    skill_module: Any,
    selected_skill_names: Sequence[str],
    step_adapter: Callable[[Any, Any], Dict[str, Any]],
    invoke: Callable[[Callable[..., Any], Any, List[Dict[str, Any]], int], Any],
    progress_callback: Optional[Callable[..., Any]] = None,
    max_script_retries: int = 3,
) -> Dict[str, Any]:
    total_steps = 0
    task_done = False
    task_reward = 0.0
    active_task_prompt = task_prompt
    active_skill_names = list(selected_skill_names)
    overall_procedure = ""
    overall_procedure_code = ""

    while total_steps < max_steps:
        if progress_callback:
            progress_callback(
                "generating_procedure",
                {
                    "runtime_recompile_count": getattr(skill_module, "runtime_recompile_count", 0),
                    "active_skill_names": list(active_skill_names),
                },
            )
        overall_procedure = skill_module.generate_overall_procedure(active_task_prompt, active_skill_names)
        print(f"\n\033[94mGenerated Overall Procedure:\n{overall_procedure}\033[0m")
        if progress_callback:
            progress_callback("procedure_generated")

        retries = 0
        recompile_triggered = False
        while retries < max_script_retries:
            controller = RuntimeRecompileController(
                benchmark=skill_module._infer_benchmark(),
                enabled=skill_module.should_use_runtime_recompile(),
                selected_skill_names=active_skill_names,
                max_total_steps=max_steps,
                total_steps_before_attempt=total_steps,
                last_recompile_step=getattr(skill_module, "runtime_last_recompile_step", -999),
                min_steps_between_recompiles=getattr(
                    skill_module,
                    "runtime_recompile_min_interval_steps",
                    2,
                ),
                stagnation_threshold=getattr(
                    skill_module,
                    "runtime_recompile_stagnation_threshold",
                    2,
                ),
                min_remaining_steps_to_recompile=getattr(
                    skill_module,
                    "runtime_recompile_min_remaining_steps",
                    1,
                ),
                trace_tail=getattr(skill_module, "runtime_recompile_trace_tail", 6),
            )
            proxy = RuntimeRecompileEnvProxy(
                env,
                controller,
                step_adapter,
                benchmark=skill_module._infer_benchmark(),
            )
            try:
                if progress_callback:
                    progress_callback("generating_procedure_code", {"retry": retries})
                overall_procedure_code = skill_module.generate_overall_procedure_code(
                    active_task_prompt,
                    overall_procedure,
                )
                print(f"\n\033[94mGenerated Procedure Code:\n{overall_procedure_code}\033[0m")
                if progress_callback:
                    progress_callback("procedure_code_generated")

                namespace: Dict[str, Any] = {}
                exec(overall_procedure_code, namespace)
                func = namespace.get("overall_procedure_code")
                if func is None:
                    raise ValueError("Function 'overall_procedure_code' not found in generated code.")

                if progress_callback:
                    progress_callback("executing_procedure")
                remaining_steps = max_steps - total_steps
                messages, task_done, task_reward, local_steps = invoke(
                    func,
                    proxy,
                    messages,
                    remaining_steps,
                )
                total_steps += int(local_steps)
                return {
                    "messages": messages,
                    "task_done": task_done,
                    "task_reward": task_reward,
                    "steps": total_steps,
                    "skill_names": list(active_skill_names),
                    "overall_procedure": overall_procedure,
                    "overall_procedure_code": overall_procedure_code,
                }
            except RuntimeSkillRecompileRequested as exc:
                decision = exc.decision.to_dict()
                append_observation_message(messages, decision.get("observation", ""))
                total_steps = max(total_steps, int(decision["step_index"]))
                task_done = bool(decision.get("task_done", False))
                task_reward = max(task_reward, float(decision.get("task_reward", 0.0) or 0.0))
                skill_module.record_runtime_recompile(decision)
                recompile_triggered = True
                if progress_callback:
                    progress_callback("runtime_recompile", decision)
                if task_done or not skill_module.can_runtime_recompile():
                    return {
                        "messages": messages,
                        "task_done": task_done,
                        "task_reward": task_reward,
                        "steps": total_steps,
                        "skill_names": list(active_skill_names),
                        "overall_procedure": overall_procedure,
                        "overall_procedure_code": overall_procedure_code,
                    }
                active_task_prompt = skill_module.build_runtime_recompile_task(
                    task_prompt,
                    messages,
                    decision,
                    remaining_steps=max_steps - total_steps,
                )
                active_skill_names = skill_module.retrieve_relevant_skills(active_task_prompt) or list(active_skill_names)
                print(
                    f"\033[94mRuntime recompile activated ({decision['reason']}). "
                    f"Updated skills: {active_skill_names}\033[0m"
                )
                break
            except Exception as exc:
                print(f"Error loading/executing procedure script: {exc}")
                retries += 1

        if not recompile_triggered:
            break

    return {
        "messages": messages,
        "task_done": task_done,
        "task_reward": task_reward,
        "steps": total_steps,
        "skill_names": list(active_skill_names),
        "overall_procedure": overall_procedure,
        "overall_procedure_code": overall_procedure_code,
    }
