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
        "no results",
        "not found",
        "no matches",
        "0 results",
        "there is no",
        "you can't see any such thing",
    ),
}

LOW_CONFIDENCE_FAILURE_PATTERNS = {}

AMBIGUOUS_MARKERS = (
    "ambiguous request",
    "please enter the number",
)

AMBIGUOUS_OPTION_PATTERN = re.compile(r"(\d+):\s*(.*?)(?=\s+\d+:|$)", re.DOTALL)

CAPACITY_MARKERS = (
    "inventory is full",
    "carrying too much",
    "already carrying",
    "already holding",
    "hands are full",
    "too many things",
)

SEARCH_EXHAUSTED_MARKERS = (
    "no results",
    "not found",
    "no matches",
    "0 results",
    "there is no",
    "you can't see any such thing",
)

PRICE_LIMIT_PATTERN = re.compile(
    r"\b(?:price\s+)?(?:lower than|less than|under|below|at most|no more than)\s+\$?\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
ATTRIBUTE_PATTERN = re.compile(
    r"\b(?:with\s+)?(color|colour|size|fit type|type|material|location|destination)\s*:\s*([^,.;\n]+)",
    re.IGNORECASE,
)
SOFT_DESCRIPTOR_PATTERN = re.compile(
    r"\b(?:for|with)\s+([a-z][a-z0-9 -]{2,40}?)\s+(?:with|and|,|\.|$)",
    re.IGNORECASE,
)


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


def _normalize_reward_value(value: Any) -> float:
    reward = _coerce_float(value)
    if reward > 1.0:
        reward = reward / 100.0
    return max(0.0, min(reward, 1.0))


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


def _strip_parenthetical_details(text: str) -> str:
    """Remove observation-only qualifiers that are not valid action syntax."""
    return re.sub(r"\s*\([^)]*\)", "", str(text or "")).strip()


def _extract_ambiguous_options(observation: str) -> List[Dict[str, str]]:
    text = str(observation or "").strip()
    if text.lower().startswith("observation:"):
        text = text.split(":", 1)[1].strip()
    key = _normalize_text(text)
    if not all(marker in key for marker in AMBIGUOUS_MARKERS):
        return []
    compact = re.sub(r"\s+", " ", text)
    options: List[Dict[str, str]] = []
    for match in AMBIGUOUS_OPTION_PATTERN.finditer(compact):
        options.append(
            {
                "index": match.group(1).strip(),
                "text": match.group(2).strip(),
            }
        )
    return options


def _resolve_ambiguous_followup(action_text: str, options: Sequence[Dict[str, str]]) -> Optional[str]:
    """Return the numeric option index for an ambiguous prompt when it is safe."""
    stripped = str(action_text or "").strip()
    if not options:
        return None
    if re.fullmatch(r"\d+", stripped):
        valid_indices = {str(option.get("index", "")).strip() for option in options}
        return stripped if stripped in valid_indices else None

    action_norm = _normalize_text(stripped)
    action_core = _normalize_text(_strip_parenthetical_details(stripped))
    for option in options:
        idx = str(option.get("index", "")).strip()
        option_text = str(option.get("text", "")).strip()
        option_norm = _normalize_text(option_text)
        option_core = _normalize_text(_strip_parenthetical_details(option_text))
        if action_norm and (action_norm == option_norm or action_norm in option_norm):
            return idx
        if action_core and (action_core == option_core or action_core in option_core):
            return idx

    unique_cores = {
        _normalize_text(_strip_parenthetical_details(option.get("text", "")))
        for option in options
        if option.get("text")
    }
    if len(unique_cores) == 1:
        option_core = next(iter(unique_cores))
        action_verb = action_core.split(" ", 1)[0] if action_core else ""
        option_verb = option_core.split(" ", 1)[0] if option_core else ""
        if action_verb and action_verb == option_verb:
            return str(options[0].get("index", "")).strip() or None
    return None


CONTAINING_ENTITY_PATTERN = re.compile(
    r"\b(?P<name>[A-Za-z][A-Za-z0-9 -]{1,60}?)\s*\(containing (?P<contents>[^)]{1,160})\)",
    re.IGNORECASE,
)


def _content_terms(text: str) -> set:
    stopwords = {
        "a",
        "an",
        "and",
        "called",
        "containing",
        "in",
        "nothing",
        "of",
        "substance",
        "the",
    }
    return {
        term
        for term in re.findall(r"[a-z][a-z0-9-]*", str(text or "").lower())
        if len(term) > 1 and term not in stopwords
    }


def _normalize_surface_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _surface_term_set(text: str, *, stopwords: Optional[set] = None) -> set:
    terms = set()
    for term in _normalize_surface_key(text).split():
        if len(term) <= 2 or (stopwords and term in stopwords):
            continue
        terms.add(term)
        if term.endswith("ies") and len(term) > 4:
            terms.add(term[:-3] + "y")
        elif term.endswith("s") and not term.endswith("ss") and len(term) > 3:
            terms.add(term[:-1])
    return terms


def _bracket_action_target(action_key: str) -> str:
    match = re.match(r"^click\[(.*)\]$", action_key, re.IGNORECASE)
    return _normalize_surface_key(match.group(1)) if match else ""


def _is_horizontal_exploration_action(action_key: str, benchmark: str = "generic") -> bool:
    """Actions that expand the search space without verifying a specific candidate."""
    key = _normalize_text(action_key)
    if not key:
        return False
    if key.startswith("search["):
        return True
    return key in {"look", "look around", "inventory"} or key.startswith(
        ("go to ", "teleport ", "move to ")
    )


def _is_candidate_commit_action(action_key: str, benchmark: str = "generic") -> bool:
    """Actions that inspect, select, transform, or otherwise verify a candidate."""
    key = _normalize_text(action_key)
    if not key:
        return False
    return key.startswith("click[") or key.startswith(
        (
            "examine ",
            "look at ",
            "open ",
            "take ",
            "pick up ",
            "focus on ",
            "activate ",
            "deactivate ",
            "use ",
            "mix ",
            "pour ",
            "measure ",
            "heat ",
            "cool ",
            "clean ",
            "slice ",
            "move ",
            "put ",
            "place ",
        )
    )


def _snapshot_has_candidates(snapshot: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(snapshot, dict):
        return False
    candidate_keys = ("candidates", "products", "visible_entities")
    return any(bool(snapshot.get(key)) for key in candidate_keys)


def _format_list(items: Sequence[Any], *, limit: int = 6) -> str:
    cleaned = [str(item).strip() for item in items if str(item).strip()]
    if not cleaned:
        return "- none"
    lines = [f"- {item}" for item in cleaned[:limit]]
    if len(cleaned) > limit:
        lines.append(f"- ... ({len(cleaned) - limit} more)")
    return "\n".join(lines)


def _infer_constraint_ledger(task: str) -> Dict[str, List[str]]:
    """Extract a conservative, benchmark-neutral constraint ledger from the task text.

    The ledger is intentionally lightweight: it only promotes explicit limits and
    labeled attributes to hard constraints, while leaving fuzzy descriptors as
    soft constraints. It is used as a runtime prompt scaffold, not as a parser of
    record.
    """
    task_text = str(task or "")
    hard: List[str] = []
    soft: List[str] = []
    unknown: List[str] = []

    for match in PRICE_LIMIT_PATTERN.finditer(task_text):
        hard.append(f"price/value limit <= {match.group(1)}")

    for label, value in ATTRIBUTE_PATTERN.findall(task_text):
        cleaned_label = re.sub(r"\s+", " ", label.lower()).strip()
        cleaned_value = re.sub(r"\s+", " ", value).strip()
        if cleaned_value:
            hard.append(f"{cleaned_label}: {cleaned_value}")

    # Preserve quoted/labeled exact identifiers without trying to understand the domain.
    for quoted in re.findall(r"[\"'`“”‘’]([^\"'`“”‘’]{2,80})[\"'`“”‘’]", task_text):
        hard.append(f"exact mention: {quoted.strip()}")

    for descriptor in SOFT_DESCRIPTOR_PATTERN.findall(task_text):
        cleaned = re.sub(r"\s+", " ", descriptor.lower()).strip(" -")
        if cleaned and not any(cleaned in item.lower() for item in hard):
            soft.append(cleaned)

    if re.search(r"\b(find|locate|search|look for|bring|get|put|place|move|buy|purchase)\b", task_text, re.I):
        unknown.append("candidate identity and availability must be verified from observation")
    if re.search(r"\b(color|size|price|location|destination|container|receptacle|tool|appliance)\b", task_text, re.I):
        unknown.append("attribute satisfaction needs direct observation or state evidence before final commit")

    return {
        "hard": list(dict.fromkeys(hard))[:8],
        "soft": list(dict.fromkeys(soft))[:6],
        "unknown": list(dict.fromkeys(unknown))[:6],
    }


def _infer_evidence_stages(constraints: Dict[str, List[str]]) -> Dict[str, List[str]]:
    hard = constraints.get("hard") or []
    soft = constraints.get("soft") or []
    result_page: List[str] = []
    detail_page: List[str] = []
    final_commit: List[str] = []

    for item in hard:
        key = item.lower()
        if "price/value limit" in key:
            result_page.append(item)
            final_commit.append(f"re-check {item}")
        elif any(marker in key for marker in ("color:", "colour:", "size:", "fit type:", "material:")):
            detail_page.append(item)
            final_commit.append(f"select/apply {item} if exposed as an option")
        elif any(marker in key for marker in ("location:", "destination:", "tool:", "container:", "receptacle:")):
            detail_page.append(item)
            final_commit.append(f"verify {item} before final action")
        else:
            result_page.append(item)
            detail_page.append(item)

    for item in soft:
        detail_page.append(f"soft evidence: {item}")

    if not result_page:
        result_page.append("rough candidate category/type and visible budget only")
    if not detail_page:
        detail_page.append("inspect details/options/state before rejecting plausible candidates")
    if not final_commit:
        final_commit.append("before final commit, verify required explicit options/states are selected")

    return {
        "result_page": list(dict.fromkeys(result_page))[:8],
        "detail_page": list(dict.fromkeys(detail_page))[:8],
        "final_commit": list(dict.fromkeys(final_commit))[:8],
    }


def _extract_candidate_queue(snapshot: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(snapshot, dict):
        return []
    candidates: List[str] = []
    for candidate in snapshot.get("candidates") or []:
        candidates.append(str(candidate))
    for product in snapshot.get("products") or []:
        if not isinstance(product, dict):
            candidates.append(str(product))
            continue
        identifier = str(product.get("id", "") or product.get("identifier", "")).strip()
        title = str(product.get("title", "")).strip()
        price = str(product.get("price", "")).strip()
        label = " | ".join(part for part in (identifier, title, price) if part)
        if label:
            candidates.append(label)
    for entity in snapshot.get("visible_entities") or []:
        candidates.append(str(entity))
    for entity in snapshot.get("inventory") or []:
        candidates.append(f"held: {entity}")
    return list(dict.fromkeys(candidates))[:8]


def build_runtime_protocol_state(
    task: str,
    decision: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    decision = decision or {}
    snapshot = decision.get("state_snapshot") or {}
    trace = decision.get("trace_tail") or []
    benchmark = str(snapshot.get("benchmark") or "generic")
    action = _normalize_text(decision.get("action", ""))
    failure_type = _normalize_text(decision.get("failure_type", decision.get("reason", "")))
    candidates = _extract_candidate_queue(snapshot)
    completed_actions = snapshot.get("completed_actions") or []
    focused_targets = snapshot.get("focused_targets") or []
    last_measurements = snapshot.get("last_measurements") or []
    object_identity_ledger = snapshot.get("object_identity_ledger") or []
    visible_actions = [str(item).strip() for item in (snapshot.get("visible_actions") or []) if str(item).strip()][:8]
    trace_actions = [
        _normalize_text(item.get("action", ""))
        for item in trace
        if isinstance(item, dict)
    ]
    horizontal_steps = sum(1 for item in trace_actions if _is_horizontal_exploration_action(item, benchmark))
    commit_steps = sum(1 for item in trace_actions if _is_candidate_commit_action(item, benchmark))

    if "selection_acknowledgement_loop" in failure_type:
        phase = "advance_after_selection"
    elif "invalid_action" in failure_type or "stale_state" in failure_type or "precondition" in failure_type:
        phase = "recover"
    elif "exploration_without_commit" in failure_type or (candidates and horizontal_steps >= 2 and commit_steps == 0):
        phase = "inspect_or_commit_best_candidate"
    elif candidates and _is_candidate_commit_action(action, benchmark):
        phase = "verify_candidate"
    elif candidates:
        phase = "inspect_candidate"
    else:
        phase = "explore"

    next_policy: List[str] = []
    next_policy.append("Preserve hard constraints; do not silently relax them during recovery.")
    if visible_actions:
        next_policy.append("Choose the next environment action from the latest visible action set; do not invent actions from older observations.")
    if candidates:
        next_policy.append("Use the candidate queue before widening exploration; reject a candidate only with fresh contradictory evidence.")
    if phase in {"inspect_or_commit_best_candidate", "inspect_candidate"}:
        next_policy.append("Inspect or commit the best-so-far candidate if hard constraints have sufficient evidence and no direct contradiction.")
    if phase == "recover":
        next_policy.append("Repair the smallest failed precondition or stale target; avoid restarting solved phases.")
    if completed_actions or focused_targets or last_measurements:
        next_policy.append("Preserve the state ledger: do not redo completed actions, focus targets, or measured facts unless the latest observation explicitly contradicts them.")
    if object_identity_ledger:
        next_policy.append("Use observed object-identity evidence such as contents/location to resolve same-name candidates; if identity cannot be distinguished, inspect or refresh rather than guessing.")
    if phase == "advance_after_selection":
        next_policy.append("Use the latest observation as the action authority: if a repeated target is still visible, treat the prior click as acknowledged and advance to a different visible next action; if it is no longer visible, treat the target as stale and choose only from visible actions.")
    if not candidates:
        next_policy.append("Explore with rewritten queries/actions first, then relax only soft constraints if no candidates appear.")

    constraints = _infer_constraint_ledger(task)
    return {
        "phase": phase,
        "constraints": constraints,
        "evidence_stages": _infer_evidence_stages(constraints),
        "candidates": candidates,
        "visible_actions": visible_actions,
        "completed_actions": list(completed_actions)[-8:],
        "focused_targets": list(focused_targets)[-8:],
        "last_measurements": list(last_measurements)[-6:],
        "ambiguous_options": list(snapshot.get("ambiguous_options") or [])[:8],
        "object_identity_ledger": list(object_identity_ledger)[-8:],
        "tried_actions": list(dict.fromkeys([item for item in trace_actions if item]))[-8:],
        "next_policy": next_policy,
    }


@dataclass
class RuntimeRecompileDecision:
    reason: str
    step_index: int
    action: str
    observation: str
    failure_type: str = ""
    repair_hint: str = ""
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
            "failure_type": self.failure_type,
            "repair_hint": self.repair_hint,
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


FORBIDDEN_ACTION_PREFIXES = (
    "abort",
    "abort:",
    "error:",
    "report failure",
    "done",
    "task complete",
)

SKILL_AS_ACTION_PATTERNS = (
    "[invoke",
    "invoke ",
    "call skill",
    "run skill",
    "trigger skill",
    "use skill",
    ".py",
    "parse_query",
    "object-locator",
    "task-verifier",
    "state-inspector",
    "search-pattern-executor",
)

def classify_runtime_failure(
    *,
    action: str,
    observation: str = "",
    benchmark: str = "generic",
    selected_skill_names: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    """Return a portable failure class and repair hint for action/observation pairs."""
    action_key = _normalize_text(action)
    observation_key = _normalize_text(observation)
    skill_names = [str(name).lower() for name in (selected_skill_names or [])]

    if not action_key or action_key in {"-", "none", "null"}:
        return {
            "failure_type": "invalid_action",
            "reason": "empty_or_placeholder_action",
            "repair_hint": "Generate a concrete environment action from the legal action space; do not emit placeholders.",
        }

    if any(action_key.startswith(prefix) for prefix in FORBIDDEN_ACTION_PREFIXES):
        return {
            "failure_type": "premature_termination",
            "reason": "non_environment_termination_action",
            "repair_hint": "Do not send abort/done/report-failure as an environment action; continue with the next verifiable action until reward or task_done changes.",
        }

    if (
        any(pattern in action_key for pattern in SKILL_AS_ACTION_PATTERNS)
        or any(name and name in action_key for name in skill_names)
    ):
        return {
            "failure_type": "skill_as_action",
            "reason": "skill_name_used_as_environment_action",
            "repair_hint": "Skills may only provide plans, constraints, and candidates. Convert skill advice into a legal environment action before env.step.",
        }

    if any(marker in observation_key for marker in AMBIGUOUS_MARKERS):
        return {
            "failure_type": "ambiguity_loop",
            "reason": "ambiguous_reference",
            "repair_hint": "Resolve ambiguity using the environment's requested numeric/index response, then resume the prior subgoal.",
        }

    failure_patterns = FAILURE_PATTERNS.get(benchmark, FAILURE_PATTERNS["generic"])
    if any(pattern in observation_key for pattern in failure_patterns):
        if any(marker in observation_key for marker in CAPACITY_MARKERS):
            failure_type = "capacity_blocked"
            repair_hint = "Preserve the carried object; deliver/place one held item or free capacity before taking another."
        elif action_key.startswith("search") and any(
            marker in observation_key for marker in SEARCH_EXHAUSTED_MARKERS
        ):
            failure_type = "search_exhausted"
            repair_hint = "Broaden the query or advance to the next candidate; do not repeat the same over-constrained search."
        elif action_key.startswith(("heat ", "cool ", "clean ")):
            failure_type = "transformation_precondition_missing"
            repair_hint = "Verify the object is held/nearby, locate the required appliance/tool, apply the transform once, then verify before delivery."
        elif action_key.startswith(("take ", "pick ", "grab ")):
            failure_type = "precondition_missing"
            repair_hint = "Refresh the latest observation, verify proximity/container state/object id, then retry or choose a visible alternative."
        elif action_key.startswith(("put ", "move ", "place ")):
            failure_type = "precondition_missing"
            repair_hint = "Verify the object is held/available and the destination is reachable/open before placing it."
        elif action_key.startswith(("go to ", "click", "search", "open ", "close ")):
            failure_type = "stale_state_or_invalid_target"
            repair_hint = "Refresh state and choose the next candidate rather than repeating a failed target blindly."
        else:
            failure_type = "action_failure"
            repair_hint = "Repair only the failed local step; preserve completed subgoals and avoid restarting solved phases."
        return {
            "failure_type": failure_type,
            "reason": "environment_rejected_action",
            "repair_hint": repair_hint,
        }

    return {"failure_type": "", "reason": "", "repair_hint": ""}


def infer_runtime_protocol_hints(
    task: str,
    decision: Optional[Dict[str, Any]],
    *,
    max_hints: int = 5,
) -> List[str]:
    """Infer portable protocol reminders for the next runtime recompile patch."""
    decision = decision or {}
    snapshot = decision.get("state_snapshot") or {}
    trace = decision.get("trace_tail") or []
    task_key = _normalize_text(task)
    action_key = _normalize_text(decision.get("action", ""))
    observation_key = _normalize_text(decision.get("observation", ""))
    failure_type = _normalize_text(decision.get("failure_type", decision.get("reason", "")))
    combined = " ".join([task_key, action_key, observation_key, failure_type])
    trace_actions = [
        _normalize_text(item.get("action", ""))
        for item in trace
        if isinstance(item, dict)
    ]
    repeated_actions = {
        action for action in trace_actions if action and trace_actions.count(action) >= 2
    }
    hints: List[str] = []

    def add(text: str):
        if text and text not in hints and len(hints) < max_hints:
            hints.append(text)

    search_trigger = (
        "search_exhausted" in failure_type
        or "stale_state_or_invalid_target" in failure_type
        or "exploration_without_commit" in failure_type
        or "stagnation" in failure_type
        or any(marker in combined for marker in SEARCH_EXHAUSTED_MARKERS)
        or any(word in task_key for word in ("find", "locate", "search", "look for"))
    )
    if search_trigger:
        add(
            "Use a candidate queue: refresh observation/results, mark failed targets or queries as tried, then choose the next visible/listed candidate instead of repeating the same action."
        )
        add(
            "When search is over-constrained, drop secondary adjectives/filters first, keep the core target requirement, and inspect the candidate state before selecting."
        )
        add(
            "Use an explore-then-commit rhythm: after several horizontal exploration steps without new information, stop expanding and verify the current best-so-far candidate before exploring again."
        )
    transform_trigger = (
        "transformation" in failure_type
        or re.search(r"\b(heat|cool|clean|wash|slice|toggle|turn on|turn off)\b", combined)
        or any(word in task_key for word in ("hot", "cold", "cool", "clean", "heated", "cooled"))
    )
    if transform_trigger:
        add(
            "For transform tasks, use the four-step protocol: acquire/bring the object, locate the required appliance or tool, apply exactly one legal transform action, then verify state before delivery."
        )
        add(
            "If the legal transform syntax is `verb {obj} with {tool}`, keep `{obj}` held/available and apply the `with` action directly; do not load, drop, or place the object into the tool unless the environment explicitly requires it."
        )
        add(
            "If a transform action fails, do not switch goals; repair the missing precondition such as location, open/closed appliance state, held object, or exact object id."
        )

    inventory = snapshot.get("inventory") or []
    transport_trigger = (
        bool(inventory)
        or "capacity_blocked" in failure_type
        or re.search(r"\b(two|all|another|each|both)\b", task_key)
        or action_key.startswith(("take ", "pick ", "grab ", "move ", "put ", "place "))
    )
    if transport_trigger:
        add(
            "Respect capacity and one-at-a-time transport: finish the current carried object's destination/transform before taking a different object."
        )
        add(
            "For multi-object tasks, repeat a short loop per item: locate visible candidate, acquire it, satisfy transform/precondition, place it, then return for the next item."
        )

    if "ambigu" in failure_type or any(marker in observation_key for marker in AMBIGUOUS_MARKERS):
        add(
            "Resolve ambiguity with only the requested numeric/index response, then resume the same subgoal without changing target semantics."
        )

    if "selection_acknowledgement_loop" in failure_type:
        add(
            "A repeated legal selection with unchanged observation should be treated as acknowledged only when the target is still supported by the latest observation; advance to a different visible option, verification action, or final commit instead of repeating it."
        )

    if repeated_actions:
        sample = sorted(repeated_actions)[0]
        if _is_candidate_commit_action(sample, str(snapshot.get("benchmark") or "generic")):
            add(
                f"The selection/commit action `{sample}` already repeated; do not issue it again. Use the latest observation to choose a different visible option, verification action, navigation action, or final commit."
            )
        else:
            add(
                f"The action `{sample}` already repeated; before trying it again, change candidate, location, query, or missing precondition."
            )

    return hints


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
        reward_plateau_threshold: int = 5,
        reward_plateau_min_progress: float = 0.2,
        high_progress_reward_threshold: float = 0.7,
        exploration_commit_enabled: bool = False,
        exploration_commit_threshold: int = 4,
        trace_tail: int = 6,
    ):
        self.benchmark = benchmark or "generic"
        self.enabled = bool(enabled)
        self.selected_skill_names = list(selected_skill_names or [])
        self.max_total_steps = int(max_total_steps) if max_total_steps is not None else None
        self.total_steps_before_attempt = int(total_steps_before_attempt)
        self.last_recompile_step = int(last_recompile_step)
        self.min_steps_between_recompiles = max(1, int(min_steps_between_recompiles))
        self.stagnation_threshold = max(1, int(stagnation_threshold))
        self.min_remaining_steps_to_recompile = max(
            1,
            int(min_remaining_steps_to_recompile),
        )
        self.reward_plateau_threshold = max(3, int(reward_plateau_threshold))
        self.reward_plateau_min_progress = max(0.0, float(reward_plateau_min_progress))
        self.high_progress_reward_threshold = max(0.0, float(high_progress_reward_threshold))
        self.exploration_commit_enabled = bool(exploration_commit_enabled)
        self.exploration_commit_threshold = max(2, int(exploration_commit_threshold))
        self.trace = deque(maxlen=max(2, int(trace_tail)))
        self.steps_consumed = 0
        self._last_observation_key = ""
        self._last_action_key = ""
        self.repeated_observation_count = 0
        self.repeated_action_count = 0
        self.best_task_reward = 0.0
        self.best_task_reward_normalized = 0.0
        self.steps_since_reward_improvement = 0
        self.horizontal_exploration_count = 0
        self.has_seen_candidates = False

    def record_step(
        self,
        *,
        action: str,
        observation: str,
        task_done: bool,
        task_reward: float,
        state_snapshot: Optional[Dict[str, Any]] = None,
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
        reward_value = float(task_reward or 0.0)
        reward_normalized = _normalize_reward_value(task_reward)
        if reward_value > self.best_task_reward + 1e-6:
            self.best_task_reward = reward_value
            self.best_task_reward_normalized = reward_normalized
            self.steps_since_reward_improvement = 0
        else:
            self.steps_since_reward_improvement += 1
        self.trace.append(
            {
                "step": self.total_steps_before_attempt + self.steps_consumed,
                "action": action_text,
                "observation": observation_text[:600],
            }
        )
        has_candidates = _snapshot_has_candidates(state_snapshot)
        self.has_seen_candidates = self.has_seen_candidates or has_candidates
        if _is_candidate_commit_action(action_key, self.benchmark):
            self.horizontal_exploration_count = 0
        elif _is_horizontal_exploration_action(action_key, self.benchmark):
            if self.has_seen_candidates:
                self.horizontal_exploration_count += 1
        elif observation_key:
            self.horizontal_exploration_count = 0

        if not self.enabled or task_done:
            return None

        classification = classify_runtime_failure(
            action=action_text,
            observation=observation_text,
            benchmark=self.benchmark,
            selected_skill_names=self.selected_skill_names,
        )
        reason = self._detect_reason(action_key, observation_key, classification)
        if reason is None:
            return None
        if reason == "exploration_without_commit":
            classification = {
                "failure_type": "exploration_without_commit",
                "reason": "horizontal_exploration_without_commit",
                "repair_hint": "Verify the best-so-far candidate before continuing broad exploration; preserve tried candidates and only resume exploration after that candidate is rejected by fresh observation.",
            }
        elif reason == "action_loop" and _is_candidate_commit_action(action_key, self.benchmark):
            classification = {
                "failure_type": "selection_acknowledgement_loop",
                "reason": "repeated_selection_without_observable_change",
                "repair_hint": (
                    "The repeated legal selection may already be applied even if the observation text did not change. "
                    "Do not retry the same selection. Continue with the next required option, verification step, or final commit unless the latest observation explicitly rejects the selection."
                ),
            }
            reason = "selection_acknowledgement_loop"

        step_index = self.total_steps_before_attempt + self.steps_consumed
        return RuntimeRecompileDecision(
            reason=reason,
            step_index=step_index,
            action=action_text,
            observation=observation_text[:1200],
            failure_type=classification.get("failure_type", ""),
            repair_hint=classification.get("repair_hint", ""),
            selected_skills_before=list(self.selected_skill_names),
            steps_consumed=self.steps_consumed,
            repeated_observation_count=self.repeated_observation_count,
            repeated_action_count=self.repeated_action_count,
            task_done=bool(task_done),
            task_reward=float(task_reward or 0.0),
            trace_tail=list(self.trace),
            state_snapshot=dict(state_snapshot or {}),
        )

    def record_guard_violation(
        self,
        *,
        action: str,
        failure_type: str,
        reason: str,
        repair_hint: str,
    ) -> Optional[RuntimeRecompileDecision]:
        if not self.enabled:
            return None
        self.steps_consumed += 1
        action_text = str(action or "").strip()
        step_index = self.total_steps_before_attempt + self.steps_consumed
        if step_index - self.last_recompile_step < self.min_steps_between_recompiles:
            return None
        if self.max_total_steps is not None:
            remaining_steps = self.max_total_steps - step_index
            if remaining_steps < self.min_remaining_steps_to_recompile:
                return None
        observation = f"Runtime guard blocked non-executable action: {reason}. {repair_hint}"
        self.trace.append(
            {
                "step": step_index,
                "action": action_text,
                "observation": observation[:600],
            }
        )
        return RuntimeRecompileDecision(
            reason=reason or "invalid_action",
            step_index=step_index,
            action=action_text,
            observation=observation,
            failure_type=failure_type or "invalid_action",
            repair_hint=repair_hint,
            selected_skills_before=list(self.selected_skill_names),
            steps_consumed=self.steps_consumed,
            task_done=False,
            task_reward=self.best_task_reward,
            trace_tail=list(self.trace),
        )

    def record_guard_hint(
        self,
        *,
        action: str,
        reason: str,
        repair_hint: str,
    ) -> int:
        self.steps_consumed += 1
        action_text = str(action or "").strip()
        step_index = self.total_steps_before_attempt + self.steps_consumed
        observation = f"Runtime guard hint: {reason}. {repair_hint}"
        self.trace.append(
            {
                "step": step_index,
                "action": action_text,
                "observation": observation[:600],
            }
        )
        return step_index

    def _detect_reason(
        self,
        action_key: str,
        observation_key: str,
        classification: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        current_step = self.total_steps_before_attempt + self.steps_consumed
        if current_step - self.last_recompile_step < self.min_steps_between_recompiles:
            return None
        if self.max_total_steps is not None:
            remaining_steps = self.max_total_steps - current_step
            if remaining_steps < self.min_remaining_steps_to_recompile:
                return None

        # Require at least 2 consecutive identical observations before ambiguity / stagnation /
        # action-loop triggers, so stagnation_threshold=1 means "fire on first *repeat*" not on step 1.
        min_repeat = max(2, self.stagnation_threshold)

        if any(marker in observation_key for marker in AMBIGUOUS_MARKERS):
            if self.repeated_observation_count >= min_repeat:
                return "ambiguity_loop"
            return None

        classification = classification or {}
        failure_type = classification.get("failure_type")
        if failure_type in {"skill_as_action", "invalid_action", "premature_termination"}:
            return classification.get("reason") or failure_type

        if failure_type == "exploration_without_commit":
            return classification.get("reason") or failure_type

        failure_patterns = FAILURE_PATTERNS.get(self.benchmark, FAILURE_PATTERNS["generic"])
        if any(pattern in observation_key for pattern in failure_patterns):
            low_confidence_patterns = LOW_CONFIDENCE_FAILURE_PATTERNS.get(self.benchmark, ())
            if any(pattern in observation_key for pattern in low_confidence_patterns):
                if (
                    self.repeated_observation_count < self.stagnation_threshold
                    and self.repeated_action_count < self.stagnation_threshold
                ):
                    return None
            return failure_type or "action_failure"

        if self.best_task_reward_normalized >= self.high_progress_reward_threshold:
            return None

        if (
            self.exploration_commit_enabled
            and self.has_seen_candidates
            and self.horizontal_exploration_count >= self.exploration_commit_threshold
        ):
            return "exploration_without_commit"

        if (
            self.best_task_reward_normalized >= self.reward_plateau_min_progress
            and self.steps_since_reward_improvement >= self.reward_plateau_threshold
        ):
            return "reward_plateau"

        # Some environments acknowledge valid candidate/option selections without changing
        # the rendered observation. Do not interrupt a sequence of different commit actions
        # solely because the page text stayed identical; repeated identical commit actions
        # are still caught by the action-loop check below.
        if (
            self.repeated_observation_count >= min_repeat
            and _is_candidate_commit_action(action_key, self.benchmark)
            and self.repeated_action_count < min_repeat
        ):
            return None

        if self.repeated_action_count >= min_repeat:
            return "action_loop"

        if self.repeated_observation_count >= min_repeat:
            return "stagnation"

        return None


class RuntimeRecompileEnvProxy:
    def __init__(
        self,
        env: Any,
        controller: RuntimeRecompileController,
        step_adapter: Callable[[Any, Any], Dict[str, Any]],
        benchmark: str = "generic",
        initial_state: Optional[Dict[str, Any]] = None,
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
        self._completed_actions: List[Dict[str, str]] = []
        self._focused_targets: List[str] = []
        self._last_measurements: List[Dict[str, str]] = []
        self._latest_ambiguous_options: List[Dict[str, str]] = []
        self._object_identity_ledger: List[Dict[str, Any]] = []
        self._pending_ambiguous_intent = ""
        self._last_raw_action_intent = ""
        self._soft_guard_counts: Dict[str, int] = {}
        self._load_state_snapshot(initial_state or {})

    def _load_state_snapshot(self, snapshot: Dict[str, Any]):
        if not isinstance(snapshot, dict):
            return
        self._current_location = _coerce_text(snapshot.get("current_location", self._current_location))
        self._latest_observation_text = _coerce_text(snapshot.get("latest_observation", self._latest_observation_text))
        self._latest_visible_entities = list(snapshot.get("visible_entities") or self._latest_visible_entities)
        self._carried_objects = list(snapshot.get("inventory") or self._carried_objects)
        self._visited_locations = list(snapshot.get("visited_locations") or self._visited_locations)
        self._open_receptacles = list(snapshot.get("open_receptacles") or self._open_receptacles)
        self._completed_transfers = list(snapshot.get("completed_transfers") or self._completed_transfers)
        self._completed_transforms = list(snapshot.get("completed_transforms") or self._completed_transforms)
        self._completed_actions = list(snapshot.get("completed_actions") or self._completed_actions)
        self._focused_targets = list(snapshot.get("focused_targets") or self._focused_targets)
        self._last_measurements = list(snapshot.get("last_measurements") or self._last_measurements)
        self._latest_ambiguous_options = list(
            snapshot.get("ambiguous_options") or self._latest_ambiguous_options
        )
        self._object_identity_ledger = list(
            snapshot.get("object_identity_ledger") or self._object_identity_ledger
        )
        self._pending_ambiguous_intent = _coerce_text(
            snapshot.get("pending_ambiguous_intent", self._pending_ambiguous_intent)
        )

    def _repair_action(self, action_text: str) -> str:
        if re.fullmatch(r"\d+", str(action_text or "").strip()) and self._pending_ambiguous_intent:
            intent_resolution = self._resolve_ambiguous_with_identity(self._pending_ambiguous_intent)
            if intent_resolution is not None:
                return intent_resolution
        identity_resolution = self._resolve_ambiguous_with_identity(action_text)
        if identity_resolution is not None:
            return identity_resolution
        ambiguous_resolution = _resolve_ambiguous_followup(action_text, self._latest_ambiguous_options)
        if ambiguous_resolution is not None:
            return ambiguous_resolution
        stripped = _strip_parenthetical_details(action_text)
        return stripped if stripped else action_text

    def _execution_controller_hint(self, action_text: str) -> str:
        if not self._controller.enabled:
            return ""
        if self._latest_ambiguous_options and not re.fullmatch(r"\d+", str(action_text or "").strip()):
            return (
                "Execution controller hint: the environment is waiting for a numeric index from the active "
                "ambiguous-choice prompt. Reply with one listed index only; do not issue a different action "
                "until the ambiguity is resolved."
            )
        if re.fullmatch(r"\d+", str(action_text or "").strip()) and not self._latest_ambiguous_options:
            return (
                "Execution controller hint: a numeric index is only legal immediately after an active "
                "ambiguous-choice prompt. The latest observation is not an ambiguity prompt, so refresh "
                "or issue one concrete legal action from the latest observation instead of sending an orphan index."
            )
        return ""

    def _note_action_before_step(self, action_text: str):
        return

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

    def _remember_object_identities(self, observation_text: str):
        text = str(observation_text or "")
        if not text:
            return
        occurrence_counts: Dict[tuple, int] = {}

        def remember(name: str, contents: str, location: str = ""):
            obj = _clean_action_phrase(name.split(":")[-1] if ":" in name else name)
            loc = _clean_action_phrase(location)
            key = (obj, loc)
            occurrence_counts[key] = occurrence_counts.get(key, 0) + 1
            self._remember_object_identity(
                obj,
                contents,
                location=loc,
                occurrence=occurrence_counts[key],
            )

        for loc_match in re.finditer(
            r"\bin (?:the )?([A-Za-z][A-Za-z0-9 -]{1,50}) is:\s*([^.\n]+)",
            text,
            re.IGNORECASE,
        ):
            location = _clean_action_phrase(loc_match.group(1))
            body = loc_match.group(2)
            for entity_match in CONTAINING_ENTITY_PATTERN.finditer(body):
                remember(entity_match.group("name"), entity_match.group("contents"), location)
        for entity_match in CONTAINING_ENTITY_PATTERN.finditer(text):
            name = entity_match.group("name")
            remember(name, entity_match.group("contents"))

    def _remember_object_identity(
        self,
        name: str,
        contents: str,
        *,
        location: str = "",
        occurrence: int = 0,
    ):
        name_clean = _clean_action_phrase(name)
        content_clean = re.sub(r"\s+", " ", str(contents or "").strip().lower())
        if not name_clean or not content_clean:
            return
        terms = sorted(_content_terms(content_clean))
        record = {
            "object": name_clean,
            "contents": content_clean,
            "terms": terms,
            "location": _clean_action_phrase(location),
            "occurrence": int(occurrence or 0),
        }
        key = (record["object"], record["contents"], record["location"], record["occurrence"])
        existing = [
            item
            for item in self._object_identity_ledger
            if (
                item.get("object"),
                item.get("contents"),
                item.get("location"),
                item.get("occurrence"),
            )
            != key
        ]
        existing.append(record)
        self._object_identity_ledger = existing[-16:]

    def _ambiguous_option_occurrence(self, option: Dict[str, str], obj: str, loc: str) -> int:
        option_text = str(option.get("text", "") or "")
        option_index = str(option.get("index", "") or "")
        option_key = _normalize_text(option_text)
        obj_key = _normalize_text(obj)
        loc_key = _normalize_text(loc)
        count = 0
        for candidate in self._latest_ambiguous_options:
            text = str(candidate.get("text", "") or "")
            text_key = _normalize_text(text)
            if obj_key and obj_key not in text_key:
                continue
            if loc_key and loc_key not in text_key:
                continue
            count += 1
            if str(candidate.get("index", "") or "") == option_index:
                return count
        return 0

    def _resolve_ambiguous_with_identity(self, action_text: str) -> Optional[str]:
        if not self._latest_ambiguous_options or not self._object_identity_ledger:
            return None
        action = str(action_text or "").strip()
        if re.fullmatch(r"\d+", action):
            return None
        action_core = _normalize_text(_strip_parenthetical_details(action))
        action_terms = _content_terms(action)
        action_core_terms = _content_terms(action_core)
        identity_term_counts: Dict[str, int] = {}
        for identity in self._object_identity_ledger:
            for term in set(identity.get("terms") or []):
                identity_term_counts[term] = identity_term_counts.get(term, 0) + 1
        shared_identity_terms = {term for term, count in identity_term_counts.items() if count > 1}
        target_terms = action_terms - action_core_terms - shared_identity_terms
        if not action_core or not action_terms:
            return None
        scored: List[tuple] = []
        for option in self._latest_ambiguous_options:
            option_text = str(option.get("text", "") or "")
            option_core = _normalize_text(_strip_parenthetical_details(option_text))
            if not option_core or option_core.split(" ", 1)[0] != action_core.split(" ", 1)[0]:
                continue
            score = 0
            for identity in self._object_identity_ledger:
                obj = str(identity.get("object") or "")
                if obj and obj not in option_core:
                    continue
                terms = set(identity.get("terms") or [])
                overlap = len((target_terms or action_terms) & terms)
                if overlap <= 0:
                    continue
                score += 10 * overlap
                loc = str(identity.get("location") or "")
                occurrence = int(identity.get("occurrence") or 0)
                option_occurrence = self._ambiguous_option_occurrence(option, obj, loc)
                if loc and loc in _normalize_text(option_text):
                    score += 3
                if occurrence and option_occurrence:
                    score += 6 if occurrence == option_occurrence else -6
                if "inventory" in _normalize_text(option_text):
                    for completed in reversed(self._completed_actions[-6:]):
                        completed_action = _normalize_text(completed.get("action", ""))
                        completed_obs = _normalize_text(completed.get("observation", ""))
                        if obj and obj in completed_action and terms & _content_terms(completed_action + " " + completed_obs):
                            score += 2
                            break
            if score > 0:
                scored.append((score, str(option.get("index", "")).strip()))
        if not scored:
            return None
        scored.sort(reverse=True)
        if len(scored) == 1 or scored[0][0] > scored[1][0]:
            return scored[0][1] or None
        return None

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

    def _remember_completed_action(self, action_text: str, observation_text: str):
        action_clean = re.sub(r"\s+", " ", str(action_text or "").strip())
        if not action_clean:
            return
        observation_clean = re.sub(r"\s+", " ", str(observation_text or "").strip())[:180]
        record = {"action": action_clean, "observation": observation_clean}
        if self._completed_actions and self._completed_actions[-1] == record:
            return
        self._completed_actions.append(record)
        self._completed_actions = self._completed_actions[-12:]

        action_key = _normalize_text(action_clean)
        focus_match = re.match(r"^focus on (.+)$", action_key)
        if focus_match:
            target = _clean_action_phrase(focus_match.group(1))
            if target and target not in self._focused_targets:
                self._focused_targets.append(target)
                self._focused_targets = self._focused_targets[-8:]

        measurement_match = re.search(
            r"\b(?:temperature|reading|measure(?:s|d)?)\b.*?(-?\d+(?:\.\d+)?)\s*(?:degrees?|celsius|c)\b",
            str(observation_text or ""),
            re.IGNORECASE,
        )
        if measurement_match:
            self._last_measurements.append(
                {
                    "action": action_clean,
                    "value": measurement_match.group(1),
                    "observation": observation_clean,
                }
            )
            self._last_measurements = self._last_measurements[-6:]

    def _observation_is_failure(self, observation_text: str) -> bool:
        observation_key = _normalize_text(observation_text)
        if not observation_key:
            return False
        if any(marker in observation_key for marker in AMBIGUOUS_MARKERS):
            return True
        failure_patterns = FAILURE_PATTERNS.get(self._benchmark) or FAILURE_PATTERNS["generic"]
        return any(pattern in observation_key for pattern in failure_patterns)

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
            "completed_actions": list(self._completed_actions[-10:]),
            "focused_targets": list(self._focused_targets[-8:]),
            "last_measurements": list(self._last_measurements[-6:]),
            "ambiguous_options": list(self._latest_ambiguous_options[:8]),
            "object_identity_ledger": list(self._object_identity_ledger[-12:]),
            "pending_ambiguous_intent": self._pending_ambiguous_intent,
        }

    def _update_state_from_action(self, action_text: str, observation_text: str):
        if not self._observation_is_failure(observation_text):
            self._remember_completed_action(action_text, observation_text)
        cleaned_action = _clean_action_phrase(action_text)
        observation_key = _normalize_text(observation_text)
        if not cleaned_action or any(
            pattern in observation_key for pattern in (FAILURE_PATTERNS.get(self._benchmark) or FAILURE_PATTERNS["generic"])
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
        self._latest_ambiguous_options = _extract_ambiguous_options(normalized_observation)
        if self._latest_ambiguous_options:
            if self._last_raw_action_intent and not re.fullmatch(r"\d+", self._last_raw_action_intent.strip()):
                self._pending_ambiguous_intent = self._last_raw_action_intent
        else:
            self._pending_ambiguous_intent = ""
        self._remember_object_identities(normalized_observation)

    def _build_guard_hint_observation(self, failure_type: str, reason: str, repair_hint: str) -> str:
        candidates = _extract_candidate_queue(self._build_state_snapshot())
        hints = [
            f"Runtime guard hint: blocked non-executable action ({reason or failure_type}).",
            repair_hint,
            "Continue with one concrete legal environment action from the latest observation.",
        ]
        if self._latest_visible_entities:
            hints.append(f"Use a visible entity/action target such as: {self._latest_visible_entities[0]}.")
        if candidates:
            hints.append("Do not summarize or stop while candidates or final actions remain.")
        return " ".join(part for part in hints if part)

    def _synthetic_guard_result(self, observation: str):
        return observation, self._controller.best_task_reward, False, {"runtime_guard_hint": True}

    def step(self, action: Any):
        original_action_text = _extract_action_text(action)
        self._last_raw_action_intent = original_action_text
        guard = classify_runtime_failure(
            action=original_action_text,
            benchmark=self._benchmark,
            selected_skill_names=self._controller.selected_skill_names,
        )
        if guard.get("failure_type") in {"skill_as_action", "invalid_action", "premature_termination"}:
            soft_hint_reasons = {"empty_or_placeholder_action", "non_environment_termination_action"}
            guard_key = guard.get("reason", "") or guard.get("failure_type", "")
            if guard_key in soft_hint_reasons and self._soft_guard_counts.get(guard_key, 0) < 1:
                self._soft_guard_counts[guard_key] = self._soft_guard_counts.get(guard_key, 0) + 1
                observation = self._build_guard_hint_observation(
                    guard.get("failure_type", ""),
                    guard.get("reason", ""),
                    guard.get("repair_hint", ""),
                )
                self._controller.record_guard_hint(
                    action=original_action_text,
                    reason=guard.get("reason", ""),
                    repair_hint=guard.get("repair_hint", ""),
                )
                return self._synthetic_guard_result(observation)
            decision = self._controller.record_guard_violation(
                action=original_action_text,
                failure_type=guard.get("failure_type", ""),
                reason=guard.get("reason", ""),
                repair_hint=guard.get("repair_hint", ""),
            )
            if decision is not None:
                decision.state_snapshot = self._build_state_snapshot()
                raise RuntimeSkillRecompileRequested(decision)

        repaired_action_text = self._repair_action(original_action_text)
        controller_hint = self._execution_controller_hint(repaired_action_text)
        if controller_hint:
            self._controller.record_guard_hint(
                action=repaired_action_text,
                reason="execution_controller",
                repair_hint=controller_hint,
            )
            return self._synthetic_guard_result(controller_hint)

        self._note_action_before_step(repaired_action_text)
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
            state_snapshot=self._build_state_snapshot(),
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
    runtime_state_snapshot: Dict[str, Any] = {}

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
            runtime_recompile_available = bool(
                getattr(skill_module, "can_runtime_recompile", lambda: False)()
            )
            controller = RuntimeRecompileController(
                benchmark=skill_module._infer_benchmark(),
                enabled=runtime_recompile_available,
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
                reward_plateau_threshold=getattr(
                    skill_module,
                    "runtime_recompile_reward_plateau_steps",
                    5,
                ),
                reward_plateau_min_progress=getattr(
                    skill_module,
                    "runtime_recompile_reward_plateau_min_progress",
                    0.2,
                ),
                high_progress_reward_threshold=getattr(
                    skill_module,
                    "runtime_recompile_high_progress_reward_threshold",
                    0.7,
                ),
                exploration_commit_enabled=getattr(
                    skill_module,
                    "runtime_recompile_exploration_commit_enabled",
                    False,
                ),
                exploration_commit_threshold=getattr(
                    skill_module,
                    "runtime_recompile_exploration_commit_threshold",
                    4,
                ),
                trace_tail=getattr(skill_module, "runtime_recompile_trace_tail", 6),
            )
            proxy = RuntimeRecompileEnvProxy(
                env,
                controller,
                step_adapter,
                benchmark=skill_module._infer_benchmark(),
                initial_state=runtime_state_snapshot,
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
                runtime_state_snapshot = dict(decision.get("state_snapshot") or runtime_state_snapshot)
                append_observation_message(messages, decision.get("observation", ""))
                total_steps = max(total_steps, int(decision["step_index"]))
                task_done = bool(decision.get("task_done", False))
                task_reward = max(task_reward, float(decision.get("task_reward", 0.0) or 0.0))
                skill_module.record_runtime_recompile(decision)
                recompile_triggered = True
                if progress_callback:
                    progress_callback("runtime_recompile", decision)
                if task_done or not runtime_recompile_available:
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
                try:
                    refreshed_skill_names = skill_module.retrieve_relevant_skills(
                        active_task_prompt,
                        carryover_skill_names=active_skill_names,
                    )
                except TypeError:
                    refreshed_skill_names = skill_module.retrieve_relevant_skills(active_task_prompt)
                merge_skill_names = getattr(
                    skill_module,
                    "merge_runtime_recompile_skill_names",
                    None,
                )
                if callable(merge_skill_names):
                    active_skill_names = merge_skill_names(
                        active_task_prompt,
                        active_skill_names,
                        refreshed_skill_names,
                        decision,
                    )
                else:
                    active_skill_names = refreshed_skill_names or list(active_skill_names)
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
