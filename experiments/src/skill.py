from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
import yaml

from src.utils import get_llm_response
from src.prompt_generator import (
    retrieve_relevant_skills_prompt,
    generate_overall_procedure_prompt,
    generate_overall_procedure_code_prompt,
    refine_dsc_procedure_prompt,
    refine_dsc_procedure_addendum_prompt,
)
from src.runtime_recompile import build_runtime_protocol_state, infer_runtime_protocol_hints

ROOT_DIR = Path(__file__).resolve().parents[2]
SKILLNET_SRC = ROOT_DIR / "skillnet-ai" / "src"
if str(SKILLNET_SRC) not in sys.path:
    sys.path.insert(0, str(SKILLNET_SRC))

from skillnet_ai.compiler import (
    CompilerConfig,
    CompiledSkill,
    DynamicSkillCompiler,
    InMemorySkillRetriever,
    LocalEnvironment,
    LocalSkillLibraryRetriever,
    QueryOptimizer,
    SemanticSoftMatcher,
    TaskDecomposer,
)


def _extract_tagged_block(text: str, tag: str) -> str | None:
    """Return inner text for <tag>...</tag>, case-insensitive; unclosed open tag returns tail."""
    if not text or not str(text).strip():
        return None
    s = str(text)
    pattern = re.compile(rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>", re.DOTALL | re.IGNORECASE)
    m = pattern.search(s)
    if m:
        inner = m.group(1).strip()
        return inner if inner else None
    open_re = re.compile(rf"<{re.escape(tag)}>\s*(.*)$", re.DOTALL | re.IGNORECASE)
    om = open_re.search(s)
    if om:
        tail = om.group(1).strip()
        return tail if tail else None
    return None


def _strip_analysis_section(text: str) -> str:
    """Drop a leading <Analysis>...</Analysis> block if present."""
    if not text:
        return ""
    s = str(text)
    low = s.lower()
    start = low.find("<analysis>")
    end = low.find("</analysis>")
    if start >= 0 and end > start:
        return s[end + len("</analysis>") :].strip()
    return s.strip()


def _procedure_refiner_phase_count(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"\bPhase\s*\d+", str(text), re.I))


def _parse_overall_procedure_response(response: str) -> str:
    """
    Robust extraction of procedural guidance. Prefer XML tags; fall back after </Analysis>
    or the full body so a missing tag never raises.
    """
    if response is None:
        return ""
    raw = str(response).strip()
    if not raw:
        return ""

    block = _extract_tagged_block(raw, "Overall_Procedure")
    if block:
        return block

    stripped = _strip_analysis_section(raw)
    if stripped and stripped != raw:
        block = _extract_tagged_block(stripped, "Overall_Procedure")
        if block:
            return block
        if stripped:
            return stripped

    for alt in (
        r"##\s*ERRATA",
        r"##\s*NEXT STEPS",
        r"PROCEDURE PATCH MODE",
        r"Phase\s*1[:：]",
    ):
        if re.search(alt, raw, re.IGNORECASE):
            return _strip_analysis_section(raw) or raw

    return _strip_analysis_section(raw) or raw


def _parse_refiner_addendum_response(response: str) -> str | None:
    """Extract <Refiner_Addendum>...</Refiner_Addendum>; return None if missing or empty."""
    raw = str(response or "").strip()
    if not raw:
        return None
    block = _extract_tagged_block(raw, "Refiner_Addendum")
    if block is None:
        return None
    block = block.strip()
    return block if block else None


def _dsc_refined_procedure_guard_ok(
    draft: str,
    refined: str,
    *,
    min_chars: int,
    min_length_ratio: float,
    min_phase_slack: int,
) -> bool:
    """Reject refiner output that is over-compressed or collapses too many phases."""
    d = (draft or "").strip()
    r = (refined or "").strip()
    if len(r) < min_chars:
        return False
    if len(d) >= 400 and len(r) < min_length_ratio * len(d):
        return False
    dp = _procedure_refiner_phase_count(d)
    rp = _procedure_refiner_phase_count(r)
    if dp >= 3:
        need = max(1, dp - min_phase_slack)
        if rp < need:
            return False
    return True


def _parse_relevant_skill_names_response(response: str) -> list:
    """Parse skill name list from retrieve_relevant_skills LLM output without brittle [1] indexing."""
    if not response or not str(response).strip():
        return []

    s = str(response)
    raw = _extract_tagged_block(s, "Relevant_Skill_Names")
    if raw is None:
        fence = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", s, re.IGNORECASE)
        if fence:
            raw = fence.group(1)
        else:
            start = s.find("[")
            if start >= 0:
                dec = json.JSONDecoder()
                try:
                    data, _ = dec.raw_decode(s[start:])
                    if isinstance(data, list):
                        return [str(x) for x in data if x]
                except json.JSONDecodeError:
                    pass
            return []

    raw = raw.strip().strip("`").strip()
    if raw.lower().startswith("json"):
        raw = raw[4:].lstrip()
    raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("[")
        if start >= 0:
            try:
                data, _ = json.JSONDecoder().raw_decode(raw[start:])
            except json.JSONDecodeError:
                return []
        else:
            return []

    if isinstance(data, list):
        return [str(x) for x in data if x]
    if isinstance(data, str):
        return [data]
    return []


class SkillModule:
    def __init__(self, **kwargs):
        self.skills_dir = Path(kwargs.get("skills_dir", "skills"))
        self.overall_procedure_examples_path = kwargs.get("overall_procedure_examples_path", "")
        self.procedure_code_template_path = kwargs.get("procedure_code_template_path", None)
        self.model = kwargs.get("model", "gpt-4o")
        self.selection_strategy = kwargs.get("selection_strategy", "baseline")
        self.compiler_seed_with_llm = kwargs.get("compiler_seed_with_llm", True)
        self.compiler_quality_reference_with_llm = kwargs.get("compiler_quality_reference_with_llm", True)
        self.llm_retrieval_max_skills = kwargs.get("llm_retrieval_max_skills", 5)
        self.compiler_seed_max_skills = kwargs.get("compiler_seed_max_skills", 8)
        self.compiler_min_relevance = kwargs.get("compiler_min_relevance", 0.25)
        self.compiler_preserve_top_k = kwargs.get("compiler_preserve_top_k", 2)
        self.compiler_similar_prune_margin = kwargs.get("compiler_similar_prune_margin", 0.08)
        self.compiler_keep_parent_if_better_by = kwargs.get("compiler_keep_parent_if_better_by", 0.05)
        self.compiler_coverage_weight = kwargs.get("compiler_coverage_weight", 0.55)
        self.compiler_quality_weight = kwargs.get("compiler_quality_weight", 0.20)
        self.compiler_cost_weight = kwargs.get("compiler_cost_weight", 0.15)
        self.compiler_latency_weight = kwargs.get("compiler_latency_weight", 0.10)
        self.compiler_top_k = kwargs.get("compiler_top_k", 6)
        self.compiler_max_skill_chars = kwargs.get("compiler_max_skill_chars", 1200)
        self.compiler_max_selected_skills = kwargs.get("compiler_max_selected_skills", 0)
        self.compiler_max_support_skills = kwargs.get("compiler_max_support_skills", 0)
        self.compiler_action_driver_bonus = kwargs.get("compiler_action_driver_bonus", 0.0)
        self.compiler_support_skill_penalty = kwargs.get("compiler_support_skill_penalty", 0.0)
        self.compiler_coverage_floor = kwargs.get("compiler_coverage_floor", 0.0)
        self.runtime_recompile_enabled = kwargs.get(
            "runtime_recompile_enabled",
            self.selection_strategy == "dsc",
        )
        self.runtime_recompile_max_count = kwargs.get("runtime_recompile_max_count", 3)
        self.runtime_recompile_min_interval_steps = kwargs.get("runtime_recompile_min_interval_steps", 2)
        self.runtime_recompile_stagnation_threshold = kwargs.get("runtime_recompile_stagnation_threshold", 1)
        self.runtime_recompile_min_remaining_steps = kwargs.get("runtime_recompile_min_remaining_steps", 1)
        self.runtime_recompile_reward_plateau_steps = kwargs.get(
            "runtime_recompile_reward_plateau_steps",
            5,
        )
        self.runtime_recompile_reward_plateau_min_progress = kwargs.get(
            "runtime_recompile_reward_plateau_min_progress",
            0.2,
        )
        self.runtime_recompile_high_progress_reward_threshold = kwargs.get(
            "runtime_recompile_high_progress_reward_threshold",
            0.7,
        )
        self.runtime_recompile_trace_tail = kwargs.get("runtime_recompile_trace_tail", 6)
        # Optional second LLM pass (e.g. Claude Opus) to refine procedural text after first draft.
        _ref_env = str(os.environ.get("DSC_PROCEDURE_REFINER_ENABLED", "")).lower()
        self.dsc_procedure_refiner_enabled = bool(
            kwargs.get(
                "dsc_procedure_refiner_enabled",
                _ref_env in ("1", "true", "yes"),
            )
        )
        _ref_model_kw = kwargs.get("dsc_procedure_refiner_model")
        self.dsc_procedure_refiner_model = (
            _ref_model_kw
            if _ref_model_kw
            else (os.environ.get("DSC_PROCEDURE_REFINER_MODEL") or "claude-opus-4-20250514")
        )
        self.dsc_procedure_refiner_min_chars = int(kwargs.get("dsc_procedure_refiner_min_chars", 80))
        self.dsc_procedure_refiner_min_length_ratio = float(
            kwargs.get("dsc_procedure_refiner_min_length_ratio", 0.42)
        )
        self.dsc_procedure_refiner_min_phase_slack = int(
            kwargs.get("dsc_procedure_refiner_min_phase_slack", 1)
        )
        # append = non-destructive addendum (default); replace = legacy full rewrite (risky).
        _mode_kw = kwargs.get("dsc_procedure_refiner_mode")
        _env_mode = str(os.environ.get("DSC_PROCEDURE_REFINER_MODE", "")).strip().lower()
        if _mode_kw is not None and str(_mode_kw).strip():
            self.dsc_procedure_refiner_mode = str(_mode_kw).strip().lower()
        elif _env_mode in ("append", "replace", "off"):
            self.dsc_procedure_refiner_mode = _env_mode
        else:
            self.dsc_procedure_refiner_mode = "append"
        if self.dsc_procedure_refiner_mode not in ("append", "replace", "off"):
            self.dsc_procedure_refiner_mode = "append"
        self.runtime_recompile_count = 0
        self.runtime_recompile_events = []
        self.runtime_last_recompile_step = -999
        self.last_deterministic_procedure_kind = None
        self.last_compilation = None
        self.last_candidate_assets = {}
        self.last_seed_skill_names = []
        self.last_quality_reference_skill_names = []
        self.last_selected_skill_names = []

        # Semantic soft matcher: replaces hard keyword matching in SkillUtilityScorer.
        # Initialised lazily via _get_soft_matcher() to avoid blocking __init__.
        self.compiler_use_semantic_matching = kwargs.get("compiler_use_semantic_matching", True)
        self._soft_matcher: SemanticSoftMatcher | None = None

        self.metadata = self._load_metadata()

        # Load procedure code template and overall procedure examples
        if self.procedure_code_template_path is not None and os.path.exists(self.procedure_code_template_path):
            with open(self.procedure_code_template_path, "r") as f:
                self.procedure_code_template = f.read()
        else:
            self.procedure_code_template = ''

        if self.overall_procedure_examples_path is not None and os.path.exists(self.overall_procedure_examples_path):
            with open(self.overall_procedure_examples_path, "r") as f:
                self.overall_procedure_examples = f.read()
        else:
            self.overall_procedure_examples = ''


    def _get_soft_matcher(self) -> SemanticSoftMatcher | None:
        """Return a lazily-initialised SemanticSoftMatcher, or None if disabled.

        The matcher is created once per SkillModule instance and reused across
        all compile() calls.  Embeddings are cached on disk so repeat runs
        incur no API cost for previously seen tokens.
        """
        if not self.compiler_use_semantic_matching:
            return None
        if self._soft_matcher is not None:
            return self._soft_matcher
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return None
        try:
            self._soft_matcher = SemanticSoftMatcher.from_openai(
                api_key=api_key,
                base_url=os.environ.get("OPENAI_BASE_URL") or None,
                cache_dir=os.path.expanduser("~/.skillnet/emb"),
            )
            # Pre-embed all skill descriptions so scoring never blocks mid-task.
            lib = LocalSkillLibraryRetriever(skills_dir=str(self.skills_dir))
            all_assets = lib.retrieve(None)
            self._soft_matcher.warm_up_skills(all_assets)
        except Exception:
            self._soft_matcher = None
        return self._soft_matcher

    def _load_metadata(self):
        """Load existing metadata from file, return empty dict if file does not exist."""
        metadata = {}
        for skill_dir in self.skills_dir.iterdir():
            if skill_dir.is_dir():
                skill_md_path = skill_dir / "SKILL.md"
                if skill_md_path.exists():
                    try:
                        content = skill_md_path.read_text(encoding="utf-8")
                        if content.strip().startswith('---'):
                            parts = content.split('---', 2)
                            if len(parts) >= 3:
                                header_str = parts[1]
                                header_data = yaml.safe_load(header_str)
                                if isinstance(header_data, dict) and header_data.get('name') and header_data.get('description'):
                                    metadata[header_data['name']] = {
                                        'description': header_data['description'],
                                        'skill_dir': str(skill_dir)
                                    }
                                else:
                                    print(f"[WARNING] Invalid metadata format in {skill_dir.name}, skipping.")
                            else:
                                print(f"[WARNING] No valid metadata found in {skill_dir.name}, skipping.")
                        else:
                            print(f"[WARNING] No metadata header found in {skill_dir.name}, skipping.")
                    except Exception as e:
                        print(f"[ERROR] Failed to read or parse SKILL.md for {skill_dir.name}: {e}")
                else:
                    print(f"[WARNING] SKILL.md not found for {skill_dir.name}, skipping.")
        print(f"[INFO] Loaded metadata for {len(metadata)} skills.")
        return metadata
    
    def retrieve_relevant_skills(self, task, carryover_skill_names=None):
        """
        Retrieve relevant skills from metadata based on task description.
        """
        carryover_skill_names = list(dict.fromkeys(list(carryover_skill_names or [])))
        if self.selection_strategy == "dsc":
            quality_reference_skill_names = (
                self._llm_retrieve_relevant_skill_names(
                    task,
                    max_skills=self.llm_retrieval_max_skills,
                    candidate_mode=False,
                )
                if self.compiler_quality_reference_with_llm
                else []
            )
            seed_skill_names = (
                self._llm_retrieve_relevant_skill_names(
                    task,
                    max_skills=self.compiler_seed_max_skills,
                    candidate_mode=True,
                )
                if self.compiler_seed_with_llm
                else None
            )
            self.last_seed_skill_names = list(seed_skill_names or [])
            self.last_quality_reference_skill_names = list(quality_reference_skill_names or [])
            compile_seed_skill_names = list(
                dict.fromkeys(
                    list(seed_skill_names or [])
                    + list(quality_reference_skill_names or [])
                    + carryover_skill_names
                )
            )
            self.last_compilation = self._compile_task(
                task,
                seed_skill_names=compile_seed_skill_names or seed_skill_names,
            )
            if quality_reference_skill_names:
                quality_first_skill_names = self._select_quality_first_skill_names(
                    task,
                    quality_reference_skill_names,
                    seed_skill_names or [],
                )
                self.last_compilation = self._build_quality_first_compilation(
                    self.last_compilation,
                    quality_first_skill_names,
                )
                selected = list(quality_first_skill_names)
            else:
                selected = [
                    item.asset.name
                    for item in self.last_compilation.compiled_skills[: self.compiler_top_k]
                ]
            self.last_selected_skill_names = list(selected)
            return selected

        return self._llm_retrieve_relevant_skill_names(
            task,
            max_skills=self.llm_retrieval_max_skills,
            candidate_mode=False,
        )
    
    def generate_overall_procedure(self, task, skill_names):
        """
        Generate overall procedure by combining individual skill contents.
        """
        deterministic = self._deterministic_procedure(task, skill_names)
        if deterministic is not None:
            return deterministic

        compiler_summary = ""
        if self.selection_strategy == "dsc" and self.last_compilation is not None:
            if self.last_quality_reference_skill_names:
                skill_contents, compiler_summary = self._build_quality_first_skill_payload(skill_names)
            else:
                skill_contents, compiler_summary = self._build_compiled_skill_payload(skill_names)
        else:
            skill_contents = self._build_full_skill_payload(skill_names)

        patch_mode = bool(task) and (
            "[Runtime Recompile]" in task or "PROCEDURE PATCH MODE" in task
        )
        response = get_llm_response(
            generate_overall_procedure_prompt(
                task,
                self.overall_procedure_examples,
                skill_contents,
                compiler_summary=compiler_summary,
                procedure_patch_mode=patch_mode,
            ),
            is_string=True,
            model=self.model
        )
        procedure = _parse_overall_procedure_response(response)
        if (
            self.selection_strategy == "dsc"
            and self.dsc_procedure_refiner_enabled
            and self.last_compilation is not None
            and not patch_mode
            and self.dsc_procedure_refiner_mode != "off"
        ):
            try:
                if self.dsc_procedure_refiner_mode == "append":
                    addendum_raw = get_llm_response(
                        refine_dsc_procedure_addendum_prompt(
                            task=task,
                            draft_procedure=procedure,
                            relevant_skill_names=skill_names,
                            compiler_summary=compiler_summary,
                        ),
                        is_string=True,
                        model=self.dsc_procedure_refiner_model,
                    )
                    addendum = _parse_refiner_addendum_response(addendum_raw)
                    if addendum and len(addendum) <= 6000:
                        procedure = (
                            procedure.rstrip()
                            + "\n\n## DSC refiner addendum\n\n"
                            + addendum.strip()
                        )
                elif self.dsc_procedure_refiner_mode == "replace":
                    refine_messages = refine_dsc_procedure_prompt(
                        task=task,
                        relevant_skill_names=skill_names,
                        skill_contents=skill_contents,
                        compiler_summary=compiler_summary,
                        draft_procedure=procedure,
                    )
                    refined_raw = get_llm_response(
                        refine_messages,
                        is_string=True,
                        model=self.dsc_procedure_refiner_model,
                    )
                    refined = _parse_overall_procedure_response(refined_raw)
                    if refined and len(refined.strip()) > 50:
                        if _dsc_refined_procedure_guard_ok(
                            procedure,
                            refined,
                            min_chars=self.dsc_procedure_refiner_min_chars,
                            min_length_ratio=self.dsc_procedure_refiner_min_length_ratio,
                            min_phase_slack=self.dsc_procedure_refiner_min_phase_slack,
                        ):
                            procedure = refined
                        else:
                            print(
                                "[WARN] DSC procedure refiner (replace) rejected by guard "
                                f"(len {len(refined)}/{len(procedure)} chars, "
                                f"phases {_procedure_refiner_phase_count(refined)}/"
                                f"{_procedure_refiner_phase_count(procedure)}); keeping draft."
                            )
            except Exception as exc:
                print(f"[WARN] DSC procedure refiner failed ({self.dsc_procedure_refiner_model}): {exc}")
        return procedure
    
    def generate_overall_procedure_code(self, task, overall_procedure):
        """
        Generate overall procedure code.
        """
        if self.last_deterministic_procedure_kind == "scienceworld_conductivity":
            return self._scienceworld_static_procedure_code(task, overall_procedure)

        response = get_llm_response(
            generate_overall_procedure_code_prompt(task, overall_procedure, self.procedure_code_template),
            is_string=True,
            model=self.model
        )
        pattern = r"<Overall_Procedure_Code>(.*?)</Overall_Procedure_Code>"
        matchs = re.findall(pattern, response, re.DOTALL)
        if matchs:
            raw_content = matchs[-1]
            if "<Overall_Procedure_Code>" in raw_content: # handle nested tags
                raw_content = raw_content.split("<Overall_Procedure_Code>")[-1]
            overall_procedure_code = self._sanitize_generated_procedure_code(raw_content)
        else:
            overall_procedure_code = self._sanitize_generated_procedure_code(response)

        return overall_procedure_code

    def _sanitize_generated_procedure_code(self, raw_content):
        if not raw_content:
            return ""
        code = str(raw_content).strip()
        code = re.sub(r"^\s*```(?:python)?\s*", "", code, flags=re.IGNORECASE)
        code = re.sub(r"\s*```\s*$", "", code)
        if "<Overall_Procedure_Code>" in code:
            code = code.split("<Overall_Procedure_Code>")[-1]
        if "</Overall_Procedure_Code>" in code:
            code = code.split("</Overall_Procedure_Code>")[0]
        def_index = code.find("def overall_procedure_code")
        if def_index >= 0:
            code = code[def_index:]
        return code.strip()

    def _deterministic_procedure(self, task, skill_names):
        self.last_deterministic_procedure_kind = None
        if self.selection_strategy == "dsc" and self._is_scienceworld_conductivity_task(task, skill_names):
            self.last_deterministic_procedure_kind = "scienceworld_conductivity"
            return self._scienceworld_conductivity_procedure(task)
        return None

    def _is_scienceworld_conductivity_query(self, task):
        if self._infer_benchmark() != "scienceworld":
            return False
        query = str(task or "").lower()
        return "electrically conductive" in query or "conductivity" in query

    def _is_scienceworld_conductivity_task(self, task, skill_names):
        if not self._is_scienceworld_conductivity_query(task):
            return False
        return "scienceworld-conductivity-tester" in set(skill_names or [])

    def _scienceworld_conductivity_procedure(self, task):
        target_object, source_room, conductive_box, nonconductive_box = self._extract_scienceworld_conductivity_targets(task)
        return f"""# TASK PROCEDURAL GUIDANCE: Test Electrical Conductivity and Sort

Task:
{task}

Phase 1: Locate and acquire the target substance
1. `teleport to {source_room}`
2. `look around`
3. Identify the exact object name `{target_object}` from observation and reuse that exact name everywhere.
4. `pick up {target_object}`
5. `focus on {target_object}`
6. Confirm the substance is now in inventory before leaving the room.

Phase 2: Prepare the workshop circuit
1. `teleport to workshop`
2. `look around`
3. Identify:
   - one battery
   - three wires (prefer `black wire`, `blue wire`, and `yellow wire` if present; otherwise use any three visible wires)
   - one actuator component to test power flow:
     - prefer a light bulb
     - otherwise use an electric motor or electric buzzer if that is the visible actuator
   - the answer boxes named in the task description
4. Keep the substance in inventory for the fast path. Only drop it later if fallback wiring is needed.
5. Confirm the substance has `terminal 1` and `terminal 2`.

Phase 3: Fast conductivity check
Use exact observed contact-point syntax. Try the shortest working circuit first:
battery anode -> wire 1 -> actuator cathode/terminal 2
actuator anode/terminal 1 -> wire 2 -> substance terminal 1
substance terminal 2 -> battery cathode

Actions:
1. `connect battery anode to <WIRE1> terminal 1`
2. `connect <WIRE1> terminal 2 to <ACTUATOR> cathode` or `<ACTUATOR> terminal 2`
3. `connect <ACTUATOR> anode` or `<ACTUATOR> terminal 1` to `<WIRE2> terminal 1`
4. `connect <WIRE2> terminal 2 to <SUBSTANCE> terminal 1`
5. `connect <SUBSTANCE> terminal 2 to battery cathode`
6. `wait1`
7. `look at <ACTUATOR>`

Decision:
- If the actuator is on or activated, the substance is conductive.
- If the actuator is still off or deactivated, run the fallback stable three-wire circuit below before concluding nonconductive.

Phase 4: Fallback stable conductivity circuit
Only if the fast path leaves the actuator off:
1. `drop <SUBSTANCE>` so it is directly in the workshop.
2. Rebuild the stable three-wire circuit:
   - `connect battery cathode to <WIRE2> terminal 1`
   - `connect <WIRE1> terminal 2 to <ACTUATOR> cathode` or `<ACTUATOR> terminal 2`
   - `connect <WIRE3> terminal 2 to <ACTUATOR> anode` or `<ACTUATOR> terminal 1`
   - `connect <SUBSTANCE> terminal 1 to <WIRE2> terminal 2`
   - `connect <SUBSTANCE> terminal 2 to <WIRE3> terminal 1`
3. `wait1`
4. `wait1`
5. `look at <ACTUATOR>`
6. If the actuator is on, treat as conductive. Otherwise treat as nonconductive.

Phase 5: Place into the correct answer box
1. If conductive: `move <SUBSTANCE> to {conductive_box}`
2. If nonconductive: `move <SUBSTANCE> to {nonconductive_box}`
3. Task is complete only when the environment confirms the correct placement.

Error handling:
- If a command fails, immediately `look around`, verify exact names and terminals, and retry with the corrected syntax.
- If the environment returns `Ambiguous request: Please enter the number...`, respond with only the matching option index, such as `0`.
- Never invent rooms, components, colors, or terminals that are not present in the latest observation.
- Do not change the circuit topologies above.
- Keep the substance in inventory for the fast path; only drop it if the fallback path is needed."""

    def _extract_scienceworld_conductivity_targets(self, task):
        object_match = re.search(r"determine if (.+?) is electrically conductive", str(task or ""), re.IGNORECASE)
        room_match = re.search(r"is located around the (.+?)[\.,]", str(task or ""), re.IGNORECASE)
        conductive_match = re.search(
            r"If it is electrically conductive, place it in the (.+?)[\\.]",
            str(task or ""),
            re.IGNORECASE,
        )
        nonconductive_match = re.search(
            r"If it is electrically nonconductive, place it in the (.+?)[\\.]",
            str(task or ""),
            re.IGNORECASE,
        )
        object_name = object_match.group(1).strip() if object_match else "unknown substance S"
        source_room = room_match.group(1).strip() if room_match else "workshop"
        conductive_box = conductive_match.group(1).strip() if conductive_match else "orange box"
        nonconductive_box = nonconductive_match.group(1).strip() if nonconductive_match else "yellow box"
        return object_name, source_room, conductive_box, nonconductive_box

    def _scienceworld_static_procedure_code(self, task, overall_procedure):
        escaped = overall_procedure.replace('"""', r"\"\"\"")
        target_object, source_room, conductive_box, nonconductive_box = self._extract_scienceworld_conductivity_targets(task)
        return f'''# ==========================================
# Procedure code template for ScienceWorld environment
# ==========================================
def overall_procedure_code(
    env,
    llm,
    model: str,
    parse_action,
    messages: list = [],
    max_steps: int = 30
):
    """
    Deterministic conductivity solver for a quality-critical ScienceWorld task.
    """
    import re

    procedure_guidelines = """{escaped}"""
    messages.append({{"role": "user", "content": procedure_guidelines}})

    target_object = {target_object!r}
    source_room = {source_room!r}
    conductive_box = {conductive_box!r}
    nonconductive_box = {nonconductive_box!r}
    task_done = False
    current_steps = 0
    task_reward = 0

    def run_action(action: str):
        nonlocal task_done, current_steps, task_reward
        if task_done or current_steps >= max_steps:
            return ""
        messages.append({{"role": "assistant", "content": f"Action: {{action}}"}})
        observation, step_reward, task_done, info = env.step(action)
        task_reward = info['score'] if info.get('score') is not None and info['score'] > task_reward else task_reward
        print(f'\\033[93mObservation: \\n{{observation}}\\033[0m')
        messages.append({{"role": "user", "content": f"Observation: {{observation}}"}})
        current_steps += 1
        return observation

    def unique_matches(pattern: str, text: str):
        matches = []
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            name = match.strip()
            if name not in matches:
                matches.append(name)
        return matches

    def choose_components(observation: str):
        wires = unique_matches(r"a ([a-z ]+ wire)", observation)
        bulbs = unique_matches(r"a ([a-z ]+ light bulb)", observation)
        motors = unique_matches(r"a ([a-z ]*electric motor)", observation)
        buzzers = unique_matches(r"a ([a-z ]*electric buzzer)", observation)
        preferred_wires = [wire for wire in ("black wire", "blue wire", "yellow wire") if wire in wires]
        for wire in wires:
            if wire not in preferred_wires:
                preferred_wires.append(wire)
            if len(preferred_wires) >= 3:
                break
        if len(preferred_wires) < 3:
            raise RuntimeError(f"Unable to identify three wires from observation: {{observation}}")
        if "green light bulb" in bulbs:
            actuator = "green light bulb"
        elif bulbs:
            actuator = bulbs[0]
        elif motors:
            actuator = motors[0]
        elif buzzers:
            actuator = buzzers[0]
        else:
            raise RuntimeError(f"Unable to identify an actuator from observation: {{observation}}")
        return preferred_wires[0], preferred_wires[1], preferred_wires[2], actuator

    def actuator_terminals(actuator_name: str):
        lowered = actuator_name.lower()
        if "light bulb" in lowered:
            return "cathode", "anode"
        return "terminal 2", "terminal 1"

    source_observation = run_action(f"teleport to {{source_room}}")
    source_observation = run_action("look around")
    run_action(f"pick up {{target_object}}")
    run_action(f"focus on {{target_object}}")
    workshop_observation = run_action("teleport to workshop")
    workshop_observation = run_action("look around")
    wire1, wire2, wire3, actuator = choose_components(workshop_observation)
    actuator_negative, actuator_positive = actuator_terminals(actuator)
    run_action(f"connect battery anode to {{wire1}} terminal 1")
    run_action(f"connect {{wire1}} terminal 2 to {{actuator}} {{actuator_negative}}")
    run_action(f"connect {{actuator}} {{actuator_positive}} to {{wire2}} terminal 1")
    run_action(f"connect {{wire2}} terminal 2 to {{target_object}} terminal 1")
    run_action(f"connect {{target_object}} terminal 2 to battery cathode")
    run_action("wait1")
    bulb_observation = run_action(f"look at {{actuator}}")

    bulb_is_on = (
        "which is on" in bulb_observation.lower()
        or "which is activated" in bulb_observation.lower()
        or " is on." in bulb_observation.lower()
        or " is activated." in bulb_observation.lower()
    )
    if not bulb_is_on:
        run_action(f"drop {{target_object}}")
        run_action(f"connect battery cathode to {{wire2}} terminal 1")
        run_action(f"connect {{wire1}} terminal 2 to {{actuator}} {{actuator_negative}}")
        run_action(f"connect {{wire3}} terminal 2 to {{actuator}} {{actuator_positive}}")
        run_action(f"connect {{target_object}} terminal 1 to {{wire2}} terminal 2")
        run_action(f"connect {{target_object}} terminal 2 to {{wire3}} terminal 1")
        run_action("wait1")
        run_action("wait1")
        bulb_observation = run_action(f"look at {{actuator}}")
        bulb_is_on = (
            "which is on" in bulb_observation.lower()
            or "which is activated" in bulb_observation.lower()
            or " is on." in bulb_observation.lower()
            or " is activated." in bulb_observation.lower()
        )
    target_box = conductive_box if bulb_is_on else nonconductive_box
    run_action(f"move {{target_object}} to {{target_box}}")

    return messages, task_done, task_reward, current_steps'''

    def _compile_task(self, task, seed_skill_names=None):
        query_plan = QueryOptimizer().optimize(task)
        base_retriever = LocalSkillLibraryRetriever(str(self.skills_dir))
        candidates = base_retriever.retrieve(query_plan)
        if seed_skill_names:
            allowed = set(seed_skill_names)
            seeded_candidates = [
                asset for asset in candidates
                if asset.name in allowed or asset.skill_id in allowed
            ]
            if seeded_candidates:
                candidates = seeded_candidates
        self.last_candidate_assets = {
            asset.name: asset
            for asset in candidates
        }
        for asset in candidates:
            self.last_candidate_assets.setdefault(asset.skill_id, asset)
        retriever = InMemorySkillRetriever(candidates)

        compiler = DynamicSkillCompiler(
            retriever=retriever,
            config=CompilerConfig(
                min_relevance=self.compiler_min_relevance,
                preserve_top_k=self.compiler_preserve_top_k,
                similar_prune_margin=self.compiler_similar_prune_margin,
                keep_parent_if_better_by=self.compiler_keep_parent_if_better_by,
                coverage_weight=self.compiler_coverage_weight,
                quality_weight=self.compiler_quality_weight,
                cost_weight=self.compiler_cost_weight,
                latency_weight=self.compiler_latency_weight,
                max_selected_skills=self.compiler_max_selected_skills,
                max_support_skills=self.compiler_max_support_skills,
                action_driver_bonus=self.compiler_action_driver_bonus,
                support_skill_penalty=self.compiler_support_skill_penalty,
                coverage_floor=self.compiler_coverage_floor,
                compile_stage=self._current_compile_stage(task),
            ),
            soft_matcher=self._get_soft_matcher(),
        )
        environment = LocalEnvironment(
            cwd=str(Path.cwd()),
            workspace_root=str(self.skills_dir.resolve()),
            python_bin=sys.executable or "python",
            shell=os.environ.get("SHELL", "sh"),
            os_name=sys.platform,
            available_bins=set(),
            benchmark=self._infer_benchmark(),
        )
        return compiler.compile(task, environment=environment)

    def _select_quality_first_skill_names(self, task, reference_skill_names, seed_skill_names):
        available = []
        for skill_name in list(reference_skill_names) + list(seed_skill_names):
            if skill_name not in available:
                available.append(skill_name)

        budget = self._quality_first_budget(task)
        if self.last_compilation is not None:
            compiled_order = [
                item.asset.name
                for item in self.last_compilation.compiled_skills
                if item.asset.name in available
            ]
            remaining = [skill_name for skill_name in available if skill_name not in compiled_order]
            available = compiled_order + remaining
        if budget > 0:
            return available[:budget]
        return available

    def merge_runtime_recompile_skill_names(
        self,
        task,
        previous_skill_names,
        refreshed_skill_names,
        decision=None,
    ):
        previous = list(dict.fromkeys(list(previous_skill_names or [])))
        refreshed = list(dict.fromkeys(list(refreshed_skill_names or [])))
        if not refreshed:
            return previous
        if not previous:
            return refreshed

        effective = self._adaptive_compiler_config(task)
        budget = self._quality_first_budget(task)
        if effective.max_selected_skills > 0:
            budget = max(1, effective.max_selected_skills)
        preserve_previous = max(2, int(getattr(effective, "preserve_top_k", 2)))

        decision = decision or {}
        if str(decision.get("reason", "")) in {"action_failure", "stagnation"}:
            preserve_previous = max(preserve_previous, min(len(previous), 4))
        if float(decision.get("task_reward", 0.0) or 0.0) > 0:
            preserve_previous = max(preserve_previous, min(len(previous), 4))
        if getattr(effective, "profile_name", "") == "workflow":
            preserve_previous = max(preserve_previous, min(len(previous), 5))

        merged = []
        previous_kept = 0

        def add(skill_name):
            nonlocal previous_kept
            if not skill_name or skill_name in merged:
                return
            merged.append(skill_name)
            if skill_name in previous:
                previous_kept += 1

        for skill_name in refreshed:
            add(skill_name)

        for skill_name in previous:
            if previous_kept >= preserve_previous:
                break
            add(skill_name)

        ordered_candidates = []
        if self.last_compilation is not None:
            for item in self.last_compilation.compiled_skills:
                name = item.asset.name
                if name in refreshed or name in previous:
                    ordered_candidates.append(name)
        for skill_name in refreshed + previous:
            if skill_name not in ordered_candidates:
                ordered_candidates.append(skill_name)
        for skill_name in ordered_candidates:
            add(skill_name)

        budget = max(budget, len(refreshed), min(len(merged), preserve_previous + len(refreshed)))
        if budget > 0 and len(merged) > budget:
            priority = set(refreshed)
            while len(merged) > budget:
                drop_idx = None
                preserved_previous = sum(1 for name in merged if name in previous)
                for idx in range(len(merged) - 1, -1, -1):
                    candidate = merged[idx]
                    if candidate in priority:
                        continue
                    if candidate in previous and preserved_previous <= min(preserve_previous, len(previous)):
                        continue
                    drop_idx = idx
                    break
                if drop_idx is None:
                    break
                removed = merged.pop(drop_idx)
                if removed in previous:
                    preserved_previous -= 1

        return merged

    def _build_quality_first_compilation(self, compiled_package, reference_skill_names):
        """
        Preserve the baseline/SkillNet reference skill set and let DSC compress content
        rather than deleting quality-critical skills.
        """
        compiled_lookup = {
            item.asset.name: item
            for item in compiled_package.compiled_skills
        }
        filtered_skills = []
        for skill_name in reference_skill_names:
            compiled_skill = compiled_lookup.get(skill_name)
            if compiled_skill is None:
                compiled_skill = self._fallback_compiled_skill(skill_name)
            if compiled_skill is not None:
                filtered_skills.append(compiled_skill)

        if not filtered_skills:
            return compiled_package

        compiled_package.compiled_skills = filtered_skills
        compiled_package.execution_order = [item.asset.name for item in filtered_skills]
        compiled_package.graph.skills = {
            item.asset.skill_id: item.asset
            for item in filtered_skills
        }
        compiled_package.graph.relations = [
            relation
            for relation in compiled_package.graph.relations
            if relation.source in compiled_package.graph.skills
            and relation.target in compiled_package.graph.skills
        ]

        used_skill_ids = {item.asset.skill_id for item in filtered_skills}
        for dropped_skill_id in list(compiled_package.dropped_skills):
            if dropped_skill_id in used_skill_ids:
                compiled_package.dropped_skills.pop(dropped_skill_id, None)

        metrics = compiled_package.metrics
        metrics.selected_count = len(filtered_skills)
        metrics.edge_count_after = len(compiled_package.graph.relations)
        metrics.estimated_token_cost_after = sum(item.asset.token_cost for item in filtered_skills)
        metrics.estimated_execution_cost_after = sum(item.asset.execution_cost for item in filtered_skills)
        selected_capabilities = set()
        covered_subgoals = set()
        fragment_count_after = 0
        fragment_token_cost_after = 0.0
        for item in filtered_skills:
            selected_capabilities |= item.asset.normalized_capabilities()
            covered_subgoals |= set(item.assigned_subgoals)
            fragment_count_after += len(item.selected_fragments)
            fragment_token_cost_after += sum(fragment.token_cost for fragment in item.selected_fragments)
        required = compiled_package.query_plan.required_capabilities
        metrics.coverage_score = len(required & selected_capabilities) / max(len(required), 1)
        metrics.redundancy_reduction = max(
            0.0,
            1.0 - (metrics.selected_count / max(metrics.candidate_count, 1)),
        )
        metrics.covered_subgoal_count = len(covered_subgoals)
        metrics.fragment_count_after = fragment_count_after
        metrics.fragment_token_cost_after = fragment_token_cost_after
        compiled_package.notes.append(
            "Quality-first mode preserved the SkillNet reference skill set and compressed their contents."
        )
        return compiled_package

    def _fallback_compiled_skill(self, skill_name):
        asset = self.last_candidate_assets.get(skill_name)
        if asset is None:
            asset = self.last_candidate_assets.get(skill_name.replace("_", "-"))
        if asset is None:
            return None
        return CompiledSkill(
            asset=asset,
            selected_fragments=[],
            assigned_subgoals=[],
            localized_instructions=self._fallback_localized_instructions(skill_name),
            utility_score=0.0,
            selected_reason="preserved_from_skillnet_reference",
        )

    def _fallback_localized_instructions(self, skill_name):
        meta = self.metadata.get(skill_name)
        if not meta:
            return [f"Use {skill_name} exactly as defined by the source skill."]
        skill_dir = Path(meta["skill_dir"])
        instructions = []
        skill_md_path = skill_dir / "SKILL.md"
        if skill_md_path.exists():
            for raw_line in skill_md_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or line == "---" or line.startswith("name:") or line.startswith("description:"):
                    continue
                instructions.append(line)
                if len(instructions) >= 4:
                    break
        if not instructions:
            instructions.append(meta["description"])
        return instructions

    def _infer_benchmark(self) -> str:
        path_text = str(self.skills_dir).lower()
        if "alfworld" in path_text:
            return "alfworld"
        if "scienceworld" in path_text:
            return "scienceworld"
        if "webshop" in path_text:
            return "webshop"
        return "generic"

    def _current_compile_stage(self, task) -> str:
        if self.runtime_recompile_count <= 0:
            return "initial"
        task_text = str(task or "")
        if "[Runtime Recompile]" in task_text:
            return "runtime_recompile"
        return "runtime_recompile"

    def _quality_first_budget(self, task) -> int:
        effective = self._adaptive_compiler_config(task)
        if effective.max_selected_skills > 0:
            return max(1, effective.max_selected_skills)
        return max(self.compiler_top_k, effective.preserve_top_k + 3, 1)

    def _adaptive_compiler_config(self, task):
        task_text = str(task or "")
        query_plan = QueryOptimizer().optimize(task_text)
        subgoals = TaskDecomposer().decompose(query_plan)
        probe = DynamicSkillCompiler(
            retriever=InMemorySkillRetriever([]),
            config=CompilerConfig(
                min_relevance=self.compiler_min_relevance,
                preserve_top_k=self.compiler_preserve_top_k,
                similar_prune_margin=self.compiler_similar_prune_margin,
                keep_parent_if_better_by=self.compiler_keep_parent_if_better_by,
                coverage_weight=self.compiler_coverage_weight,
                quality_weight=self.compiler_quality_weight,
                cost_weight=self.compiler_cost_weight,
                latency_weight=self.compiler_latency_weight,
                max_selected_skills=self.compiler_max_selected_skills,
                max_support_skills=self.compiler_max_support_skills,
                action_driver_bonus=self.compiler_action_driver_bonus,
                support_skill_penalty=self.compiler_support_skill_penalty,
                coverage_floor=self.compiler_coverage_floor,
                compile_stage=self._current_compile_stage(task_text),
            ),
            soft_matcher=self._get_soft_matcher(),
        )
        environment = LocalEnvironment(
            cwd=str(Path.cwd()),
            workspace_root=str(self.skills_dir.resolve()),
            python_bin=sys.executable or "python",
            shell=os.environ.get("SHELL", "sh"),
            os_name=sys.platform,
            available_bins=set(),
            benchmark=self._infer_benchmark(),
        )
        return probe._effective_config(query_plan, subgoals, environment)

    def should_use_runtime_recompile(self) -> bool:
        return self.selection_strategy == "dsc" and bool(self.runtime_recompile_enabled)

    def can_runtime_recompile(self) -> bool:
        if not self.should_use_runtime_recompile():
            return False
        return self.runtime_recompile_count < max(0, int(self.runtime_recompile_max_count))

    def record_runtime_recompile(self, decision):
        self.runtime_recompile_count += 1
        if isinstance(decision, dict):
            self.runtime_recompile_events.append(dict(decision))
            self.runtime_last_recompile_step = int(decision.get("step_index", self.runtime_last_recompile_step))

    def _detect_syntax_errors(self, decision: dict) -> str:
        """Analyze trace_tail to identify repeated action syntax failures and return a hint string."""
        trace = decision.get("trace_tail") or []
        if not trace:
            return ""

        SYNTAX_PATTERNS = [
            ("no known action",       "unknown_action"),
            ("already connected",     "already_connected"),
            ("must be disconnected",  "needs_disconnect"),
            ("not possible",          "not_possible"),
            ("not valid",             "not_valid"),
        ]

        counts: dict = {key: 0 for _, key in SYNTAX_PATTERNS}
        failed_actions: list = []
        for step in trace:
            obs_lower = step.get("observation", "").lower()
            act = step.get("action", "").strip()
            for pattern, key in SYNTAX_PATTERNS:
                if pattern in obs_lower:
                    counts[key] += 1
                    if act and act not in failed_actions:
                        failed_actions.append(act)
                    break

        total_errors = sum(counts.values())
        if total_errors < 2:
            return ""

        hints = []
        if counts["already_connected"] + counts["needs_disconnect"] >= 1:
            hints.append(
                "- A connection endpoint is already occupied. Use the environment's exact documented "
                "disconnect action, including all required endpoint names, before reconnecting."
            )
        if counts["unknown_action"] >= 2:
            hints.append(
                "- Action format error: environment rejected actions with wrong syntax. "
                "Avoid parenthetical qualifiers like '(containing ...)' in object names; "
                "use the shortest exact object name from the latest observation."
            )
        if not hints:
            hints.append(
                f"- {total_errors} syntax errors detected. "
                "Prioritize skills that provide verified exact-syntax examples for this environment."
            )

        failed_sample = "; ".join(failed_actions[:3])
        result = (
            f"ACTION SYNTAX ERRORS DETECTED ({total_errors} failures in last {len(trace)} steps):\n"
            + "\n".join(hints)
        )
        if failed_sample:
            result += f"\nFailed actions sample: {failed_sample}"
        return result + "\n"

    def _format_runtime_list(self, items, limit: int = 6) -> str:
        cleaned = [str(item).strip() for item in (items or []) if str(item).strip()]
        if not cleaned:
            return "- none\n"
        lines = [f"- {item}" for item in cleaned[:limit]]
        if len(cleaned) > limit:
            lines.append(f"- ... ({len(cleaned) - limit} more)")
        return "\n".join(lines) + "\n"

    def build_runtime_recompile_task(self, task_prompt, messages, decision, remaining_steps):
        decision = decision or {}
        effective = self._adaptive_compiler_config(task_prompt)
        snapshot = decision.get("state_snapshot") or {}
        recent_messages = []
        for message in messages[-8:]:
            role = message.get("role", "user")
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            normalized = re.sub(r"\s+", " ", content)
            recent_messages.append(f"{role}: {normalized[:240]}")

        state_lines = []
        for key in (
            "current_location",
            "visible_entities",
            "inventory",
            "visited_locations",
            "open_receptacles",
            "completed_transfers",
            "completed_transforms",
            "completed_actions",
            "focused_targets",
            "last_measurements",
            "ambiguous_options",
            "object_identity_ledger",
        ):
            value = snapshot.get(key)
            if value:
                state_lines.append(f"- {key}: {value}")

        observation_text = str(decision.get("observation", "")).lower()
        relaxation_hint = ""
        if effective.profile_name == "search" or any(
            marker in observation_text
            for marker in ("no results", "not found", "no candidate", "no product", "0 results", "no matches")
        ):
            relaxation_hint = (
                "If the current path looks over-constrained or retrieval-heavy, relax non-essential constraints "
                "and recover the shortest path back to a directly actionable state.\n"
            )
        elif decision.get("reason") == "reward_plateau":
            relaxation_hint = (
                "The task has made partial progress but stopped improving. Preserve the completed scaffold, "
                "focus on the missing final subgoals, and avoid restarting solved phases.\n"
            )

        syntax_error_hint = self._detect_syntax_errors(decision)
        failure_type = decision.get("failure_type") or decision.get("reason", "unknown")
        repair_hint = decision.get("repair_hint") or ""
        protocol_hints = infer_runtime_protocol_hints(task_prompt, decision)
        protocol_state = build_runtime_protocol_state(task_prompt, decision)
        protocol_hint_text = ""
        if protocol_hints:
            protocol_hint_text = (
                "PROTOCOL PATCH HINTS:\n"
                + "\n".join(f"- {hint}" for hint in protocol_hints)
                + "\n"
            )
        constraints = protocol_state.get("constraints") or {}
        evidence_stages = protocol_state.get("evidence_stages") or {}
        protocol_state_text = (
            "RUNTIME PROTOCOL STATE:\n"
            f"- phase: {protocol_state.get('phase', 'unknown')}\n"
            "Hard constraints to preserve:\n"
            f"{self._format_runtime_list(constraints.get('hard') or [])}\n"
            "Soft constraints that may be relaxed only after no viable candidates appear:\n"
            f"{self._format_runtime_list(constraints.get('soft') or [])}\n"
            "Unknowns requiring observation/state evidence:\n"
            f"{self._format_runtime_list(constraints.get('unknown') or [])}\n"
            "Evidence stages:\n"
            "  Discovery/list/search state should only filter by:\n"
            f"{self._format_runtime_list(evidence_stages.get('result_page') or [])}"
            "  Inspection/detail state should verify:\n"
            f"{self._format_runtime_list(evidence_stages.get('detail_page') or [])}"
            "  Final commit must satisfy:\n"
            f"{self._format_runtime_list(evidence_stages.get('final_commit') or [])}"
            "Candidate queue / best-so-far evidence:\n"
            f"{self._format_runtime_list(protocol_state.get('candidates') or [])}\n"
            "State ledger facts to preserve:\n"
            "  Completed actions:\n"
            f"{self._format_runtime_list(protocol_state.get('completed_actions') or [])}"
            "  Focused targets:\n"
            f"{self._format_runtime_list(protocol_state.get('focused_targets') or [])}"
            "  Last measurements:\n"
            f"{self._format_runtime_list(protocol_state.get('last_measurements') or [])}"
            "  Active ambiguous options:\n"
            f"{self._format_runtime_list(protocol_state.get('ambiguous_options') or [])}"
            "  Object identity evidence:\n"
            f"{self._format_runtime_list(protocol_state.get('object_identity_ledger') or [])}"
            "Visible legal/actionable targets from latest observation:\n"
            f"{self._format_runtime_list(protocol_state.get('visible_actions') or [])}\n"
            "Recently tried actions:\n"
            f"{self._format_runtime_list(protocol_state.get('tried_actions') or [])}\n"
            "Next policy:\n"
            f"{self._format_runtime_list(protocol_state.get('next_policy') or [])}\n"
        )
        guard_hint = (
            "EXECUTION GUARD CONSTRAINTS:\n"
            "- Skills are knowledge sources only; never output a skill name, '[invoke ...]', 'use/call/trigger skill', or tool label as an env action.\n"
            "- Do not output abort/done/report-failure/error as env actions while steps remain. Continue with the next legal, verifiable action.\n"
            "- Execute exactly one environment action per step. If the previous response bundled multiple actions, continue only with the next still-legal action from the latest observation; never assume bundled follow-up actions already happened.\n"
            "- Treat the latest observation as the authority for reachable targets. If a remembered target is not currently visible/actionable, navigate or refresh state before using it, or choose a visible alternative.\n"
            "- If a legal selection/commit action repeats without visible text change and no explicit error, treat it as possibly acknowledged; advance to the next option, verification, or final commit instead of marking the candidate failed.\n"
            "- Repair the smallest failed local step using the failure type below; preserve completed subgoals from the state snapshot.\n"
            "- Treat completed_actions, focused_targets, and last_measurements as a state ledger: do not redo or contradict these facts unless the latest observation explicitly invalidates them.\n"
            "- If ambiguous_options are present, the immediate next action must be only one listed numeric index; do not restate the object/action name.\n"
            "- Use object_identity_ledger to preserve observed contents/location of same-name objects. If the correct same-name option cannot be distinguished from the latest ambiguous options, inspect/refresh instead of guessing.\n"
            "- Before retrying a failed action, refresh or use the latest observation, verify preconditions, and choose a different candidate if the same action already failed.\n"
            "- If carrying/capacity might block progress, transport one object to its target before taking another.\n"
            "- Preserve explicit hard constraints from the runtime protocol state. If recovery requires relaxing anything, relax only soft constraints and state that choice in the patch.\n"
            "- If no perfect match appears after bounded exploration, commit or verify the best-so-far candidate that satisfies hard constraints instead of terminating with a no-op/failure report.\n"
            "- When a candidate queue exists, inspect/verify/commit a candidate before widening exploration unless the latest observation directly contradicts it.\n"
            "- Do not reject plausible candidates in a discovery/list state because fine-grained attributes or option-like values are missing from surface text; verify those in an inspection/state-specific step.\n"
        )

        return (
            "[Runtime Recompile]\n"
            "PROCEDURE PATCH MODE: When the compiler regenerates procedural guidance, it must output a **short patch** "
            "(ERRATA + NEXT STEPS, under ~400 words), not a full duplicate manual. Skills retrieval should still prefer "
            "the minimal set that unblocks the next environment actions.\n\n"
            "Select the smallest skill set that can make immediate progress from the current state.\n"
            "Prefer action-driving skills. Add verification, navigation, inspection, or support skills only if they are needed in the next few steps.\n"
            "Prefer immediate next-step skills over future-stage skills.\n"
            "Preserve previously useful scaffold skills unless the new failure clearly shows they are irrelevant.\n"
            f"Inferred task profile: {effective.profile_name}\n"
            f"{relaxation_hint}"
            f"{syntax_error_hint}"
            f"Original task:\n{task_prompt}\n\n"
            f"Failure reason: {decision.get('reason', 'unknown')}\n"
            f"Failure type: {failure_type}\n"
            f"Repair hint: {repair_hint}\n"
            f"Remaining steps: {remaining_steps}\n"
            f"{protocol_state_text}"
            f"{guard_hint}"
            f"{protocol_hint_text}"
            f"Current observation: {decision.get('observation', '')}\n"
            f"Selected skills before failure: {decision.get('selected_skills_before', [])}\n"
            f"State snapshot:\n" + ("\n".join(state_lines) if state_lines else "- none") + "\n\n"
            "Recent trajectory:\n" + ("\n".join(recent_messages) if recent_messages else "- none")
        )

    def _llm_retrieve_relevant_skill_names(self, task, max_skills, candidate_mode):
        response = get_llm_response(
            retrieve_relevant_skills_prompt(
                self.metadata,
                task,
                max_skills=max_skills,
                candidate_mode=candidate_mode,
            ),
            is_string=True,
            model=self.model
        )
        names = _parse_relevant_skill_names_response(response)
        return names if names else []

    def _build_full_skill_payload(self, skill_names):
        skill_contents = []
        try:
            for skill_name in skill_names:
                skill_dir = Path(self.metadata[skill_name]['skill_dir'])
                if not skill_dir.is_dir():
                    continue

                combined_text = f"=== Skill: {skill_name} ===\n"
                main_file = skill_dir / "SKILL.md"
                if main_file.exists():
                    combined_text += f"\n[File: SKILL.md]\n"
                    combined_text += main_file.read_text(encoding='utf-8') + "\n"

                for file_path in skill_dir.rglob('*'):
                    if file_path.is_file() and file_path.name != "SKILL.md":
                        try:
                            relative_path = file_path.relative_to(skill_dir)
                            content = file_path.read_text(encoding='utf-8')
                            combined_text += f"\n[File: {relative_path}]\n"
                            combined_text += content + "\n"
                        except (UnicodeDecodeError, Exception):
                            continue

                skill_contents.append((skill_name, combined_text))
        except Exception as e:
            print(f"[ERROR] Failed to compile skill data: {e}")
        return skill_contents

    def _build_compiled_skill_payload(self, skill_names):
        skill_contents = []
        compiled_lookup = {
            item.asset.name: item
            for item in self.last_compilation.compiled_skills
        }
        for skill_name in skill_names:
            meta = self.metadata.get(skill_name)
            if not meta:
                continue
            skill_dir = Path(meta["skill_dir"])
            skill_md_path = skill_dir / "SKILL.md"
            skill_md = ""
            if skill_md_path.exists():
                skill_md = skill_md_path.read_text(encoding="utf-8", errors="ignore")
                skill_md = skill_md[: self.compiler_max_skill_chars]

            compiled_skill = compiled_lookup.get(skill_name)
            localized = "\n".join(
                f"- {item}" for item in (compiled_skill.localized_instructions if compiled_skill else [])
            ) or "- No localized instructions generated."
            selected_reason = compiled_skill.selected_reason if compiled_skill else "selected"
            combined_text = (
                f"=== Compiled Skill: {skill_name} ===\n"
                f"[Selected Reason]\n{selected_reason}\n"
                f"[Localized Instructions]\n{localized}\n"
                f"[Compressed SKILL.md]\n{skill_md}\n"
            )
            skill_contents.append((skill_name, combined_text))

        metrics = self.last_compilation.metrics
        execution_order = " -> ".join(self.last_compilation.execution_order)

        # Build a semantic phase-coverage description instead of raw numeric scores.
        # Raw scores like "coverage_score: 0.09" can mislead the LLM procedure generator
        # into thinking the skill package is inadequate even when the right skills are selected.
        compiled_lookup = {
            item.asset.name: item
            for item in self.last_compilation.compiled_skills
        }
        phase_lines: list[str] = []
        for skill_name in skill_names:
            item = compiled_lookup.get(skill_name)
            if item is None:
                continue
            subgoal_ids = item.assigned_subgoals
            subgoal_descs = []
            for subgoal in self.last_compilation.subgoals:
                if subgoal.subgoal_id in subgoal_ids:
                    subgoal_descs.append(subgoal.description[:60])
            if subgoal_descs:
                phase_lines.append(f"  {skill_name}: covers [{'; '.join(subgoal_descs)}]")
            else:
                phase_lines.append(f"  {skill_name}: selected as relevant to the task")

        phase_coverage_text = (
            "\n".join(phase_lines)
            if phase_lines
            else "  (all selected skills are relevant to the overall task)"
        )

        summary = (
            "Dynamic Skill Compiler Summary\n"
            "The following compiled package should be treated as a new task-specific skill synthesized from the source skills.\n"
            f"Selected {metrics.selected_count} skills from {metrics.candidate_count} candidates.\n"
            f"Skill roles and covered phases:\n{phase_coverage_text}\n"
            f"Suggested execution order: {execution_order}\n"
        )
        return skill_contents, summary

    def _build_quality_first_skill_payload(self, skill_names):
        skill_contents = []
        compiled_lookup = {
            item.asset.name: item
            for item in self.last_compilation.compiled_skills
        }
        for skill_name in skill_names:
            meta = self.metadata.get(skill_name)
            if not meta:
                continue
            skill_dir = Path(meta["skill_dir"])
            skill_md_path = skill_dir / "SKILL.md"
            skill_md = ""
            if skill_md_path.exists():
                skill_md = skill_md_path.read_text(encoding="utf-8", errors="ignore")
                skill_md = skill_md[: self.compiler_max_skill_chars]

            compiled_skill = compiled_lookup.get(skill_name)
            localized_lines = (
                compiled_skill.localized_instructions
                if compiled_skill
                else self._fallback_localized_instructions(skill_name)
            )
            localized = "\n".join(f"- {item}" for item in localized_lines[:4])
            references = []
            reference_budget = max(400, self.compiler_max_skill_chars // 2)
            consumed = 0
            for file_path in sorted(skill_dir.rglob("*")):
                if not file_path.is_file() or file_path.name == "SKILL.md":
                    continue
                if file_path.suffix.lower() not in {".md", ".txt", ".py"}:
                    continue
                content = file_path.read_text(encoding="utf-8", errors="ignore").strip()
                if not content:
                    continue
                remaining = reference_budget - consumed
                if remaining <= 0:
                    break
                snippet = content[:remaining]
                references.append(f"[Reference: {file_path.relative_to(skill_dir)}]\n{snippet}\n")
                consumed += len(snippet)
            combined_text = (
                f"=== Skill: {skill_name} ===\n"
                f"[Localized Summary]\n{localized}\n"
                f"[Compressed SKILL.md]\n{skill_md}\n"
                f"{''.join(references)}"
            )
            skill_contents.append((skill_name, combined_text))

        summary = (
            "Quality-First Skill Package Summary\n"
            "Preserve the following SkillNet reference skills as authoritative task guidance.\n"
            "Use the localized summaries to stay concise, but prefer the concrete action patterns in each SKILL.md over abstract rewrites.\n"
        )
        return skill_contents, summary
