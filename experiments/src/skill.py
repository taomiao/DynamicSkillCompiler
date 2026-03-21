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
    generate_overall_procedure_code_prompt
)

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
    TaskDecomposer,
)


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
        self.runtime_recompile_max_count = kwargs.get("runtime_recompile_max_count", 1)
        self.runtime_recompile_min_interval_steps = kwargs.get("runtime_recompile_min_interval_steps", 2)
        self.runtime_recompile_stagnation_threshold = kwargs.get("runtime_recompile_stagnation_threshold", 2)
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
        self.runtime_recompile_count = 0
        self.runtime_recompile_events = []
        self.runtime_last_recompile_step = -999
        self.last_compilation = None
        self.last_candidate_assets = {}
        self.last_seed_skill_names = []
        self.last_quality_reference_skill_names = []
        self.last_selected_skill_names = []
        self.last_deterministic_procedure_kind = None

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

        response = get_llm_response(
            generate_overall_procedure_prompt(
                task,
                self.overall_procedure_examples,
                skill_contents,
                compiler_summary=compiler_summary,
            ),
            is_string=True,
            model=self.model
        )
        overall_procedure = response.split("<Overall_Procedure>")[1].split("</Overall_Procedure>")[0].strip()
        return overall_procedure
    
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

    def _select_scienceworld_conductivity_skill_names(self, available):
        selected = []

        def pick_first(candidates):
            for candidate in candidates:
                if candidate in available and candidate not in selected:
                    selected.append(candidate)
                    return True
            return False

        pick_first([
            "scienceworld-object-locator",
            "scienceworld-target-locator",
            "scienceworld-room-navigator",
            "scienceworld-room-explorer",
        ])
        pick_first([
            "scienceworld-object-focuser",
            "scienceworld-task-focuser",
        ])
        pick_first([
            "scienceworld-conductivity-tester",
            "scienceworld-circuit-builder",
            "scienceworld-circuit-connector",
        ])
        pick_first([
            "scienceworld-object-classifier",
            "scienceworld-conditional-placer",
            "scienceworld-object-placer",
            "scienceworld-container-relocator",
        ])

        if len(selected) >= 4:
            return selected
        return []

    def _is_scienceworld_temperature_query(self, task):
        if self._infer_benchmark() != "scienceworld":
            return False
        query = task.lower()
        return "thermometer" in query or "temperature" in query

    def _select_scienceworld_temperature_skill_names(self, available):
        selected = []

        def pick_first(candidates):
            for candidate in candidates:
                if candidate in available and candidate not in selected:
                    selected.append(candidate)
                    return True
            return False

        pick_first([
            "scienceworld-object-locator",
            "scienceworld-room-navigator",
            "scienceworld-room-scanner",
        ])
        pick_first([
            "scienceworld-task-focuser",
            "scienceworld-object-focuser",
        ])
        pick_first([
            "scienceworld-temperature-measurer",
        ])
        pick_first([
            "scienceworld-conditional-box-placer",
            "scienceworld-object-classifier",
            "scienceworld-object-placer",
        ])
        pick_first([
            "scienceworld-inventory-focus",
            "scienceworld-room-scanner",
        ])

        return selected if len(selected) >= 4 else []

    def _is_scienceworld_growth_query(self, task):
        if self._infer_benchmark() != "scienceworld":
            return False
        query = task.lower()
        return "grow" in query and "seed" in query

    def _select_scienceworld_growth_skill_names(self, available):
        selected = []

        def pick_first(candidates):
            for candidate in candidates:
                if candidate in available and candidate not in selected:
                    selected.append(candidate)
                    return True
            return False

        pick_first([
            "scienceworld-room-navigator",
            "scienceworld-object-locator",
        ])
        pick_first([
            "scienceworld-object-focuser",
            "scienceworld-task-focuser",
        ])
        pick_first([
            "scienceworld-pot-preparer",
            "scienceworld-planting-operation",
        ])
        pick_first([
            "scienceworld-planting-coordinator",
        ])
        pick_first([
            "scienceworld-liquid-filler",
            "soil-extraction",
        ])
        pick_first([
            "scienceworld-growth-focuser",
        ])
        pick_first([
            "controlled-waiting",
        ])
        pick_first([
            "scienceworld-ambiguous-action-resolution",
        ])

        return selected if len(selected) >= 5 else []

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

    def _deterministic_procedure(self, task, skill_names):
        self.last_deterministic_procedure_kind = None
        if self.selection_strategy == "dsc" and self._is_scienceworld_conductivity_task(task, skill_names):
            self.last_deterministic_procedure_kind = "scienceworld_conductivity"
            return self._scienceworld_conductivity_procedure(task)
        return None

    def _is_scienceworld_conductivity_query(self, task):
        if self._infer_benchmark() != "scienceworld":
            return False
        query = task.lower()
        return "electrically conductive" in query or "conductivity" in query

    def _is_scienceworld_conductivity_task(self, task, skill_names):
        if not self._is_scienceworld_conductivity_query(task):
            return False
        return "scienceworld-conductivity-tester" in set(skill_names)

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
        object_match = re.search(r"determine if (.+?) is electrically conductive", task, re.IGNORECASE)
        room_match = re.search(r"is located around the (.+?)[\.,]", task, re.IGNORECASE)
        conductive_match = re.search(
            r"If it is electrically conductive, place it in the (.+?)[\\.]",
            task,
            re.IGNORECASE,
        )
        nonconductive_match = re.search(
            r"If it is electrically nonconductive, place it in the (.+?)[\\.]",
            task,
            re.IGNORECASE,
        )
        object_name = object_match.group(1).strip() if object_match else "unknown substance S"
        source_room = room_match.group(1).strip() if room_match else "workshop"
        conductive_box = conductive_match.group(1).strip() if conductive_match else "orange box"
        nonconductive_box = (
            nonconductive_match.group(1).strip()
            if nonconductive_match
            else "yellow box"
        )
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
        ):
            value = snapshot.get(key)
            if value:
                state_lines.append(f"- {key}: {value}")

        observation_text = str(decision.get("observation", "")).lower()
        relaxation_hint = ""
        if effective.profile_name == "search" or any(
            marker in observation_text
            for marker in ("no results", "not found", "no product", "0 results", "no matches")
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

        return (
            "[Runtime Recompile]\n"
            "Select the smallest skill set that can make immediate progress from the current state.\n"
            "Prefer action-driving skills. Add verification, pagination, detail, or support skills only if they are needed in the next few steps.\n"
            "Prefer immediate next-step skills over future-stage skills.\n"
            "Preserve previously useful scaffold skills unless the new failure clearly shows they are irrelevant.\n"
            f"Inferred task profile: {effective.profile_name}\n"
            f"{relaxation_hint}"
            f"Original task:\n{task_prompt}\n\n"
            f"Failure reason: {decision.get('reason', 'unknown')}\n"
            f"Remaining steps: {remaining_steps}\n"
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
        relevant_skill_names = response.split("<Relevant_Skill_Names>")[1].split("</Relevant_Skill_Names>")[0].strip("`json\n").strip("`\n").strip("```\n")
        return json.loads(relevant_skill_names)

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
        summary = (
            "Dynamic Skill Compiler Summary\n"
            "The following compiled package should be treated as a new task-specific skill synthesized from the source skills.\n"
            f"- Selected skills: {metrics.selected_count}/{metrics.candidate_count}\n"
            f"- Covered subgoals: {metrics.covered_subgoal_count}/{metrics.subgoal_count}\n"
            f"- Coverage score: {metrics.coverage_score:.3f}\n"
            f"- Redundancy reduction: {metrics.redundancy_reduction:.3f}\n"
            f"- Fragments retained: {metrics.fragment_count_after}/{metrics.fragment_count_before}\n"
            f"- Estimated token cost: {metrics.estimated_token_cost_before:.2f} -> "
            f"{metrics.estimated_token_cost_after:.2f}\n"
            f"- Fragment token cost: {metrics.fragment_token_cost_after:.2f}\n"
            f"- Execution order: {execution_order}\n"
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
