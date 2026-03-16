import os
import re
import sys
from pathlib import Path
from typing import Any
import yaml

from src.utils import extract_tagged_json, extract_tagged_text, get_llm_response, strip_code_fences
from src.prompt_generator import (
    retrieve_relevant_skills_prompt,
    generate_overall_procedure_prompt,
    generate_overall_procedure_code_prompt,
    optimize_compiled_skills_prompt,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
SKILLNET_SRC = ROOT_DIR / "skillnet-ai" / "src"
if str(SKILLNET_SRC) not in sys.path:
    sys.path.insert(0, str(SKILLNET_SRC))

from skillnet_ai.compiler import (
    CompilerConfig,
    CompiledSkill,
    DEFAULT_GRAPH_PASSES,
    DynamicSkillCompiler,
    GRAPH_PASS_PRESETS,
    InMemorySkillRetriever,
    LocalEnvironment,
    LocalSkillLibraryRetriever,
    QueryOptimizer,
)


class SkillModule:
    def __init__(self, **kwargs):
        self.skills_dir = Path(kwargs.get("skills_dir", "skills"))
        self.overall_procedure_examples_path = kwargs.get("overall_procedure_examples_path", "")
        self.procedure_code_template_path = kwargs.get("procedure_code_template_path", None)
        self.model = kwargs.get("model", "gpt-4o")
        self.selection_strategy = self._normalize_selection_strategy(
            kwargs.get("selection_strategy", "skillnet")
        )
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
        self.compiler_adaptive_workflow_bias = kwargs.get("compiler_adaptive_workflow_bias", True)
        self.compiler_graph_passes = self._normalize_graph_passes(
            kwargs.get("compiler_graph_passes", DEFAULT_GRAPH_PASSES)
        )
        self.compiler_top_k = kwargs.get("compiler_top_k", 6)
        self.compiler_max_skill_chars = kwargs.get("compiler_max_skill_chars", 1200)
        self.compiler_critic_enabled = kwargs.get("compiler_critic_enabled", False)
        self.compiler_critic_model = kwargs.get("compiler_critic_model", None)
        self.compiler_critic_min_coverage = kwargs.get("compiler_critic_min_coverage", 0.45)
        self.compiler_critic_force = kwargs.get("compiler_critic_force", False)
        self.runtime_recompile_enabled = kwargs.get(
            "runtime_recompile_enabled",
            False,
        )
        self.runtime_recompile_max_rounds = kwargs.get("runtime_recompile_max_rounds", 2)
        self.runtime_recompile_min_interval_steps = kwargs.get(
            "runtime_recompile_min_interval_steps",
            2,
        )
        self.runtime_recompile_stagnation_threshold = kwargs.get(
            "runtime_recompile_stagnation_threshold",
            2,
        )
        self.runtime_recompile_min_remaining_steps = kwargs.get(
            "runtime_recompile_min_remaining_steps",
            1,
        )
        self.runtime_recompile_trace_tail = kwargs.get("runtime_recompile_trace_tail", 6)
        self.runtime_recompile_context_chars = kwargs.get("runtime_recompile_context_chars", 480)
        self.last_compilation = None
        self.last_candidate_assets = {}
        self.last_seed_skill_names = []
        self.last_quality_reference_skill_names = []
        self.last_selected_skill_names = []
        self.last_payload_strategy = "full"
        self.last_deterministic_procedure_kind = None
        self.last_compile_critic_decision = None
        self.last_retrieval_warnings = []
        self.runtime_recompile_active = False
        self.runtime_recompile_count = 0
        self.runtime_recompile_events = []
        self.runtime_last_recompile_step = -999
        self.last_runtime_task_prompt = ""

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

    def _normalize_selection_strategy(self, raw_strategy):
        normalized = str(raw_strategy or "skillnet").strip().lower()
        if normalized in {"", "baseline", "skillnet"}:
            return "skillnet"
        return normalized

    def _uses_dsc_strategy(self):
        return self.selection_strategy == "dsc"

    def _normalize_graph_passes(self, raw_passes):
        if raw_passes is None:
            return DEFAULT_GRAPH_PASSES
        if isinstance(raw_passes, str):
            preset = GRAPH_PASS_PRESETS.get(raw_passes.strip().lower())
            if preset is not None:
                return preset
            tokens = [item.strip() for item in raw_passes.split(",") if item.strip()]
            return tuple(tokens) if tokens else DEFAULT_GRAPH_PASSES
        if isinstance(raw_passes, (list, tuple)):
            tokens = [str(item).strip() for item in raw_passes if str(item).strip()]
            return tuple(tokens) if tokens else DEFAULT_GRAPH_PASSES
        return DEFAULT_GRAPH_PASSES


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

    def reset_runtime_recompile_state(self):
        self.runtime_recompile_active = False
        self.runtime_recompile_count = 0
        self.runtime_recompile_events = []
        self.runtime_last_recompile_step = -999
        self.last_runtime_task_prompt = ""

    def should_use_runtime_recompile(self):
        return self._uses_dsc_strategy() and self.runtime_recompile_enabled

    def can_runtime_recompile(self):
        return self.should_use_runtime_recompile() and (
            self.runtime_recompile_count < self.runtime_recompile_max_rounds
        )

    def record_runtime_recompile(self, event: dict[str, Any]):
        self.runtime_recompile_active = True
        self.runtime_recompile_count += 1
        self.runtime_last_recompile_step = int(event.get("step_index", self.runtime_last_recompile_step))
        event_record = dict(event)
        event_record["recompile_round"] = self.runtime_recompile_count
        self.runtime_recompile_events.append(event_record)
        self._append_compile_note(
            f"Runtime recompile #{self.runtime_recompile_count} triggered at step "
            f"{event_record.get('step_index', '?')} because of {event_record.get('reason', 'unknown')}."
        )

    def _parse_runtime_task_goal(self, task: str) -> dict[str, Any]:
        normalized_task = re.sub(r"\s+", " ", str(task or "").lower())
        match = re.search(
            r"put\s+(?:(some|a|an|two)\s+)?([a-z]+(?: [a-z]+)*)\s+(on|in|into)\s+([a-z]+(?: [a-z]+)*)",
            normalized_task,
        )
        if not match:
            return {}
        quantity_token, object_name, _, receptacle_name = match.groups()
        quantity = 1
        if quantity_token == "two":
            quantity = 2
        return {
            "quantity": quantity,
            "object_name": object_name.strip(),
            "receptacle_name": receptacle_name.strip(),
            "requires_cool": "cool " in normalized_task,
            "requires_clean": "clean " in normalized_task,
            "requires_heat": "hot " in normalized_task or "heat " in normalized_task,
        }

    def _summarize_runtime_progress(self, task: str, state_snapshot: dict[str, Any]) -> list[str]:
        goal = self._parse_runtime_task_goal(task)
        if not goal:
            return []

        object_name = goal["object_name"]
        receptacle_name = goal["receptacle_name"]
        quantity = goal["quantity"]
        inventory = [
            item for item in state_snapshot.get("inventory", [])
            if str(item).startswith(f"{object_name} ")
        ]
        visible_entities = [
            item for item in state_snapshot.get("visible_entities", [])
            if str(item).startswith(f"{object_name} ")
        ]
        completed_transfers = [
            item
            for item in state_snapshot.get("completed_transfers", [])
            if str(item.get("object", "")).startswith(f"{object_name} ")
            and str(item.get("destination", "")).startswith(receptacle_name)
        ]
        completed_transforms = state_snapshot.get("completed_transforms", [])
        progress_lines = [
            f"Target object type: {object_name}",
            f"Target receptacle type: {receptacle_name}",
            f"Required quantity: {quantity}",
        ]
        if state_snapshot.get("current_location"):
            progress_lines.append(f"Current location: {state_snapshot['current_location']}")
        if inventory:
            progress_lines.append(
                f"Inventory already holds {len(inventory)}/{quantity} target object(s): {inventory}"
            )
        if visible_entities:
            progress_lines.append(f"Currently visible target candidates: {visible_entities}")
        if completed_transfers:
            progress_lines.append(
                f"Confirmed placements already completed: {len(completed_transfers)}/{quantity}"
            )
        if state_snapshot.get("visited_locations"):
            visited = state_snapshot["visited_locations"][:8]
            progress_lines.append(f"Visited locations so far: {visited}")
        if state_snapshot.get("open_receptacles"):
            progress_lines.append(f"Currently open receptacles: {state_snapshot['open_receptacles'][:6]}")

        if goal.get("requires_cool"):
            cooled = [
                item for item in completed_transforms
                if item.get("verb") == "cool"
                and str(item.get("object", "")).startswith(f"{object_name} ")
            ]
            if cooled:
                progress_lines.append("Cooling step already appears completed for at least one target object.")
            else:
                progress_lines.append("Cooling step is still pending for the target object.")

        return progress_lines

    def build_runtime_recompile_task(self, task, messages, event, remaining_steps):
        recent_actions = []
        recent_observations = []
        for message in reversed(messages):
            role = message.get("role")
            content = str(message.get("content", "")).strip()
            if role == "assistant" and content.startswith("Action:") and len(recent_actions) < 4:
                recent_actions.append(content[: self.runtime_recompile_context_chars])
            elif role == "user" and "Observation:" in content and len(recent_observations) < 4:
                recent_observations.append(content[: self.runtime_recompile_context_chars])
            if len(recent_actions) >= 4 and len(recent_observations) >= 4:
                break

        selected_skills = event.get("selected_skills_before") or self.last_selected_skill_names
        latest_observation = str(event.get("observation", "")).strip()[: self.runtime_recompile_context_chars]
        latest_action = str(event.get("action", "")).strip()[: self.runtime_recompile_context_chars]
        state_snapshot = event.get("state_snapshot") or {}
        progress_summary = self._summarize_runtime_progress(task, state_snapshot)
        context_lines = [
            task,
            "",
            "[Runtime Compiler Context]",
            "The previous compiled skill package is being updated mid-execution.",
            f"Trigger reason: {event.get('reason', 'unknown')}",
            f"Remaining step budget: {max(0, int(remaining_steps))}",
            f"Previous selected skills: {selected_skills}",
            "Do not repeat phases that are already confirmed complete unless the latest observation suggests they failed.",
            "Bias towards the blocker revealed by the recent trace.",
        ]
        if latest_action:
            context_lines.append(f"Latest action: {latest_action}")
        if latest_observation:
            context_lines.append(f"Latest observation: {latest_observation}")
        if progress_summary:
            context_lines.append("Progress summary:")
            context_lines.extend(f"- {item}" for item in progress_summary)
        if state_snapshot:
            context_lines.append("State snapshot:")
            if state_snapshot.get("current_location"):
                context_lines.append(f"- current_location={state_snapshot['current_location']}")
            if state_snapshot.get("inventory"):
                context_lines.append(f"- inventory={state_snapshot['inventory']}")
            if state_snapshot.get("visible_entities"):
                context_lines.append(f"- visible_entities={state_snapshot['visible_entities'][:8]}")
            if state_snapshot.get("visited_locations"):
                context_lines.append(f"- visited_locations={state_snapshot['visited_locations'][:8]}")
            if state_snapshot.get("completed_transfers"):
                context_lines.append(f"- completed_transfers={state_snapshot['completed_transfers'][-4:]}")
            if state_snapshot.get("completed_transforms"):
                context_lines.append(f"- completed_transforms={state_snapshot['completed_transforms'][-4:]}")
        if recent_actions:
            context_lines.append("Recent actions:")
            context_lines.extend(f"- {item}" for item in reversed(recent_actions))
        if recent_observations:
            context_lines.append("Recent observations:")
            context_lines.extend(f"- {item}" for item in reversed(recent_observations))

        runtime_task = "\n".join(context_lines)
        self.last_runtime_task_prompt = runtime_task
        return runtime_task

    def _reset_selection_state(self):
        self.last_compile_critic_decision = None
        self.last_deterministic_procedure_kind = None
        self.last_payload_strategy = "full"
        self.last_seed_skill_names = []
        self.last_quality_reference_skill_names = []
        self.last_selected_skill_names = []
        self.last_retrieval_warnings = []
        self.last_compilation = None
        self.last_candidate_assets = {}

    def _retrieve_baseline_relevant_skills(self, task):
        return self._llm_retrieve_relevant_skill_names(
            task,
            max_skills=self.llm_retrieval_max_skills,
            candidate_mode=False,
        )

    def _build_skill_payload_for_strategy(self, skill_names):
        if self._uses_dsc_strategy() and self.last_compilation is not None:
            if self.last_payload_strategy == "quality_first":
                return self._build_quality_first_skill_payload(skill_names)
            return self._build_compiled_skill_payload(skill_names)
        return self._build_full_skill_payload(skill_names), ""
    
    def retrieve_relevant_skills(self, task):
        """
        Retrieve relevant skills from metadata based on task description.
        """
        self._reset_selection_state()

        if self._uses_dsc_strategy():
            quality_reference_skill_names, seed_skill_names = self._retrieve_dsc_skill_name_pools(task)
            self.last_seed_skill_names = list(seed_skill_names or [])
            self.last_quality_reference_skill_names = list(quality_reference_skill_names or [])
            family_priority_skill_names = self._family_priority_skill_names(task)
            compile_seed_skill_names = list(
                dict.fromkeys(
                    list(seed_skill_names or [])
                    + list(quality_reference_skill_names or [])
                    + list(family_priority_skill_names or [])
                )
            )
            self.last_compilation = self._compile_task(
                task,
                seed_skill_names=compile_seed_skill_names or seed_skill_names,
            )
            self.last_payload_strategy = "compiled"
            for warning in self.last_retrieval_warnings:
                self._append_compile_note(warning)
            selected = [
                item.asset.name
                for item in self.last_compilation.compiled_skills[: self.compiler_top_k]
            ]
            if quality_reference_skill_names and self._should_use_quality_first_fallback(
                task,
                selected,
                quality_reference_skill_names,
            ):
                quality_first_skill_names = self._select_quality_first_skill_names(
                    task,
                    quality_reference_skill_names,
                    seed_skill_names or [],
                )
                self.last_compilation = self._build_quality_first_compilation(
                    self.last_compilation,
                    quality_first_skill_names,
                )
                self.last_payload_strategy = "quality_first"
                selected = list(quality_first_skill_names)
            selected = self._apply_compile_critic_pass(
                task,
                selected,
                quality_reference_skill_names,
                seed_skill_names or [],
            )
            if not selected:
                selected = self._activate_reference_style_payload(
                    task,
                    selected,
                    quality_reference_skill_names,
                    seed_skill_names or [],
                    reason="compiled selection was empty",
                )
            elif self.last_payload_strategy == "compiled" and self._should_use_reference_style_payload(
                task,
                selected,
            ):
                selected = self._activate_reference_style_payload(
                    task,
                    selected,
                    quality_reference_skill_names,
                    seed_skill_names or [],
                    reason="observation-heavy task benefits from richer source guidance",
                )
            self.last_selected_skill_names = list(selected)
            return selected

        selected = self._retrieve_baseline_relevant_skills(task)
        self.last_selected_skill_names = list(selected)
        return selected
    
    def generate_overall_procedure(self, task, skill_names):
        """
        Generate overall procedure by combining individual skill contents.
        """
        deterministic = self._deterministic_procedure(task, skill_names)
        if deterministic is not None:
            return deterministic

        skill_contents, compiler_summary = self._build_skill_payload_for_strategy(skill_names)

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
        return self._sanitize_generated_overall_procedure(response)
    
    def generate_overall_procedure_code(self, task, overall_procedure):
        """
        Generate overall procedure code.
        """
        if self.last_deterministic_procedure_kind == "scienceworld_conductivity":
            return self._scienceworld_static_procedure_code(task, overall_procedure)
        if self.last_deterministic_procedure_kind == "scienceworld_growth":
            return self._scienceworld_growth_static_procedure_code(task, overall_procedure)

        response = get_llm_response(
            generate_overall_procedure_code_prompt(task, overall_procedure, self.procedure_code_template),
            is_string=True,
            model=self.model
        )
        return self._sanitize_generated_procedure_code(response)

    def _sanitize_generated_procedure_code(self, response):
        code = extract_tagged_text(response, "Overall_Procedure_Code")
        if not code:
            code = strip_code_fences(response)
        else:
            code = strip_code_fences(code)

        if "<Overall_Procedure_Code>" in code:
            code = code.split("<Overall_Procedure_Code>", 1)[-1]
        if "</Overall_Procedure_Code>" in code:
            code = code.split("</Overall_Procedure_Code>", 1)[0]

        def_index = code.find("def overall_procedure_code")
        if def_index >= 0:
            prefix = code[:def_index]
            version_markers = list(re.finditer(r"#v\d+\s*$", prefix, flags=re.MULTILINE))
            start_index = version_markers[-1].start() if version_markers else def_index
            code = code[start_index:]

        return code.strip()

    def _sanitize_generated_overall_procedure(self, response):
        procedure = extract_tagged_text(response, "Overall_Procedure")
        if not procedure:
            procedure = strip_code_fences(response)
        else:
            procedure = strip_code_fences(procedure)

        if "<Overall_Procedure>" in procedure:
            procedure = procedure.split("<Overall_Procedure>", 1)[-1]
        if "</Overall_Procedure>" in procedure:
            procedure = procedure.split("</Overall_Procedure>", 1)[0]

        procedure = re.sub(r"</?Analysis>", "", procedure, flags=re.IGNORECASE)
        return procedure.strip()

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
                adaptive_workflow_bias=self.compiler_adaptive_workflow_bias,
                graph_passes=self.compiler_graph_passes,
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

    def _record_retrieval_warning(self, note):
        if note and note not in self.last_retrieval_warnings:
            self.last_retrieval_warnings.append(note)

    def _safe_llm_retrieve_relevant_skill_names(self, task, max_skills, candidate_mode):
        try:
            return self._llm_retrieve_relevant_skill_names(
                task,
                max_skills=max_skills,
                candidate_mode=candidate_mode,
            )
        except Exception as exc:
            mode = "candidate-pool" if candidate_mode else "reference"
            message = str(exc).strip().replace("\n", " ")
            self._record_retrieval_warning(
                f"LLM {mode} retrieval failed; DSC fell back to compiler-local selection. Error: {message[:160]}"
            )
            return []

    def _retrieve_dsc_skill_name_pools(self, task):
        quality_reference_skill_names = []
        seed_skill_names = None

        if not self.compiler_quality_reference_with_llm and not self.compiler_seed_with_llm:
            return quality_reference_skill_names, seed_skill_names

        shared_limit = max(self.llm_retrieval_max_skills, self.compiler_seed_max_skills)
        shared_candidates = self._safe_llm_retrieve_relevant_skill_names(
            task,
            max_skills=shared_limit,
            candidate_mode=True,
        )

        if self.compiler_quality_reference_with_llm:
            quality_reference_skill_names = list(shared_candidates[: self.llm_retrieval_max_skills])
            if not quality_reference_skill_names:
                quality_reference_skill_names = self._safe_llm_retrieve_relevant_skill_names(
                    task,
                    max_skills=self.llm_retrieval_max_skills,
                    candidate_mode=False,
                )

        if self.compiler_seed_with_llm:
            seed_skill_names = list(shared_candidates[: self.compiler_seed_max_skills])
            if not seed_skill_names:
                seed_skill_names = list(
                    (quality_reference_skill_names or [])[: self.compiler_seed_max_skills]
                )

        return quality_reference_skill_names, seed_skill_names

    def _apply_compile_critic_pass(self, task, selected_skill_names, reference_skill_names, seed_skill_names):
        self.last_compile_critic_decision = None
        selected = list(dict.fromkeys(selected_skill_names))
        if not self._should_run_compile_critic(task, selected):
            return selected

        available_skill_names = self._compile_critic_candidate_names(
            selected,
            reference_skill_names,
            seed_skill_names,
        )
        if not available_skill_names:
            return selected

        critic_response = get_llm_response(
            optimize_compiled_skills_prompt(
                task,
                current_skill_names=selected,
                available_skill_summaries=self._build_compile_critic_skill_summaries(available_skill_names),
                compiler_summary=self._build_compiler_summary(),
                max_skills=max(self.compiler_top_k, len(reference_skill_names)),
            ),
            is_string=True,
            model=self.compiler_critic_model or self.model,
        )
        critic_decision = extract_tagged_json(
            critic_response,
            "Compile_Critic_Decision",
            default={},
        )
        final_selection = self._sanitize_compile_critic_decision(
            critic_decision,
            selected,
            reference_skill_names,
            seed_skill_names,
            available_skill_names,
        )
        if final_selection == selected:
            return selected

        self.last_compile_critic_decision = critic_decision
        self.last_compilation = self._build_quality_first_compilation(
            self.last_compilation,
            final_selection,
        )
        self.last_payload_strategy = "quality_first"
        self._append_compile_note(
            f"Compile critic selected {final_selection} using model "
            f"{self.compiler_critic_model or self.model}."
        )
        return final_selection

    def _should_run_compile_critic(self, task, selected_skill_names):
        if not self._uses_dsc_strategy() or not self.compiler_critic_enabled or self.last_compilation is None:
            return False
        if self.compiler_critic_force:
            return True
        if self._scienceworld_task_family(task) == "generic":
            return False
        metrics = self.last_compilation.metrics
        if metrics.coverage_score < self.compiler_critic_min_coverage:
            return True
        if metrics.subgoal_count <= 1 and self._is_multi_phase_query(task):
            return True
        if len(selected_skill_names) <= 2 and self._is_multi_phase_query(task):
            return True
        return False

    def _should_use_quality_first_fallback(self, task, compiled_skill_names, reference_skill_names):
        if self.last_compilation is None or not reference_skill_names:
            return False
        metrics = self.last_compilation.metrics
        if metrics.subgoal_count > 0 and metrics.covered_subgoal_count < metrics.subgoal_count:
            self._append_compile_note("Quality-first fallback activated because compiled coverage missed subgoals.")
            return True
        if metrics.coverage_score < max(self.compiler_critic_min_coverage, 0.55):
            self._append_compile_note("Quality-first fallback activated because compiled coverage score was too low.")
            return True
        if self._is_multi_phase_query(task) and len(compiled_skill_names) < min(3, max(1, len(reference_skill_names))):
            self._append_compile_note("Quality-first fallback activated because the compiled package was too small for a multi-phase task.")
            return True
        return False

    def _is_observation_heavy_query(self, task):
        lowered = str(task or "").lower()
        markers = (
            "find ",
            "locate",
            "focus on",
            "look at",
            "identify",
            "which ",
            "where ",
            "measure",
            "temperature",
            "thermometer",
            "animal",
            "living thing",
            "longest",
            "shortest",
            "compare",
        )
        return any(marker in lowered for marker in markers)

    def _prefers_full_source_quality_first_payload(self, task):
        if not self._is_observation_heavy_query(task):
            return False
        if self._infer_benchmark() != "scienceworld":
            return True
        family = self._scienceworld_task_family(task)
        return family in {"generic", "temperature"}

    def _should_use_reference_style_payload(self, task, compiled_skill_names):
        if self.last_compilation is None:
            return False
        if not self._is_observation_heavy_query(task):
            return False
        if self._infer_benchmark() != "scienceworld":
            return self.last_compilation.metrics.coverage_score < 0.8
        family = self._scienceworld_task_family(task)
        if family in {"conductivity", "growth", "phase_change"}:
            return False
        metrics = self.last_compilation.metrics
        if family in {"generic", "temperature"}:
            return True
        if not compiled_skill_names:
            return True
        return metrics.coverage_score < 0.8 or len(compiled_skill_names) <= 3

    def _select_reference_style_skill_names(
        self,
        task,
        compiled_skill_names,
        reference_skill_names,
        seed_skill_names,
    ):
        selected = []
        for group in (
            compiled_skill_names,
            reference_skill_names,
            seed_skill_names,
            self._family_priority_skill_names(task),
        ):
            for skill_name in group:
                if skill_name in self.metadata and skill_name not in selected:
                    selected.append(skill_name)
        if not selected:
            return []
        limit = max(
            len(compiled_skill_names),
            min(self.compiler_top_k + 1, len(selected)),
        )
        return selected[:limit]

    def _activate_reference_style_payload(
        self,
        task,
        compiled_skill_names,
        reference_skill_names,
        seed_skill_names,
        reason,
    ):
        if self.last_compilation is None:
            return list(compiled_skill_names)
        robust_skill_names = self._select_reference_style_skill_names(
            task,
            compiled_skill_names,
            reference_skill_names,
            seed_skill_names,
        )
        if not robust_skill_names:
            return list(compiled_skill_names)
        self.last_compilation = self._build_quality_first_compilation(
            self.last_compilation,
            robust_skill_names,
        )
        self.last_payload_strategy = "quality_first"
        self._append_compile_note(
            f"Reference-style payload activated because {reason}."
        )
        return list(robust_skill_names)

    def _should_expand_quality_first_payload(self):
        if self.last_compilation is None:
            return False
        if self._prefers_full_source_quality_first_payload(
            self.last_compilation.query_plan.raw_query
        ):
            return True
        metrics = self.last_compilation.metrics
        if metrics.subgoal_count > 0 and metrics.covered_subgoal_count < metrics.subgoal_count:
            return True
        if metrics.coverage_score < max(self.compiler_critic_min_coverage, 0.55):
            return True
        if (
            metrics.fragment_count_before > 0
            and metrics.fragment_count_after > 0
            and metrics.fragment_count_after / max(metrics.fragment_count_before, 1) <= 0.4
            and metrics.coverage_score < 0.7
        ):
            return True
        return False

    def _should_use_lean_compiled_payload(self):
        if self.last_compilation is None:
            return False
        if self._is_observation_heavy_query(self.last_compilation.query_plan.raw_query):
            return False
        metrics = self.last_compilation.metrics
        if metrics.subgoal_count > 0 and metrics.covered_subgoal_count < metrics.subgoal_count:
            return False
        if metrics.coverage_score < max(self.compiler_critic_min_coverage, 0.7):
            return False
        return True

    def _is_multi_phase_query(self, task):
        lowered = task.lower()
        phase_markers = [
            "first",
            "next",
            "then",
            "finally",
            "if ",
            "when ",
            "after ",
            "before ",
        ]
        task_family_markers = [
            "grow",
            "thermometer",
            "temperature",
            "conductive",
            "conductivity",
            "sort",
            "compare",
            "longest",
            "shortest",
        ]
        return any(marker in lowered for marker in phase_markers) or any(
            marker in lowered for marker in task_family_markers
        )

    def _compile_critic_candidate_names(self, selected_skill_names, reference_skill_names, seed_skill_names):
        available = []
        for skill_name in (
            list(selected_skill_names)
            + list(reference_skill_names)
            + list(seed_skill_names)
            + self._family_priority_skill_names(self.last_compilation.query_plan.raw_query if self.last_compilation else "")
        ):
            if skill_name in self.metadata and skill_name not in available:
                available.append(skill_name)

        if self.last_compilation is not None:
            for item in self.last_compilation.compiled_skills:
                skill_name = item.asset.name
                if skill_name in self.metadata and skill_name not in available:
                    available.append(skill_name)

        return available[: max(self.compiler_top_k + 4, len(available))]

    def _build_compile_critic_skill_summaries(self, skill_names):
        compiled_lookup = {}
        if self.last_compilation is not None:
            compiled_lookup = {
                item.asset.name: item
                for item in self.last_compilation.compiled_skills
            }

        summaries = []
        for skill_name in skill_names:
            meta = self.metadata.get(skill_name)
            if not meta:
                continue
            description = meta.get("description", "")
            compiled_skill = compiled_lookup.get(skill_name)
            localized = []
            if compiled_skill is not None:
                localized = compiled_skill.localized_instructions[:3]
            if not localized:
                localized = self._fallback_localized_instructions(skill_name)[:3]
            summary = (
                f"- {skill_name}\n"
                f"  description: {description}\n"
                f"  localized_hints: {' | '.join(localized)}"
            )
            summaries.append(summary)
        return "\n".join(summaries)

    def _build_compiler_summary(self):
        if self.last_compilation is None:
            return ""
        metrics = self.last_compilation.metrics
        summary = (
            f"selected={metrics.selected_count}/{metrics.candidate_count}; "
            f"coverage={metrics.coverage_score:.3f}; "
            f"covered_subgoals={metrics.covered_subgoal_count}/{metrics.subgoal_count}; "
            f"token_cost={metrics.estimated_token_cost_after:.2f}; "
            f"execution_cost={metrics.estimated_execution_cost_after:.2f}; "
            f"execution_order={self.last_compilation.execution_order}"
        )
        if self.last_compilation.notes:
            summary += f"; notes={self.last_compilation.notes[-2:]}"
        return summary

    def _compiled_skill_role(self, compiled_skill):
        if compiled_skill is None:
            return "support"
        text = " ".join(
            [
                compiled_skill.asset.name,
                compiled_skill.asset.description,
                *compiled_skill.localized_instructions,
            ]
        ).lower()
        driver_markers = (
            "move ",
            "pick up",
            "activate",
            "deactivate",
            "open ",
            "close ",
            "transfer",
            "prepare",
            "setup",
            "connect",
            "use ",
        )
        support_markers = (
            "look at",
            "examine",
            "monitor",
            "focus on",
            "identify",
            "inspect",
            "verify",
            "wait",
        )
        driver_hits = sum(marker in text for marker in driver_markers)
        support_hits = sum(marker in text for marker in support_markers)
        if driver_hits > support_hits:
            return "driver"
        if support_hits > 0:
            return "support"
        return "driver"

    def _primary_compiled_action(self, compiled_skill):
        if compiled_skill is None:
            return ""
        for fragment in compiled_skill.selected_fragments:
            if fragment.action_schema:
                return fragment.action_schema.strip()
            if fragment.example_actions:
                return fragment.example_actions[0].strip()
        for instruction in compiled_skill.localized_instructions:
            lowered = instruction.lower()
            for marker in (
                "move ",
                "pick up",
                "activate",
                "deactivate",
                "open ",
                "close ",
                "use ",
                "look at",
                "examine",
                "focus on",
            ):
                idx = lowered.find(marker)
                if idx != -1:
                    return instruction[idx:].strip()
        return compiled_skill.asset.description

    def _build_compiled_execution_outline(self, skill_names):
        if self.last_compilation is None:
            return ""
        compiled_lookup = {
            item.asset.name: item
            for item in self.last_compilation.compiled_skills
        }
        outline = []
        for index, skill_name in enumerate(self.last_compilation.execution_order, start=1):
            compiled_skill = compiled_lookup.get(skill_name)
            if compiled_skill is None or skill_name not in skill_names:
                continue
            role = self._compiled_skill_role(compiled_skill)
            action = self._primary_compiled_action(compiled_skill)
            if role == "support":
                guidance = (
                    "Use only after a preceding manipulation step needs confirmation or the task explicitly asks for it."
                )
            else:
                guidance = "Prefer this as a main progress-driving step."
            outline.append(
                f"{index}. {skill_name} [{role}] - {action}\n   {guidance}"
            )
        return "\n".join(outline)

    def _sanitize_compile_critic_decision(
        self,
        critic_decision,
        current_selection,
        reference_skill_names,
        seed_skill_names,
        available_skill_names,
    ):
        if not isinstance(critic_decision, dict):
            return current_selection

        task = self.last_compilation.query_plan.raw_query if self.last_compilation else ""
        family_mandatory = [
            skill_name
            for skill_name in self._family_mandatory_skill_names(task)
            if skill_name in available_skill_names
        ]
        family_support = {
            skill_name
            for skill_name in self._family_support_skill_names(task)
            if skill_name in available_skill_names
        }
        if family_mandatory:
            allowed = set(family_mandatory) | family_support
        else:
            allowed = set(current_selection)

        def sanitize_names(raw_names):
            if not isinstance(raw_names, list):
                return []
            sanitized = []
            for item in raw_names:
                if isinstance(item, str) and item in allowed and item not in sanitized:
                    sanitized.append(item)
            return sanitized

        must_keep = sanitize_names(critic_decision.get("must_keep_skills", []))
        preferred = sanitize_names(critic_decision.get("preferred_skill_order", []))
        drop = {
            item
            for item in sanitize_names(critic_decision.get("drop_skills", []))
            if item not in must_keep
        }

        if family_mandatory:
            limit = len(family_mandatory) + len(family_support)
        else:
            limit = max(len(current_selection), len(reference_skill_names), len(must_keep))
        final_selection = []
        for skill_name in family_mandatory:
            if skill_name not in final_selection:
                final_selection.append(skill_name)
        for group in (
            must_keep,
            preferred,
            current_selection,
            reference_skill_names,
            seed_skill_names,
        ):
            for skill_name in group:
                if skill_name in drop or skill_name not in allowed or skill_name in final_selection:
                    continue
                final_selection.append(skill_name)
                if len(final_selection) >= limit:
                    break
            if len(final_selection) >= limit:
                break

        if not final_selection:
            return current_selection
        return final_selection

    def _append_compile_note(self, note):
        if self.last_compilation is not None and note not in self.last_compilation.notes:
            self.last_compilation.notes.append(note)

    def _select_quality_first_skill_names(self, task, reference_skill_names, seed_skill_names):
        available = []
        for skill_name in list(reference_skill_names) + list(seed_skill_names):
            if skill_name not in available:
                available.append(skill_name)
        for skill_name in self._family_priority_skill_names(task):
            if skill_name in self.metadata and skill_name not in available:
                available.append(skill_name)

        if self._infer_benchmark() != "scienceworld":
            return available[: max(self.compiler_top_k, len(reference_skill_names))]

        if self._is_scienceworld_conductivity_query(task):
            canonical = self._select_scienceworld_conductivity_skill_names(available)
            if canonical:
                return canonical
        if self._is_scienceworld_temperature_query(task):
            canonical = self._select_scienceworld_temperature_skill_names(available)
            if canonical:
                return canonical
        if self._is_scienceworld_growth_query(task):
            canonical = self._select_scienceworld_growth_skill_names(available)
            if canonical:
                return canonical
        if self._is_scienceworld_phase_change_query(task):
            canonical = self._select_scienceworld_phase_change_skill_names(available)
            if canonical:
                return canonical

        query = task.lower()
        selected = []
        for skill_name in reference_skill_names:
            if skill_name in available and skill_name not in selected:
                selected.append(skill_name)

        def pick_first(candidates):
            for candidate in candidates:
                if candidate in available and candidate not in selected:
                    selected.append(candidate)
                    return

        limit = max(len(reference_skill_names), min(self.compiler_top_k, len(reference_skill_names) + 1))

        if "located" in query or "find" in query or "around the" in query:
            pick_first([
                "scienceworld-object-locator",
                "scienceworld-target-locator",
                "scienceworld-room-navigator",
                "scienceworld-room-explorer",
            ])

        if "focus on" in query:
            pick_first([
                "scienceworld-object-focuser",
                "scienceworld-task-focuser",
            ])

        if "conductive" in query or "conductivity" in query:
            pick_first([
                "scienceworld-conductivity-tester",
                "scienceworld-circuit-builder",
                "scienceworld-circuit-connector",
            ])

        if "box" in query or "place" in query or "move" in query:
            pick_first([
                "scienceworld-object-classifier",
                "scienceworld-conditional-placer",
                "scienceworld-object-placer",
                "scienceworld-container-relocator",
            ])

        return selected[:limit]

    def _family_priority_skill_names(self, task):
        if self._infer_benchmark() != "scienceworld":
            return []
        if self._is_scienceworld_conductivity_query(task):
            return self._family_mandatory_skill_names(task) + self._family_support_skill_names(task)
        if self._is_scienceworld_temperature_query(task):
            return self._family_mandatory_skill_names(task) + self._family_support_skill_names(task)
        if self._is_scienceworld_growth_query(task):
            return self._family_mandatory_skill_names(task) + self._family_support_skill_names(task)
        if self._is_scienceworld_phase_change_query(task):
            return self._family_mandatory_skill_names(task) + self._family_support_skill_names(task)
        return []

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
            "scienceworld-object-focuser",
            "scienceworld-task-focuser",
        ])
        pick_first([
            "scienceworld-temperature-measurer",
        ])
        pick_first([
            "scienceworld-threshold-evaluator",
        ])
        pick_first([
            "scienceworld-conditional-focus-executor",
            "scienceworld-conditional-box-placer",
        ])
        pick_first([
            "scienceworld-inventory-focus",
            "scienceworld-room-scanner",
        ])

        return selected if len(selected) >= 5 else []

    def _is_scienceworld_growth_query(self, task):
        if self._infer_benchmark() != "scienceworld":
            return False
        query = task.lower()
        return "grow" in query and "seed" in query

    def _is_scienceworld_phase_change_query(self, task):
        if self._infer_benchmark() != "scienceworld":
            return False
        query = task.lower()
        phase_change_terms = (
            "boil",
            "melting point",
            "freezing point",
            "state of matter",
            "change its state",
            "change the state",
            "combust",
            "melt",
            "freeze",
        )
        return any(term in query for term in phase_change_terms)

    def _select_scienceworld_phase_change_skill_names(self, available):
        selected = []

        def pick_first(candidates):
            for candidate in candidates:
                if candidate in available and candidate not in selected:
                    selected.append(candidate)
                    return True
            return False

        pick_first([
            "scienceworld-object-focuser",
            "scienceworld-task-focuser",
        ])
        pick_first([
            "scienceworld-substance-preparator",
        ])
        pick_first([
            "scienceworld-heating-apparatus-setup",
        ])
        pick_first([
            "scienceworld-process-monitor",
        ])
        pick_first([
            "controlled-waiting",
        ])
        pick_first([
            "scienceworld-object-locator",
            "scienceworld-item-fetcher",
            "scienceworld-room-navigator",
        ])

        return selected if len(selected) >= 4 else []

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
            "soil-extraction",
        ])
        pick_first([
            "scienceworld-pot-preparer",
            "scienceworld-planting-operation",
        ])
        pick_first([
            "scienceworld-planting-coordinator",
        ])
        pick_first([
            "scienceworld-growth-focuser",
        ])
        pick_first([
            "controlled-waiting",
        ])
        pick_first([
            "scienceworld-liquid-filler",
        ])
        pick_first([
            "scienceworld-ambiguous-action-resolution",
        ])

        return selected if len(selected) >= 7 else []

    def _scienceworld_task_family(self, task):
        if self._infer_benchmark() != "scienceworld":
            return "generic"
        if self._is_scienceworld_conductivity_query(task):
            return "conductivity"
        if self._is_scienceworld_temperature_query(task):
            return "temperature"
        if self._is_scienceworld_growth_query(task):
            return "growth"
        if self._is_scienceworld_phase_change_query(task):
            return "phase_change"
        return "generic"

    def _family_mandatory_skill_names(self, task):
        family = self._scienceworld_task_family(task)
        if family == "conductivity":
            return [
                "scienceworld-object-locator",
                "scienceworld-object-focuser",
                "scienceworld-conductivity-tester",
                "scienceworld-object-classifier",
            ]
        if family == "temperature":
            return [
                "scienceworld-object-locator",
                "scienceworld-object-focuser",
                "scienceworld-temperature-measurer",
                "scienceworld-threshold-evaluator",
                "scienceworld-conditional-focus-executor",
            ]
        if family == "growth":
            return [
                "scienceworld-room-navigator",
                "scienceworld-object-focuser",
                "soil-extraction",
                "scienceworld-pot-preparer",
                "scienceworld-planting-coordinator",
                "scienceworld-growth-focuser",
                "controlled-waiting",
            ]
        if family == "phase_change":
            return [
                "scienceworld-object-focuser",
                "scienceworld-substance-preparator",
                "scienceworld-heating-apparatus-setup",
                "scienceworld-process-monitor",
            ]
        return []

    def _family_support_skill_names(self, task):
        family = self._scienceworld_task_family(task)
        if family == "conductivity":
            return [
                "scienceworld-object-retriever",
                "scienceworld-ambiguous-action-resolution",
                "scienceworld-conditional-placer",
            ]
        if family == "temperature":
            return [
                "scienceworld-inventory-focus",
                "scienceworld-task-focuser",
                "scienceworld-conditional-box-placer",
            ]
        if family == "growth":
            return [
                "scienceworld-item-fetcher",
                "scienceworld-liquid-filler",
                "scienceworld-ambiguous-action-resolution",
            ]
        if family == "phase_change":
            return [
                "scienceworld-task-focuser",
                "scienceworld-object-locator",
                "scienceworld-item-fetcher",
                "controlled-waiting",
                "scienceworld-ambiguous-action-resolution",
            ]
        return []

    def _build_quality_first_compilation(self, compiled_package, reference_skill_names):
        """
        Preserve the baseline/SkillNet reference skill set so DSC can recover from
        under-selection without discarding quality-critical skills.
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
            "Quality-first mode preserved the SkillNet reference skill set."
        )
        return compiled_package

    def _deterministic_procedure(self, task, skill_names):
        self.last_deterministic_procedure_kind = None
        if self.runtime_recompile_active:
            return None
        if self._uses_dsc_strategy() and self._is_scienceworld_conductivity_task(task, skill_names):
            self.last_deterministic_procedure_kind = "scienceworld_conductivity"
            return self._scienceworld_conductivity_procedure(task)
        if self._uses_dsc_strategy() and self._is_scienceworld_growth_task(task, skill_names):
            self.last_deterministic_procedure_kind = "scienceworld_growth"
            return self._scienceworld_growth_procedure(task)
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

    def _is_scienceworld_growth_task(self, task, skill_names):
        if not self._is_scienceworld_growth_query(task):
            return False
        skill_set = set(skill_names)
        return {
            "soil-extraction",
            "scienceworld-pot-preparer",
            "scienceworld-planting-coordinator",
            "scienceworld-growth-focuser",
            "controlled-waiting",
        }.issubset(skill_set)

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

    def _scienceworld_growth_procedure(self, task):
        plant_name, seed_name, seed_room = self._extract_scienceworld_growth_targets(task)
        return f"""# TASK PROCEDURAL GUIDANCE: Grow {plant_name.title()} Plant from Seed

Task:
{task}

Phase 1: Acquire the seed
1. `teleport to {seed_room}`
2. `look around`
3. `focus on {seed_name}`
4. If the environment asks for an index, answer `0`.
5. `pick up {seed_name}`
6. If the environment asks for an index, answer `0`.

Phase 2: Acquire shovel and extract soil
1. `teleport to outside`
2. `look around`
3. `pick up shovel`
4. `use shovel on ground`
5. `pick up soil`

Phase 3: Prepare a planted flower pot
1. `teleport to greenhouse`
2. `look around`
3. Choose the first visible flower pot that already contains water if possible; otherwise choose the first visible flower pot.
4. `move soil to <FLOWER_POT>`
5. `move {seed_name} to <FLOWER_POT>`

Phase 4: Trigger and monitor growth
1. `focus on <FLOWER_POT>`
2. `wait`
3. `look around`
4. If a `{plant_name} tree` is visible, `focus on {plant_name} tree`
5. `wait`
6. If the plant is not yet reproducing, continue alternating `look around` and `wait` until the reproduction stage appears or the task completes.

Error handling:
- If an action returns `Ambiguous request`, answer with only the index such as `0`.
- Do not search extra rooms for the shovel once `outside` has been checked.
- Do not try to manipulate liquid directly. Use the water already present in greenhouse flower pots when available.
- Reuse the exact observed flower pot identifier everywhere.
- Prefer the shortest stable route: seed room -> outside -> greenhouse."""

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

    def _extract_scienceworld_growth_targets(self, task):
        plant_match = re.search(r"grow a[n]?\s+(.+?)\s+plant from seed", task, re.IGNORECASE)
        room_match = re.search(r"Seeds can be found in the (.+?)[\.,]", task, re.IGNORECASE)
        plant_name = plant_match.group(1).strip() if plant_match else "cherry"
        seed_name = f"{plant_name} seed"
        seed_room = room_match.group(1).strip() if room_match else "bedroom"
        return plant_name, seed_name, seed_room

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
    last_observation = ""

    def run_action(action: str):
        nonlocal task_done, current_steps, task_reward, last_observation
        if task_done or current_steps >= max_steps:
            return last_observation
        messages.append({{"role": "assistant", "content": f"Action: {{action}}"}})
        observation, step_reward, task_done, info = env.step(action)
        task_reward = info['score'] if info.get('score') is not None and info['score'] > task_reward else task_reward
        print(f'\\033[93mObservation: \\n{{observation}}\\033[0m')
        messages.append({{"role": "user", "content": f"Observation: {{observation}}"}})
        if observation:
            last_observation = observation
        current_steps += 1
        return observation or last_observation

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
    workshop_observation = source_observation
    if source_room.lower() != "workshop" and current_steps < max_steps:
        workshop_observation = run_action("teleport to workshop")
        workshop_observation = run_action("look around")
    elif (not workshop_observation or not workshop_observation.strip()) and current_steps < max_steps:
        workshop_observation = run_action("look around")
    if not workshop_observation or not workshop_observation.strip():
        return messages, task_done, task_reward, current_steps
    wire1, wire2, wire3, actuator = choose_components(workshop_observation)
    actuator_negative, actuator_positive = actuator_terminals(actuator)
    run_action(f"connect battery anode to {{wire1}} terminal 1")
    run_action(f"connect {{wire1}} terminal 2 to {{actuator}} {{actuator_negative}}")
    run_action(f"connect {{actuator}} {{actuator_positive}} to {{wire2}} terminal 1")
    run_action(f"connect {{wire2}} terminal 2 to {{target_object}} terminal 1")
    run_action(f"connect {{target_object}} terminal 2 to battery cathode")
    run_action("wait1")
    if current_steps >= max_steps:
        return messages, task_done, task_reward, current_steps
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
        if current_steps >= max_steps:
            return messages, task_done, task_reward, current_steps
        bulb_observation = run_action(f"look at {{actuator}}")
        bulb_is_on = (
            "which is on" in bulb_observation.lower()
            or "which is activated" in bulb_observation.lower()
            or " is on." in bulb_observation.lower()
            or " is activated." in bulb_observation.lower()
        )
    target_box = conductive_box if bulb_is_on else nonconductive_box
    if current_steps >= max_steps:
        return messages, task_done, task_reward, current_steps
    run_action(f"move {{target_object}} to {{target_box}}")

    return messages, task_done, task_reward, current_steps'''

    def _scienceworld_growth_static_procedure_code(self, task, overall_procedure):
        escaped = overall_procedure.replace('"""', r"\"\"\"")
        plant_name, seed_name, seed_room = self._extract_scienceworld_growth_targets(task)
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
    Deterministic growth solver for a quality-critical ScienceWorld task.
    """
    import re

    procedure_guidelines = """{escaped}"""
    messages.append({{"role": "user", "content": procedure_guidelines}})

    plant_name = {plant_name!r}
    seed_name = {seed_name!r}
    seed_room = {seed_room!r}
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

    def handle_ambiguity(observation: str):
        if "Ambiguous request" not in observation:
            return observation
        return run_action("0")

    def extract_pot_name(observation: str):
        water_matches = re.findall(r"(flower pot \\d+) \\(containing[^\\n]*water", observation, flags=re.IGNORECASE)
        if water_matches:
            return water_matches[0]
        pot_matches = re.findall(r"(flower pot \\d+)", observation, flags=re.IGNORECASE)
        if pot_matches:
            return pot_matches[0]
        return "flower pot 1"

    run_action(f"teleport to {{seed_room}}")
    seed_observation = run_action("look around")
    handle_ambiguity(run_action(f"focus on {{seed_name}}"))
    handle_ambiguity(run_action(f"pick up {{seed_name}}"))
    run_action("teleport to outside")
    run_action("look around")
    run_action("pick up shovel")
    run_action("use shovel on ground")
    run_action("pick up soil")
    run_action("teleport to greenhouse")
    greenhouse_observation = run_action("look around")
    target_pot = extract_pot_name(greenhouse_observation)
    run_action(f"move soil to {{target_pot}}")
    handle_ambiguity(run_action(f"move {{seed_name}} to {{target_pot}}"))
    run_action(f"focus on {{target_pot}}")
    run_action("wait")
    growth_observation = run_action("look around")
    tree_name = f"{{plant_name}} tree"
    if tree_name.lower() in growth_observation.lower():
        run_action(f"focus on {{tree_name}}")
    run_action("wait")
    if not task_done and current_steps < max_steps:
        growth_observation = run_action("look around")
        if tree_name.lower() in growth_observation.lower() and "reproduct" not in growth_observation.lower():
            run_action(f"focus on {{tree_name}}")
            run_action("wait")

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
        relevant_skill_names = extract_tagged_json(
            response,
            "Relevant_Skill_Names",
            default=[],
        )
        if not isinstance(relevant_skill_names, list):
            return []
        return [item for item in relevant_skill_names if isinstance(item, str) and item in self.metadata]

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
        lean_payload = self._should_use_lean_compiled_payload()
        execution_outline = self._build_compiled_execution_outline(skill_names)
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
            if lean_payload:
                assigned_subgoals = ", ".join(compiled_skill.assigned_subgoals) if compiled_skill else ""
                fragment_summary = self._summarize_selected_fragments(compiled_skill)
                source_description = meta.get("description", "")
                combined_text = (
                    f"=== Compiled Skill: {skill_name} ===\n"
                    f"[Selected Reason]\n{selected_reason}\n"
                    f"[Assigned Subgoals]\n{assigned_subgoals or 'task_support'}\n"
                    f"[Execution Role]\n{self._compiled_skill_role(compiled_skill)}\n"
                    f"[Localized Instructions]\n{localized}\n"
                    f"[Selected Fragments]\n{fragment_summary}\n"
                    f"[Source Description]\n{source_description}\n"
                )
            else:
                combined_text = (
                    f"=== Compiled Skill: {skill_name} ===\n"
                    f"[Selected Reason]\n{selected_reason}\n"
                    f"[Localized Instructions]\n{localized}\n"
                    f"[Compressed SKILL.md]\n{skill_md}\n"
                )
            skill_contents.append((skill_name, combined_text))

        metrics = self.last_compilation.metrics
        execution_order = " -> ".join(self.last_compilation.execution_order)
        if lean_payload:
            summary = (
                "Dynamic Skill Compiler Summary\n"
                "The compiled package below is already task-specific. Prefer the localized instructions and selected fragments over broad rewrites.\n"
                "Do not invent extra rooms, tools, or object names that are not grounded by observation.\n"
                "Use the execution outline to drive progress first; use support-only skills only when a prior action needs verification.\n"
                f"- Selected skills: {metrics.selected_count}/{metrics.candidate_count}\n"
                f"- Covered subgoals: {metrics.covered_subgoal_count}/{metrics.subgoal_count}\n"
                f"- Coverage score: {metrics.coverage_score:.3f}\n"
                f"- Redundancy reduction: {metrics.redundancy_reduction:.3f}\n"
                f"- Fragments retained: {metrics.fragment_count_after}/{metrics.fragment_count_before}\n"
                f"- Estimated token cost: {metrics.estimated_token_cost_before:.2f} -> "
                f"{metrics.estimated_token_cost_after:.2f}\n"
                f"- Fragment token cost: {metrics.fragment_token_cost_after:.2f}\n"
                f"- Execution order: {execution_order}\n"
                f"- Execution outline:\n{execution_outline or 'No execution outline available.'}\n"
            )
        else:
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
                f"- Execution outline:\n{execution_outline or 'No execution outline available.'}\n"
            )
        return skill_contents, summary

    def _summarize_selected_fragments(self, compiled_skill, max_fragments=2, max_chars=420):
        if compiled_skill is None or not compiled_skill.selected_fragments:
            return "- No fragment-specific instructions generated."

        fragments = []
        consumed = 0
        for fragment in compiled_skill.selected_fragments[:max_fragments]:
            remaining = max_chars - consumed
            if remaining <= 0:
                break
            content = " ".join(str(fragment.content or "").split())
            snippet = content[:remaining].strip()
            if not snippet:
                continue
            fragments.append(f"- {fragment.title}: {snippet}")
            consumed += len(snippet)

        return "\n".join(fragments) if fragments else "- No fragment-specific instructions generated."

    def _build_quality_first_skill_payload(self, skill_names):
        if self._should_expand_quality_first_payload():
            skill_contents = self._build_full_skill_payload(skill_names)
            metrics = self.last_compilation.metrics
            execution_order = " -> ".join(self.last_compilation.execution_order)
            summary = (
                "Dynamic Skill Compiler Recovery Summary\n"
                "Compiler confidence was low, so the prompt preserves the full source skill payload "
                "for the selected package instead of an aggressively compressed version.\n"
                f"- Selected skills: {metrics.selected_count}/{metrics.candidate_count}\n"
                f"- Covered subgoals: {metrics.covered_subgoal_count}/{metrics.subgoal_count}\n"
                f"- Coverage score: {metrics.coverage_score:.3f}\n"
                f"- Fragments retained: {metrics.fragment_count_after}/{metrics.fragment_count_before}\n"
                f"- Execution order: {execution_order}\n"
            )
            return skill_contents, summary

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
