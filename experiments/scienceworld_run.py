import os
import re
import concurrent.futures
from tqdm import tqdm
import json
import sys
import argparse
import time
import importlib.util
from pathlib import Path
from typing import Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
scienceworld_candidates = [
    os.path.join(current_dir, "ScienceWorld"),
    os.path.join(current_dir, "scienceworld"),
]
scienceworld_path = next((path for path in scienceworld_candidates if os.path.isdir(path)), scienceworld_candidates[0])
if scienceworld_path not in sys.path:
    sys.path.append(scienceworld_path)

from scienceworld import ScienceWorldEnv

from src.scienceworld.prompts.system_prompt import scienceworld_system_prompt
from src.runtime_recompile import execute_compiled_procedure
from src.skill import SkillModule
from src.utils import chat_completion


def llm(prompt, model="YOUR_MODEL_NAME"):
    print(f'Calling LLM with model: {model}')
    ensure_task_not_timed_out()
    # Normalize messages format
    if isinstance(prompt, list):
        messages = prompt
    elif isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        raise ValueError(f'prompt must be a list or a string, but got {type(prompt)}')
    response = chat_completion(messages=messages, model=model)
    # Extract content
    content = response.choices[0].message.content
    if content is not None:
        return content
    return "Output Error"



# ==========================================
# Constants & Configuration
# ==========================================

class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

# ==========================================
# Helper Functions
# ==========================================

def _normalize_action_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _strip_leading_articles(text: str) -> str:
    return re.sub(r"^(?:the|a|an)\s+", "", text.strip(), flags=re.IGNORECASE)


def _strip_instance_details(text: str) -> str:
    return re.sub(r"\([^)]*\)", "", text).strip()


def _extract_ambiguous_options(observation: str):
    text = observation.strip()
    if text.startswith("Observation:"):
        text = text[len("Observation:"):].strip()
    if "Ambiguous request:" not in text or "Please enter the number" not in text:
        return []
    compact = re.sub(r"\s+", " ", text)
    return [
        (match.group(1), match.group(2).strip())
        for match in re.finditer(r"(\d+):\s*(.*?)(?=\s+\d+:|$)", compact)
    ]


def _resolve_ambiguous_action(action: str, last_observation: str) -> str:
    options = _extract_ambiguous_options(last_observation)
    if not options:
        return action

    stripped = action.strip()
    if re.fullmatch(r"\d+", stripped):
        return stripped

    action_norm = _normalize_action_text(stripped)
    action_core = _normalize_action_text(_strip_instance_details(stripped))
    for idx, option in options:
        option_norm = _normalize_action_text(option)
        option_core = _normalize_action_text(_strip_instance_details(option))
        if action_norm and (action_norm == option_norm or action_norm in option_norm):
            return idx
        if action_core and action_core == option_core:
            return idx

    unique_cores = {
        _normalize_action_text(_strip_instance_details(option))
        for _, option in options
    }
    if len(unique_cores) == 1:
        return options[0][0]
    return action


def _normalize_indexed_object_guess(action: str, last_observation: str) -> str:
    text = last_observation.strip()
    if text.startswith("Observation:"):
        text = text[len("Observation:"):].strip()
    if not text:
        return action

    indexed_action = re.match(r"^(focus on|pick up|examine|open)\s+(.+?)\s+(\d+)$", action, re.IGNORECASE)
    if not indexed_action:
        return action

    verb, object_name, _ = indexed_action.groups()
    lowered_action = _normalize_action_text(action)
    lowered_observation = _normalize_action_text(text)
    if lowered_action in lowered_observation:
        return action

    base_action = f"{verb} {object_name}".strip()
    if _normalize_action_text(object_name) in lowered_observation:
        return base_action
    return action


def _normalize_scienceworld_action(action: str) -> str:
    normalized = re.sub(r"\s+", " ", action or "").strip().strip(".")
    if not normalized:
        return ""

    def clean_object(text: str, drop_suffix: Optional[str] = None) -> str:
        value = _strip_leading_articles(text)
        if drop_suffix:
            value = re.sub(rf"\s+{drop_suffix}$", "", value, flags=re.IGNORECASE)
        return value.strip()

    exact_aliases = {
        "look": "look around",
        "wait 1": "wait1",
        "wait one step": "wait1",
    }
    lowered = _normalize_action_text(normalized)
    if lowered in exact_aliases:
        return exact_aliases[lowered]

    alias_patterns = [
        (r"^(?:teleport(?: to)?)\s+(.+)$", lambda match: f"teleport to {clean_object(match.group(1))}"),
        (r"^(?:focus)\s+(.+)$", lambda match: f"focus on {clean_object(match.group(1))}"),
        (r"^(?:inspect|check)\s+(.+)$", lambda match: f"examine {clean_object(match.group(1))}"),
        (r"^(?:turn on|switch on|power on|start)\s+(.+)$", lambda match: f"activate {clean_object(match.group(1))}"),
        (r"^(?:turn off|switch off|power off|stop)\s+(.+)$", lambda match: f"deactivate {clean_object(match.group(1))}"),
        (r"^(open|close)\s+(.+?)\s+(?:door|lid)$", lambda match: f"{match.group(1).lower()} {clean_object(match.group(2))}"),
        (r"^(?:take|grab|get)\s+(.+?)(?:\s+from\s+.+)?$", lambda match: f"pick up {clean_object(match.group(1))}"),
        (r"^(?:pick up)\s+(.+?)(?:\s+from\s+.+)?$", lambda match: f"pick up {clean_object(match.group(1))}"),
        (r"^(?:pick)\s+(.+?)\s+up(?:\s+from\s+.+)?$", lambda match: f"pick up {clean_object(match.group(1))}"),
        (r"^(?:put|place|move)\s+(.+?)\s+(?:in|into|on|onto|to)\s+(.+)$", lambda match: f"move {clean_object(match.group(1))} to {clean_object(match.group(2))}"),
        (r"^(?:connect)\s+(.+?)\s+(?:with|and)\s+(.+)$", lambda match: f"connect {clean_object(match.group(1))} to {clean_object(match.group(2))}"),
    ]

    for pattern, replacer in alias_patterns:
        match = re.match(pattern, normalized, re.IGNORECASE)
        if match:
            return replacer(match)

    direct_patterns = [
        (r"^(open|close|activate|deactivate|disconnect|examine|look at|read|pick up|focus on|mix)\s+(.+)$", lambda match: f"{match.group(1).lower()} {clean_object(match.group(2))}"),
        (r"^(move)\s+(.+?)\s+to\s+(.+)$", lambda match: f"move {clean_object(match.group(2))} to {clean_object(match.group(3))}"),
        (r"^(connect)\s+(.+?)\s+to\s+(.+)$", lambda match: f"connect {clean_object(match.group(2))} to {clean_object(match.group(3))}"),
        (r"^(pour)\s+(.+?)\s+into\s+(.+)$", lambda match: f"pour {clean_object(match.group(2))} into {clean_object(match.group(3))}"),
        (r"^(use)\s+(.+?)\s+on\s+(.+)$", lambda match: f"use {clean_object(match.group(2))} on {clean_object(match.group(3))}"),
        (r"^(use)\s+(.+)$", lambda match: f"use {clean_object(match.group(2))}"),
        (r"^(teleport to)\s+(.+)$", lambda match: f"teleport to {clean_object(match.group(2))}"),
        (r"^(wait1?)$", lambda match: match.group(1).lower()),
    ]

    for pattern, replacer in direct_patterns:
        match = re.match(pattern, normalized, re.IGNORECASE)
        if match:
            return replacer(match)

    return normalized


def parse_action(response: str, last_observation: str = "") -> str:
    """
    Extract an executable action and map ambiguous-choice follow-ups back to the required index.
    """
    pattern = re.compile(r"Action:\s*(.+)", re.IGNORECASE)
    match = pattern.search(response)
    if match:
        action = match.group(1).strip().strip('"\'*`')
    else:
        stripped = response.strip().strip('"\'*`')
        action = stripped if re.fullmatch(r"\d+", stripped) else ""
    action = _normalize_scienceworld_action(action)
    action = _normalize_indexed_object_guess(action, last_observation)
    return _resolve_ambiguous_action(action, last_observation)


def atomic_write_json(path: str, payload: dict):
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path_obj.with_suffix(path_obj.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
    os.replace(tmp_path, path_obj)


def update_task_status(output_path, idx, task_name, var_idx, stage, extra=None):
    payload = {
        "idx": idx,
        "task_name": task_name,
        "variation_idx": var_idx,
        "stage": stage,
    }
    if extra:
        payload.update(extra)
    atomic_write_json(
        os.path.join(output_path, f"status_{idx}.json"),
        payload,
    )


class TaskTimedOutError(TimeoutError):
    pass


_TASK_DEADLINE = None


def set_task_deadline(timeout_seconds):
    global _TASK_DEADLINE
    if timeout_seconds and timeout_seconds > 0:
        _TASK_DEADLINE = time.monotonic() + float(timeout_seconds)
    else:
        _TASK_DEADLINE = None


def clear_task_deadline():
    global _TASK_DEADLINE
    _TASK_DEADLINE = None


def ensure_task_not_timed_out():
    if _TASK_DEADLINE is not None and time.monotonic() > _TASK_DEADLINE:
        raise TaskTimedOutError("Task timed out before completing.")

def run_standard_procedure(env, llm, model, messages, max_steps):
    """
    Executes the standard interaction loop when no skills are used.
    """
    task_done = False
    task_reward = 0
    current_steps = 0

    while not task_done and current_steps < max_steps:
        ensure_task_not_timed_out()
        current_steps += 1
        
        # 1. Get Agent Response
        try:
            response = llm(messages, model)
            print(f'{Colors.GREEN}Agent response: \n{response}{Colors.RESET}')
        except Exception as e:
            print(f'{Colors.RED}Error in LLM call: {e}{Colors.RESET}')
            break

        messages.append({"role": "assistant", "content": response})

        # 2. Parse and Execute Action
        last_observation = messages[-1]["content"] if messages and messages[-1]["role"] == "user" else ""
        action = parse_action(response, last_observation)
        ensure_task_not_timed_out()
        observation, step_reward, task_done, info = env.step(action)

        task_reward = info['score'] if info['score'] is not None and info['score'] > task_reward else task_reward

        print(f'{Colors.YELLOW}Observation: \n{observation}{Colors.RESET}')

        # 3. Update History
        messages.append({"role": "user", "content": f"Observation: {observation}"})

        if task_done:
            print(f'{Colors.GREEN}Whole Task completed! Reward: {task_reward}{Colors.RESET}')

    return messages, task_done, task_reward, current_steps

# ==========================================
# Core Execution Logic
# ==========================================

def scienceworld_run_single(env, task_name, var_idx, args, Skill_Module=None, progress_callback=None):
    """
    Execute a single task variation.
    """
    # 1. Reset Environment & Get Initial Observation
    # ScienceWorld reset returns (observation, info)
    ensure_task_not_timed_out()
    obs, info = env.reset()
    
    # Construct Task Description
    query = env.get_task_description()
    
    print(f'{Colors.RED}Processing: {task_name} (Var: {var_idx}){Colors.RESET}')
    print(f"Query: {query}")
    if progress_callback:
        progress_callback("task_loaded", {"query": query})

    # 2. Initialize Prompt / Messages
    messages = [{"role": "system", "content": scienceworld_system_prompt}]
    messages.append({"role": "user", "content": query})

    # 3. Execution Strategy (Skill vs Standard)
    task_done = False
    task_reward = 0
    steps = 0
    relevant_skill_names = []
    overall_procedure = ""
    overall_procedure_code = ""


    use_skill = False
    if Skill_Module is not None:
        Skill_Module.reset_runtime_recompile_state()
        if progress_callback:
            progress_callback("retrieving_skills")
        ensure_task_not_timed_out()
        relevant_skill_names = Skill_Module.retrieve_relevant_skills(query)
        if relevant_skill_names:
            use_skill = True
            print(f'{Colors.BLUE}Retrieved relevant skills: {relevant_skill_names}{Colors.RESET}')
        if progress_callback:
            progress_callback("skills_retrieved", {"relevant_skill_names": relevant_skill_names})
    
    if use_skill:
        ensure_task_not_timed_out()
        def step_adapter(action, result):
            observation, step_reward, task_done_flag, info = result
            reward = info.get("score") if isinstance(info, dict) else None
            if reward is None:
                reward = step_reward
            return {
                "action": action,
                "observation": observation,
                "task_done": task_done_flag,
                "task_reward": reward,
            }

        def invoke(func, runtime_env, runtime_messages, remaining_steps):
            ensure_task_not_timed_out()
            return func(
                runtime_env,
                llm,
                args.model,
                parse_action,
                runtime_messages,
                remaining_steps,
            )

        execution = execute_compiled_procedure(
            env=env,
            llm=llm,
            model=args.model,
            task_prompt=query,
            messages=messages,
            max_steps=args.max_steps,
            skill_module=Skill_Module,
            selected_skill_names=relevant_skill_names,
            step_adapter=step_adapter,
            invoke=invoke,
            progress_callback=progress_callback,
        )
        messages = execution["messages"]
        task_done = execution["task_done"]
        task_reward = execution["task_reward"]
        steps = execution["steps"]
        relevant_skill_names = execution["skill_names"]
        overall_procedure = execution["overall_procedure"]
        overall_procedure_code = execution["overall_procedure_code"]
        if task_done:
            print(f'{Colors.GREEN} Task completed! Reward: {task_reward}{Colors.RESET}')
    else:
        if progress_callback:
            progress_callback("running_standard_procedure")
        messages, task_done, task_reward, steps = run_standard_procedure(
            env, llm, args.model, messages, args.max_steps
        )

    if progress_callback:
        progress_callback(
            "completed",
            {
                "task_done": task_done,
                "reward": task_reward,
                "steps": steps,
                "relevant_skill_names": relevant_skill_names,
            },
        )

    return {
        "query": query,
        "name": task_name,
        "variation_idx": var_idx,
        "task_done": task_done,
        "reward": task_reward,
        "steps": steps,
        "messages": messages,
        "relevant_skill_names": relevant_skill_names,
        "runtime_recompile_count": (
            Skill_Module.runtime_recompile_count if Skill_Module else 0
        ),
        "runtime_recompile_events": (
            Skill_Module.runtime_recompile_events if Skill_Module else []
        ),
        "skill_strategy": getattr(Skill_Module, "selection_strategy", "none") if Skill_Module else "none",
        "compiler_metrics": (
            Skill_Module.last_compilation.metrics.__dict__
            if Skill_Module and Skill_Module.last_compilation is not None
            else None
        ),
        "compiler_pass_traces": (
            [
                {
                    "pass_name": trace.pass_name,
                    "before_selected": trace.before_selected,
                    "after_selected": trace.after_selected,
                    "added": trace.added,
                    "removed": trace.removed,
                    "dropped_delta": trace.dropped_delta,
                }
                for trace in Skill_Module.last_compilation.pass_traces
            ]
            if Skill_Module and Skill_Module.last_compilation is not None
            else None
        ),
        "payload_strategy": (
            Skill_Module.last_payload_strategy
            if Skill_Module is not None
            else "full"
        ),
        "compiler_notes": (
            list(Skill_Module.last_compilation.notes)
            if Skill_Module and Skill_Module.last_compilation is not None
            else []
        ),
        "seed_skill_names": (
            list(Skill_Module.last_seed_skill_names)
            if Skill_Module is not None
            else []
        ),
        "quality_reference_skill_names": (
            list(Skill_Module.last_quality_reference_skill_names)
            if Skill_Module is not None
            else []
        ),
    }

# ==========================================
# Worker Function
# ==========================================

def eval_single_variation(idx, indices, args, output_path):
    """
    Worker function: Instantiates its own ScienceWorldEnv.
    task_info: tuple (task_name, variation_index)
    """
    task_name, var_idx = indices[idx]
    env = None
    save_file = f'{output_path}/idx_{idx}.json'
    task_state = {
        "stage": "starting",
        "query": "",
        "relevant_skill_names": [],
    }

    def progress_callback(stage, extra=None):
        task_state["stage"] = stage
        if extra:
            if "query" in extra:
                task_state["query"] = extra["query"]
            if "relevant_skill_names" in extra:
                task_state["relevant_skill_names"] = extra["relevant_skill_names"]
        update_task_status(output_path, idx, task_name, var_idx, stage, extra)

    try:
        clear_task_deadline()
        progress_callback("starting")
        # 1. Independent Environment Initialization
        env = ScienceWorldEnv(taskName=None, serverPath=None, envStepLimit=args.max_steps)
        
        # 2. Load Specific Task & Variation
        env.load(task_name, var_idx, simplificationStr="easy")
        progress_callback("environment_loaded")

        # 3. Initialize SkillModule (if enabled)
        Skill_Module = None
        if args.use_skill:
            skill_config = {
                "skills_dir": "src/skills/scienceworld",
                "overall_procedure_examples_path": "src/scienceworld/scienceworld_overall_procedure_examples.txt",
                "procedure_code_template_path": "src/scienceworld/scienceworld_procedure_code_template.py",
                "model": args.model,
                "selection_strategy": args.skill_strategy,
                "compiler_min_relevance": args.compiler_min_relevance,
                "compiler_preserve_top_k": args.compiler_preserve_top_k,
                "compiler_similar_prune_margin": args.compiler_similar_prune_margin,
                "compiler_keep_parent_if_better_by": args.compiler_keep_parent_if_better_by,
                "compiler_coverage_weight": args.compiler_coverage_weight,
                "compiler_quality_weight": args.compiler_quality_weight,
                "compiler_cost_weight": args.compiler_cost_weight,
                "compiler_latency_weight": args.compiler_latency_weight,
                "compiler_adaptive_workflow_bias": not args.compiler_disable_workflow_bias,
                "compiler_graph_passes": args.compiler_graph_passes,
                "compiler_critic_enabled": args.compiler_critic_enabled,
                "compiler_critic_model": args.compiler_critic_model,
                "compiler_critic_min_coverage": args.compiler_critic_min_coverage,
                "compiler_critic_force": args.compiler_critic_force,
                "runtime_recompile_enabled": args.runtime_recompile,
                "runtime_recompile_max_rounds": args.runtime_recompile_max_rounds,
                "runtime_recompile_min_interval_steps": args.runtime_recompile_min_interval_steps,
                "runtime_recompile_stagnation_threshold": args.runtime_recompile_stagnation_threshold,
                "runtime_recompile_min_remaining_steps": args.runtime_recompile_min_remaining_steps,
            }
            Skill_Module = SkillModule(**skill_config)
            progress_callback("skill_module_ready")

        set_task_deadline(args.task_timeout)

        # 4. Run Task
        result = scienceworld_run_single(
            env,
            task_name,
            var_idx,
            args,
            Skill_Module,
            progress_callback=progress_callback,
        )

        # 5. Save Results (Atomic write per file)
        atomic_write_json(save_file, result)
            
        return result

    except Exception as e:
        print(f"Error in task {task_name} var {var_idx}: {e}")
        failure_result = {
            "query": task_state["query"],
            "name": task_name,
            "variation_idx": var_idx,
            "task_done": False,
            "reward": 0,
            "steps": 0,
            "messages": [],
            "relevant_skill_names": task_state["relevant_skill_names"],
            "skill_strategy": args.skill_strategy if args.use_skill else "none",
            "compiler_metrics": None,
            "payload_strategy": (
                Skill_Module.last_payload_strategy
                if args.use_skill and Skill_Module is not None
                else "full"
            ),
            "compiler_notes": (
                list(Skill_Module.last_retrieval_warnings)
                if args.use_skill and Skill_Module is not None
                else []
            ),
            "last_stage": task_state["stage"],
            "error": str(e),
        }
        progress_callback("failed", {"error": str(e)})
        atomic_write_json(save_file, failure_result)
        return failure_result
    finally:
        clear_task_deadline()
        # 6. CRITICAL: Close the environment to kill the Java process
        if env:
            env.close()

# ==========================================
# Main Entry Point
# ==========================================

def main(args):
    model_name = args.model
    skill_mode = args.skill_strategy if args.use_skill else "none"
    
    # Setup output directory
    output_path = f'results/scienceworld/{model_name}/{args.split}_{args.exp_name}_skill_{skill_mode}'
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    if args.split == 'dev':
        with open('src/scienceworld/data/valid_indices.json', 'r') as f:
            indices = json.load(f)
    elif args.split == 'test':
        with open('src/scienceworld/data/test_indices.json', 'r') as f:
            indices = json.load(f)

    num_games = len(indices)
    print(f"Total games to evaluate: {num_games}")

    # ---------------------------------------------------------
    # Checkpointing (Filter finished)
    # ---------------------------------------------------------
    tasks_to_run = []
    finished_games = 0
    all_rewards = 0
    all_steps = 0
    existing_files = set()

    if os.path.exists(output_path):
        for file in os.listdir(output_path):
            if file.endswith('.json') and file.startswith('idx_'):
                try:
                    idx = int(file.split('_')[1].split('.')[0])
                    existing_files.add(idx)
                    
                    with open(f'{output_path}/{file}', 'r') as f:
                        res = json.load(f)
                        all_rewards += res['reward']
                        all_steps += res['steps']
                    finished_games += 1
                except:
                    continue

    # Filter tasks to run
    for idx in range(num_games):
        if idx not in existing_files:
            tasks_to_run.append(idx)

    if args.task_limit is not None:
        tasks_to_run = tasks_to_run[: args.task_limit]

    print(f"Already finished: {finished_games}, Remaining: {len(tasks_to_run)}")

    # ---------------------------------------------------------
    # Parallel Execution
    # ---------------------------------------------------------
    max_workers = args.max_workers
    print(f"Starting parallel evaluation with {max_workers} workers...")

    pbar = tqdm(total=len(tasks_to_run), desc="Evaluating ScienceWorld")
    if max_workers <= 1:
        for idx in tasks_to_run:
            try:
                result = eval_single_variation(idx, indices, args, output_path)
                if result:
                    finished_games += 1
                    all_rewards += result['reward']
                    all_steps += result['steps']
                    current_avg_reward = all_rewards / finished_games if finished_games > 0 else 0
                    current_avg_steps = all_steps / finished_games if finished_games > 0 else 0
                    pbar.set_postfix({
                        'Avg Reward': f'{current_avg_reward:.2f}',
                        'Avg Steps': f'{current_avg_steps:.2f}'
                    })
            except Exception as exc:
                print(f'\nTask {idx} generated an exception: {exc}')
            pbar.update(1)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(
                    eval_single_variation,
                    idx,
                    indices,
                    args,
                    output_path
                ): idx for idx in tasks_to_run
            }

            for future in concurrent.futures.as_completed(future_to_task):
                idx = future_to_task[future]
                try:
                    result = future.result()
                    if result:
                        finished_games += 1
                        all_rewards += result['reward']
                        all_steps += result['steps']
                        current_avg_reward = all_rewards / finished_games if finished_games > 0 else 0
                        current_avg_steps = all_steps / finished_games if finished_games > 0 else 0
                        pbar.set_postfix({
                            'Avg Reward': f'{current_avg_reward:.2f}',
                            'Avg Steps': f'{current_avg_steps:.2f}'
                        })
                except Exception as exc:
                    print(f'\nTask {idx} generated an exception: {exc}')
                pbar.update(1)
    pbar.close()

    # Final Summary
    print(f"Evaluation finished. Total Games: {finished_games}")
    if finished_games > 0:
        print(f"Final Avg Reward: {all_rewards / finished_games}")
        print(f"Final Avg Steps: {all_steps / finished_games}")

def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-4o')
    parser.add_argument('--split', type=str, default='dev', choices=['dev', 'test'])
    parser.add_argument('--max_workers', type=int, default=5, help="Number of parallel workers")
    parser.add_argument('--max_steps', type=int, default=30)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--use_skill', action='store_true')
    parser.add_argument('--skill_strategy', type=str, default='skillnet', choices=['baseline', 'skillnet', 'dsc'])
    parser.add_argument('--compiler_min_relevance', type=float, default=0.25)
    parser.add_argument('--compiler_preserve_top_k', type=int, default=2)
    parser.add_argument('--compiler_similar_prune_margin', type=float, default=0.08)
    parser.add_argument('--compiler_keep_parent_if_better_by', type=float, default=0.05)
    parser.add_argument('--compiler_coverage_weight', type=float, default=0.55)
    parser.add_argument('--compiler_quality_weight', type=float, default=0.20)
    parser.add_argument('--compiler_cost_weight', type=float, default=0.15)
    parser.add_argument('--compiler_latency_weight', type=float, default=0.10)
    parser.add_argument('--compiler_graph_passes', type=str, default='')
    parser.add_argument('--compiler_disable_workflow_bias', action='store_true')
    parser.add_argument('--compiler_critic_enabled', action='store_true')
    parser.add_argument('--compiler_critic_model', type=str, default='')
    parser.add_argument('--compiler_critic_min_coverage', type=float, default=0.45)
    parser.add_argument('--compiler_critic_force', action='store_true')
    parser.add_argument('--runtime_recompile', action='store_true')
    parser.add_argument('--runtime_recompile_max_rounds', type=int, default=2)
    parser.add_argument('--runtime_recompile_min_interval_steps', type=int, default=2)
    parser.add_argument('--runtime_recompile_stagnation_threshold', type=int, default=2)
    parser.add_argument('--runtime_recompile_min_remaining_steps', type=int, default=1)
    parser.add_argument('--task_limit', type=int, default=None)
    parser.add_argument('--llm_timeout', type=float, default=90.0)
    parser.add_argument('--llm_retry_attempts', type=int, default=3)
    parser.add_argument('--llm_retry_delay', type=float, default=3.0)
    parser.add_argument('--task_timeout', type=int, default=0)
    return parser


def configure_runtime(args):
    os.environ["EXPERIMENT_LLM_TIMEOUT_SECONDS"] = str(args.llm_timeout)
    os.environ["EXPERIMENT_LLM_RETRY_ATTEMPTS"] = str(args.llm_retry_attempts)
    os.environ["EXPERIMENT_LLM_RETRY_DELAY_SECONDS"] = str(args.llm_retry_delay)


def _run_as_imported_module(args):
    # Script-mode execution under Python.app was observed to hang inside the
    # ScienceWorld runner before task progress advanced. Reloading the file as a
    # named module matches the imported execution path, which runs reliably.
    # Use the real module name so ProcessPoolExecutor workers can import it.
    spec = importlib.util.spec_from_file_location("scienceworld_run", __file__)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Register the module before execution so worker processes can import and
    # pickle top-level callables from this file by module name.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.main(args)


if __name__ == '__main__':
    parser = build_arg_parser()
    args = parser.parse_args()
    configure_runtime(args)
    _run_as_imported_module(args)
