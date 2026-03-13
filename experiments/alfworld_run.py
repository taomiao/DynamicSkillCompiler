import os
from openai import OpenAI
import re
try:
    from retry import retry
except ImportError:
    def retry(*_args, **_kwargs):
        def decorator(func):
            return func
        return decorator
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json
import argparse
import yaml
import sys

from src.alfworld.prompts.system_prompt import alfworld_system_prompt
from src.skill import SkillModule

current_dir = os.path.dirname(os.path.abspath(__file__))
alfworld_path = os.path.join(current_dir, "alfworld")
if os.path.isdir(alfworld_path) and alfworld_path not in sys.path:
    sys.path.append(alfworld_path)

import alfworld
import alfworld.agents.environment
from alfworld.agents.environment import get_environment

client = None


def get_client():
    global client
    if client is None:
        client = OpenAI(
            api_key=os.environ["API_KEY"],
            base_url=os.environ["BASE_URL"]
        )
    return client

@retry(tries=5, delay=5, backoff=2, jitter=(1, 3))
def llm(prompt, model="YOUR_MODEL_NAME"):
    print(f'Calling LLM with model: {model}')
    # Normalize messages format
    if isinstance(prompt, list):
        messages = prompt
    elif isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        raise ValueError(f'prompt must be a list or a string, but got {type(prompt)}')
    response = get_client().chat.completions.create(
        model=model,
        messages=messages
    )
    # Extract content
    content = response.choices[0].message.content
    if content is not None:
        return content
    return "Output Error"


def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]    
    return ob

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

def parse_action(response: str) -> str:
    """
    Extracts the action from the LLM response using regex.
    """
    pattern = re.compile(r"Action:\s*(.+)", re.IGNORECASE)
    match = pattern.search(response)
    if match:
        return match.group(1).strip().strip('"\'*`')
    return ""

def run_standard_procedure(env, llm, model, process_ob, messages, max_steps):
    """
    Executes the standard interaction loop when no skill is available
    or when falling back from a failed skill retrieval.
    """
    task_done = False
    task_reward = 0
    current_steps = 0

    while not task_done and current_steps < max_steps:
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
        action = parse_action(response)
        action_list = [action]
        
        observation, task_reward, done, info = env.step(action_list)

        # 3. Process Observation
        observation, task_reward, task_done = (
            process_ob(observation[0]),
            info["won"][0],
            done[0]
        )

        print(f'{Colors.YELLOW}Observation: \n{observation}{Colors.RESET}')

        # 4. Update History
        messages.append({"role": "user", "content": f"Observation: {observation}"})

        if task_done:
            print(f'{Colors.GREEN}Whole Task completed! Reward: {task_reward}{Colors.RESET}')

    return messages, task_done, task_reward, current_steps

# ==========================================
# Core Execution Logic
# ==========================================

def alfworld_run_single(env, obs=[], names=[], max_steps=30, model=None, Skill_Module=None):
    """
    Execute a batch of Alfworld tasks (typically size 1 when parallelized).
    Handles both standard execution and skill-augmented procedure execution.
    """
    results = []
    
    for task_idx, (ob, name) in enumerate(zip(obs, names)):
        print(f'{Colors.RED}Processing task {task_idx + 1}/{len(obs)}: {name}{Colors.RESET}')

        # Extract specific task query
        query = ob.split('Your task is to: ')[-1].split('\n')[0].strip()
        
        # Initialize context
        messages = [{"role": "system", "content": alfworld_system_prompt}]
        messages.append({"role": "user", "content": ob})
        
        # Initialize loop variables
        task_done = False
        task_reward = 0
        steps = 0
        relevant_skill_names = []
        overall_procedure = ""
        overall_procedure_code = ""

        # Determine Execution Strategy (Skill_Module vs Standard)
        use_skill = False
        if Skill_Module is not None:
            relevant_skill_names = Skill_Module.retrieve_relevant_skills(ob)
            if relevant_skill_names:
                use_skill = True
                print(f'{Colors.BLUE}Retrieved relevant skills: {relevant_skill_names}{Colors.RESET}')
            else:
                print(f"[INFO] No relevant skills found for task: {name}. Falling back to standard execution.")

        # Execute Strategy
        if use_skill:
            overall_procedure = Skill_Module.generate_overall_procedure(ob, relevant_skill_names)
            print(f'\n{Colors.BLUE}Generated Overall Procedure:\n{overall_procedure}{Colors.RESET}')
            
            MAX_RETRIES = 3
            retries = 0
            
            while retries < MAX_RETRIES:
                try:
                    overall_procedure_code = Skill_Module.generate_overall_procedure_code(ob, overall_procedure)
                    print(f'\n{Colors.BLUE}Generated Overall Procedure Code:\n{overall_procedure_code}{Colors.RESET}')

                    # Dynamic execution of generated procedure
                    namespace = {}
                    exec(overall_procedure_code, namespace)
                    func = namespace["overall_procedure_code"]
                    
                    messages, task_done, task_reward, steps = func(
                        env, llm, model, process_ob, parse_action, messages, max_steps
                    )
                    
                    if task_done:
                        print(f'{Colors.GREEN} Task completed! Reward: {task_reward}{Colors.RESET}')
                    break 
                except Exception as e:
                    print(f'Error loading/executing procedure script: {e}')
                    retries += 1
        else:
            # Fallback to Standard Execution Loop
            messages, task_done, task_reward, steps = run_standard_procedure(
                env, llm, model, process_ob, messages, max_steps
            )
        
        # Record Results
        results.append({
            "query": query,
            "name": name,
            "task_done": task_done,
            "reward": task_reward,
            "steps": steps,
            "messages": messages,
            "relevant_skill_names": relevant_skill_names,
            "skill_strategy": getattr(Skill_Module, "selection_strategy", "none") if Skill_Module else "none",
            "compiler_metrics": (
                Skill_Module.last_compilation.metrics.__dict__
                if Skill_Module and Skill_Module.last_compilation is not None
                else None
            ),
        })
    
    return results

def eval_single_game(game_idx, args, config, split, output_path):
    """
    Worker function: Initializes an independent environment, fast-forwards to the 
    specific task, and performs the evaluation.
    """
    try:
        # 1. Independent Environment Initialization (Batch Size = 1)
        # This ensures thread safety/process isolation.
        env = get_environment(config["env"]["type"])(config, train_eval=split)
        env = env.init_env(batch_size=1)
        
        # 2. Fast-forward to specific Game ID
        obs_list = []
        info = {}
        for _ in range(game_idx + 1):
            obs_list, info = env.reset()
            
        # 3. Initialize SkillModule (if enabled)
        Skill_Module = None
        if args.use_skill:
            skill_config = {
                "skills_dir": "src/skills/alfworld",
                "overall_procedure_examples_path": "src/alfworld/alfworld_overall_procedure_examples.txt",
                "procedure_code_template_path": "src/alfworld/alfworld_procedure_code_template.py",
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
            }
            Skill_Module = SkillModule(**skill_config)

        # 4. Prepare Observation
        ob_str = '\n'.join(obs_list[0].split('\n\n')[1:])
        game_name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
        
        # 5. Execute Task
        batch_results = alfworld_run_single(
            env=env,
            obs=[ob_str], 
            names=[game_name], 
            max_steps=args.max_steps,
            model=args.model,
            Skill_Module=Skill_Module
        )
        
        result = batch_results[0]

        # 6. Save Results (Atomic write per file)
        save_file = f'{output_path}/idx_{game_idx}.json'
        with open(save_file, 'w') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
            
        return result

    except Exception as e:
        print(f"Error in game {game_idx}: {e}")
        return None
    finally:
        # 6. CRITICAL: Close the environment
        if env:
            env.close()

# ==========================================
# Main Entry Point
# ==========================================

def main(args):
    model_name = args.model
    skill_mode = args.skill_strategy if args.use_skill else "none"
    
    # Load configuration
    with open('src/alfworld/base_config.yaml') as reader:
        config = yaml.safe_load(reader)
    
    # Determine split
    split = "eval_in_distribution" if args.split == 'dev' else "eval_out_of_distribution"

    # Setup output directory
    output_path = f'results/alfworld/{model_name}/{args.split}_{args.exp_name}_skill_{skill_mode}'
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # ---------------------------------------------------------
    # Determine Total Game Count
    # ---------------------------------------------------------
    # Temporarily initialize environment to get the total number of games
    temp_env = get_environment(config["env"]["type"])(config, train_eval=split)
    temp_env = temp_env.init_env(batch_size=1)
    num_games = len(temp_env.gamefiles)
    del temp_env 
    print(f"Total games to evaluate: {num_games}")

    # ---------------------------------------------------------
    # Identify Remaining Tasks (Checkpointing)
    # ---------------------------------------------------------
    tasks_to_run = []
    finished_games = 0
    all_rewards = 0
    all_steps = 0
    existing_files = set()

    # Scan existing results
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

    pbar = tqdm(total=len(tasks_to_run), desc="Evaluating ALFWorld")
    if max_workers <= 1:
        for idx in tasks_to_run:
            try:
                result = eval_single_game(idx, args, config, split, output_path)
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
                print(f'\nGame {idx} generated an exception: {exc}')
            pbar.update(1)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    eval_single_game,
                    idx,
                    args,
                    config,
                    split,
                    output_path
                ): idx for idx in tasks_to_run
            }

            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
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
                    print(f'\nGame {idx} generated an exception: {exc}')
                pbar.update(1)
    pbar.close()

    # Final Summary
    print(f"Evaluation finished. Total Games: {finished_games}")
    if finished_games > 0:
        print(f"Final Avg Reward: {all_rewards / finished_games}")
        print(f"Final Avg Steps: {all_steps / finished_games}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-4o')
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--max_workers', type=int, default=10, help="Number of parallel workers")
    parser.add_argument('--max_steps', type=int, default=30)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--use_skill', action='store_true')
    parser.add_argument('--skill_strategy', type=str, default='baseline', choices=['baseline', 'dsc'])
    parser.add_argument('--compiler_min_relevance', type=float, default=0.25)
    parser.add_argument('--compiler_preserve_top_k', type=int, default=2)
    parser.add_argument('--compiler_similar_prune_margin', type=float, default=0.08)
    parser.add_argument('--compiler_keep_parent_if_better_by', type=float, default=0.05)
    parser.add_argument('--compiler_coverage_weight', type=float, default=0.55)
    parser.add_argument('--compiler_quality_weight', type=float, default=0.20)
    parser.add_argument('--compiler_cost_weight', type=float, default=0.15)
    parser.add_argument('--compiler_latency_weight', type=float, default=0.10)
    parser.add_argument('--task_limit', type=int, default=None)
    args = parser.parse_args()
        
    main(args)
