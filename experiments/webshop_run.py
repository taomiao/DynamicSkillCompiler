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
import gym 
import yaml
import sys
import argparse
import traceback
import multiprocessing

from src.webshop.prompts.system_prompt import webshop_system_prompt
from src.skill import SkillModule
from src.runtime_recompile import execute_compiled_procedure

current_dir = os.path.dirname(os.path.abspath(__file__))
webshop_candidates = [
    os.path.join(current_dir, "webshop"),
    os.path.join(current_dir, "WebShop"),
]
webshop_path = next((path for path in webshop_candidates if os.path.isdir(path)), webshop_candidates[0])
if webshop_path not in sys.path:
    sys.path.append(webshop_path)
from web_agent_site.envs import WebAgentTextEnv
from web_agent_site.engine.engine import load_products
from web_agent_site.engine.goal import get_goals
from web_agent_site.utils import DEFAULT_FILE_PATH


client = None


def get_client():
    global client
    if client is None:
        client = OpenAI(
            api_key=os.environ["API_KEY"],
            base_url=os.environ["BASE_URL"]
        )
    return client


def filter_valid_session_ids(session_ids, num_products=None):
    if num_products is None:
        return session_ids

    all_products, _, product_prices, _ = load_products(
        filepath=DEFAULT_FILE_PATH,
        num_products=num_products,
        human_goals=False,
    )
    max_goal_idx = len(get_goals(all_products, product_prices, human_goals=False))
    valid_session_ids = [sid for sid in session_ids if isinstance(sid, int) and 0 <= sid < max_goal_idx]
    invalid_count = len(session_ids) - len(valid_session_ids)
    print(
        f"Filtered invalid WebShop sessions for num_products={num_products}: "
        f"kept {len(valid_session_ids)}, dropped {invalid_count}, goal_count={max_goal_idx}"
    )
    return valid_session_ids

@retry(tries=5, delay=5, backoff=2, jitter=(1, 3))
def llm(prompt, model="YOUR_MODEL_NAME"):
    print(f'Calling LLM with model: {model}')
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
    content = response.choices[0].message.content
    return content if content is not None else "Output Error"

class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def parse_action(response: str) -> str:
    pattern = re.compile(r"Action:\s*(.+)", re.IGNORECASE)
    match = pattern.search(response)
    if match:
        return match.group(1).strip().strip('"\'*`')
    return ""

def run_standard_procedure(env, llm, model, messages, max_steps):
    task_done = False
    task_reward = 0
    current_steps = 0

    while not task_done and current_steps < max_steps:
        current_steps += 1
        
        try:
            response = llm(messages, model)
            print(f'{Colors.GREEN}Agent response: \n{response}{Colors.RESET}')
        except Exception as e:
            print(f'{Colors.RED}Error in LLM call: {e}{Colors.RESET}')
            break

        messages.append({"role": "assistant", "content": response})
        action = parse_action(response)
        
        observation, reward, done, info = env.step(action)

        task_reward = reward
        task_done = done

        print(f'{Colors.YELLOW}Observation: \n{observation}{Colors.RESET}')

        messages.append({"role": "user", "content": f"Observation: {observation}"})

        if task_done:
            print(f'{Colors.GREEN}Whole Task completed! Reward: {task_reward}{Colors.RESET}')

    return messages, task_done, task_reward, current_steps


def _webshop_step_adapter(action, result):
    observation, reward, done, _info = result
    return {
        "action": action,
        "observation": observation,
        "task_done": done,
        "task_reward": reward,
    }


def _invoke_compiled_webshop(func, env, model, messages, remaining_steps):
    return func(
        env,
        llm,
        model,
        parse_action,
        messages,
        remaining_steps,
    )


def webshop_run_single(env, ob, instruction_text, max_steps=30, model=None, Skill_Module=None):
    results = []
    
    query = instruction_text
    
    # Initialize context
    messages = [{"role": "system", "content": webshop_system_prompt}]
    messages.append({"role": "user", "content": ob})
    
    task_done = False
    task_reward = 0
    steps = 0
    relevant_skill_names = []
    overall_procedure = ""
    overall_procedure_code = ""

    use_skill = False
    if Skill_Module is not None:
        relevant_skill_names = Skill_Module.retrieve_relevant_skills(ob)
        if relevant_skill_names:
            use_skill = True
            print(f'{Colors.BLUE}Retrieved relevant skills: {relevant_skill_names}{Colors.RESET}')
        else:
            print(f"[INFO] No relevant skills found for task: {ob}. Falling back to standard procedure.")


    # Execute Strategy
    if use_skill:
        compiled_result = execute_compiled_procedure(
            env=env,
            llm=llm,
            model=model,
            task_prompt=ob,
            messages=messages,
            max_steps=max_steps,
            skill_module=Skill_Module,
            selected_skill_names=relevant_skill_names,
            step_adapter=_webshop_step_adapter,
            invoke=lambda func, proxy, msg_history, remaining: _invoke_compiled_webshop(
                func,
                proxy,
                model,
                msg_history,
                remaining,
            ),
        )
        messages = compiled_result["messages"]
        task_done = compiled_result["task_done"]
        task_reward = compiled_result["task_reward"]
        steps = compiled_result["steps"]
        relevant_skill_names = compiled_result["skill_names"]
        overall_procedure = compiled_result["overall_procedure"]
        overall_procedure_code = compiled_result["overall_procedure_code"]
        if task_done:
            print(f'{Colors.GREEN} Task completed! Reward: {task_reward}{Colors.RESET}')
    else:
        messages, task_done, task_reward, steps = run_standard_procedure(
            env, llm, model, messages, max_steps
        )
    
    results.append({
        "query": query,
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
        "runtime_recompile_count": getattr(Skill_Module, "runtime_recompile_count", 0) if Skill_Module else 0,
        "runtime_recompile_events": getattr(Skill_Module, "runtime_recompile_events", []) if Skill_Module else [],
    })
    
    return results

def eval_single_game(game_idx, session_id, args, output_path):
    try:
        # WebShop environment setup
        port = 3500 + game_idx
        base_url = 'http://127.0.0.1:' + str(port)
        env = gym.make(
            'WebAgentTextEnv-v0',
            observation_mode='text',
            num_products=args.num_products,
            base_url=base_url,
        )
        print(f'{Colors.BLUE}Initialized WebShop environment for game {game_idx} at port {port}{Colors.RESET}')
        ob = env.reset(session=session_id)[0] 

        print(f'{Colors.RED}Processing task {game_idx} (Session {session_id}): {ob}{Colors.RESET}')
        
        instruction_text = env.instruction_text

        # Initialize SkillModule (if enabled)
        Skill_Module = None
        if args.use_skill:
            skill_config = {
                "skills_dir": "src/skills/webshop",
                "overall_procedure_examples_path": "src/webshop/webshop_overall_procedure_examples.txt",
                "procedure_code_template_path": "src/webshop/webshop_procedure_code_template.py",
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

        batch_results = webshop_run_single(
            env=env,
            ob=ob, 
            instruction_text=instruction_text,
            max_steps=args.max_steps,
            model=args.model,
            Skill_Module=Skill_Module
        )
        
        result = batch_results[0]
        result['session_id'] = session_id
        save_file = f'{output_path}/idx_{game_idx}.json'
        with open(save_file, 'w') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
            
        return result

    except Exception as e:
        print(f"Error in game {game_idx} (Session {session_id}):")
        traceback.print_exc()
        return None
    finally:
        try:
            env.close()
        except:
            pass

def main(args):
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    model_name = args.model
    skill_mode = args.skill_strategy if args.use_skill else "none"
    output_path = f'results/webshop/{model_name}/{args.exp_name}_skill_{skill_mode}'
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    with open('src/webshop/data/test_indices.json', 'r') as f:
        session_ids = json.load(f)
    if args.filter_invalid_sessions:
        session_ids = filter_valid_session_ids(session_ids, num_products=args.num_products)

    # ---------------------------------------------------------
    # Identify Remaining Tasks (Checkpointing)
    # ---------------------------------------------------------
    num_games = len(session_ids)
    tasks_to_run = {}
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
            # tasks_to_run.append(idx)
            tasks_to_run[idx] = session_ids[idx]

    if args.task_limit is not None:
        limited_items = list(tasks_to_run.items())[: args.task_limit]
        tasks_to_run = dict(limited_items)

    print(f"Already finished: {finished_games}, Remaining: {len(tasks_to_run)}")


    # ---------------------------------------------------------
    # Parallel Execution
    # ---------------------------------------------------------
    print(f"Starting parallel evaluation with {args.max_workers} workers...")

    pbar = tqdm(total=len(tasks_to_run), desc="Evaluating WebShop")
    if args.max_workers <= 1:
        for idx, session_id in tasks_to_run.items():
            try:
                result = eval_single_game(idx, session_id, args, output_path)
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
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_idx = {
                executor.submit(eval_single_game, idx, session_id, args, output_path): idx
                for idx, session_id in tasks_to_run.items()
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
    parser.add_argument('--max_workers', type=int, default=1, help="Number of parallel workers")
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
    parser.add_argument('--num_products', type=int, default=None)
    parser.add_argument('--task_limit', type=int, default=None)
    parser.add_argument('--filter_invalid_sessions', action='store_true')
    args = parser.parse_args()
    main(args)
