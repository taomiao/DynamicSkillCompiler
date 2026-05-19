#v2
def overall_procedure_code(
    env,
    llm,
    model: str,
    process_ob,
    parse_action,
    messages: list = [],
    max_steps: int = 30
):
    """
    Procedure code template for solving a household interactive task using iterative LLM-guided actions. 
    
    INPUTS:
    - env: The environment instance to interact with. (Predefined, call directly)
    - llm: The language model function to generate agent responses. (Predefined, call directly)
    - model: The specific model name or identifier to use with the llm. (Predefined, pass directly)
    - process_ob: A function to process raw environment observations. (Predefined, call directly)
    - parse_action: A function to extract executable actions from LLM responses. (Predefined, call directly)
    - messages: Pre-populated dialogue context as a list of message dicts. (Contains messages describing the environment background and the task)
    - max_steps: Maximum number of interaction steps before termination. (Predefined, pass directly)
    RETURNS:
    - messages: The updated dialogue context including all interactions.
    - task_done: Boolean indicating if the task was successfully completed.
    - reward: The final reward received from the environment.
    - current_steps: The total number of steps taken in the interaction loop.
    These Input Arguments and Return Types MUST NOT be changed.

    This function serves as a STRICT TEMPLATE to enforce grounded, environment-driven task execution.
    """

    # ------------------------------------------------------------------
    # [SECTION] Overall Procedural Guidance INJECTION
    # INSTRUCTION: You should add detailed Overall Procedural Guidance here to help the agent better complete the task. 
    #              These guidelines will be visible to the agent as user messages.
    #              You may define additional helper variables and functions as needed to help the generated code running correctly and efficiently.
    # ------------------------------------------------------------------
    # Example: 
    procedure_guidelines = "<detailed_overall_procedural_guidelines_here>"
    messages.append({"role": "user", "content": procedure_guidelines})
    # ------------------------------------------------------------------

    # --- Core task execution state ---
    task_done = False
    current_steps = 0
    reward = 0  # always track latest reward

    # --- Main agent loop (IMMUTABLE STRUCTURE) ---
    while not task_done and current_steps < max_steps:

        # ============================================================
        # 1) THOUGHT PHASE — Query LLM based on current message state
        #    - DO NOT change this call signature
        # ============================================================
        try:
            response = llm(messages, model)
            print(f'\033[92mAgent response: \n{response}\033[0m')
        except Exception as e:
            print(f'\033[91mError in LLM call: {e}\033[0m')
            break

        # ============================================================
        # 2) Persist assistant response into dialogue
        # ============================================================
        messages.append({"role": "assistant", "content": response})

        # ============================================================
        # 3) ACTION PARSING PHASE — Extract executable command
        #    Expected format: "Action: <command>"
        # ============================================================
        action = parse_action(response)

        # ============================================================
        # 4) ENVIRONMENT INTERACTION PHASE — Execute Action
        #    - Must use env.step()
        #    - Do not alter the list wrapping behavior
        #    - Task completion (task_done) and reward MUST ONLY be
        #      derived from env.step() outputs. DO NOT INFER.
        # ============================================================
        action_list = [action]
        observation, reward, task_done, info = env.step(action_list)

        # ============================================================
        # 5) OBSERVATION NORMALIZATION PHASE
        #    - Convert env raw outputs into semantic signals
        #    - Do NOT modify this section
        # ============================================================
        won = info.get("won", [reward]) if isinstance(info, dict) else [reward]
        done_value = task_done[0] if isinstance(task_done, (list, tuple)) else bool(task_done)
        observation, reward, task_done = (
            process_ob(observation[0]),
            won[0] if won else reward,
            done_value
        )

        print(f'\033[93mObservation: \n{observation}\033[0m')

        # ============================================================
        # 6) FEEDBACK PHASE — Feed environment signal back to agent
        # ============================================================
        messages.append({"role": "user", "content": f"Observation: {observation}"})

        # ==============================================================================
        # [SECTION] RUNTIME ANALYSIS & INTERVENTION (Optional, Leave empty if not needed.)
        # INSTRUCTION: Analyze 'observation' or 'response'. 
        #              You can append EXTRA hints to 'messages' here.
        #              WARNING: Do NOT use 'continue', 'break', or 'return'.
        # ==============================================================================
        # Example (change as needed):
        if "<specific_condition>" in observation:
            extra_hint = "<extra_hint_here>"
            messages.append({"role": "user", "content": extra_hint})
        # ==============================================================

        # ============================================================
        # 7) STEP ACCOUNTING
        # ============================================================
        current_steps += 1

    return messages, task_done, reward, current_steps
