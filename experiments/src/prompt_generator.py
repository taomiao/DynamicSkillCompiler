def retrieve_relevant_skills_prompt(metadata, task, max_skills=5, candidate_mode=False):
    template = "You are an expert in task analysis and skills retrieval."
    selection_rule = (
        f"Match the task requirements with the most relevant skills based on the metadata descriptions. "
        f"Include a broader but still high-signal candidate pool for downstream compilation. "
        f"(Target {max_skills} skills, and do not exceed {max_skills + 2} unless absolutely necessary)"
        if candidate_mode
        else f"Match the task requirements with the most relevant skills based on the metadata descriptions. "
             f"Only include skills that are most helpful for accomplishing the task. "
             f"(No more than {max_skills} unless necessary)"
    )
    prompt = f'''
Your task is to analyze the given task description and retrieve the most relevant skills from the provided metadata (name and description pairs).

# Principles
1.  **Contextual Understanding**: Thoroughly analyze the task description to grasp its requirements.
2.  **Relevance Matching**: {selection_rule}
3.  **Clarity and Precision**: Clearly list the names of the relevant skills in your output.
4.  **Output Format**: Return the skill names as a JSON list of strings. Must strictly match names in the provided metadata.
5.  **Allow Empty List**: If no relevant skills are found, return an empty list `[]`.

# Input Data
1.  **Metadata:** 
{metadata}
2.  **Task Description:**
{task}

Keep your output in the format below:
<Analysis> your analysis here </Analysis>
<Relevant_Skill_Names> relevant skill names list in JSON format here </Relevant_Skill_Names>
'''
    return [{"role": "system", "content": template}, {"role": "user", "content": prompt}]


def generate_overall_procedure_prompt(task, overall_procedure_examples, skill_contents, compiler_summary=""):
    template = "You are a Senior Systems Architect."
    prompt = f'''
You are given a complex task description, relevant skill contents, and examples of well-structured procedural guidance texts.
Your task is to create a structured "Procedural Guidance" text. This text will serve as the "brain" for an AI Agent, telling it exactly how to solve the task.

# CRITICAL INSTRUCTIONS (Follow these strictly)
1.  **Structure with Phases**: You MUST divide the task into logical phases (e.g., Phase 1: Search, Phase 2: Acquire, Phase 3: Place).
2.  **Use Exact Syntax**: When describing actions, you MUST quote the exact syntax from the "Available Actions" (e.g., use "go to {{recep}}" instead of "walk to the desk").
3.  **State & Preconditions**: Explicitly mention necessary preconditions (e.g., "If the drawer is closed, use 'open drawer' before taking").
4.  **Error Handling**: Include a rule for what to do if the agent encounters an error or unexpected situation.
5.  **Observation Grounding**: Do NOT hard-code object colors, wire colors, bulbs, boxes, rooms, or container mappings unless they are explicitly stated in the task description or must be confirmed from environment observation first.
6.  **No Invented Components**: If a skill mentions example components, treat them as patterns only. Your procedure must tell the agent to inspect the current environment and use the components that actually exist in this variation.
7.  **Executable Electrical Syntax**: For electrical tasks, do not emit abstract commands like `connect X to Y`. Require exact contact-point actions based on observation, such as `battery anode`, `battery cathode`, `wire terminal 1`, `wire terminal 2`, `bulb anode`, `bulb cathode`, or object `terminal 1/2`.
8.  **Separate Tool Search from Target Search**: If the task requires both a tool and a target object, write separate phases for locating/acquiring the tool and locating/acquiring the target. Do not assume they are in the same room.
9.  **Ambiguity Handling**: If the environment returns `Ambiguous request: Please enter the number...`, the next action must be only the selected index, such as `0`, not a restated object name.
10. **Liquid Handling Is Special**: Never tell the agent to `pick up water`, `move water`, or otherwise manipulate a liquid directly. Use a container, `pour`, or a fill sequence such as `move <container> to <source>` -> `activate <source>` -> `deactivate <source>` -> `pick up <container>`.
11. **Growth Tasks Need Monitoring**: For seed or plant growth tasks, after planting into prepared soil and water, explicitly `focus on` the planted seed/plant and use `wait` cycles before checking the next growth stage.

# Input Data
1.  **Task Description:**
{task}
2.  **Compiler Summary / Compiled Skill Package:**
{compiler_summary if compiler_summary else "[No compiler summary provided]"}
3.  **Relevant Skill Contents:**
{skill_contents}

# Examples
Here are some examples of well-structured Procedural Guidance texts you should emulate (focus on style, structure, and level of detail instead of content):
{overall_procedure_examples}


Keep your output in the format below:
<Analysis> your analysis here </Analysis>
<Overall_Procedure> your generated overall procedure guidance here </Overall_Procedure>
'''
    return [{"role": "system", "content": template}, {"role": "user", "content": prompt}]

def generate_overall_procedure_code_prompt(task, overall_procedure, procedure_code_template):
    template = "You are a Senior Software Engineer specializing in AI-driven automation and safe code generation."
    prompt = f'''
You are given a complex task description, overall procedural guidance, and a strict procedure code template.
Your task is to implement a Python function `overall_procedure_code` that follows the provided procedure code template and incorporates the overall procedure guidance to help Agent solve a complex task.

# Principles
1.  **Adherence to Template**: Strictly follow the provided procedure code template. Input arguments and return types must align with the template. (Most Important)
2.  **Inject Overall Procedural Guidance**: Help the agent accomplish the task by injecting the overall procedure guidance into the function.
4.  **Error Handling**: Implement robust error handling for LLM calls and environment interactions.
5.  **Function Naming**: Name the function `overall_procedure_code` as per the template.

# Input Data
1.  **Task Description:**
{task}
2.  **Overall Procedural Guidance:**
{overall_procedure}
3.  **Procedure Code Template:**
```python
{procedure_code_template}
```

Keep your output in the format below:
<Analysis> your analysis here </Analysis>
<Overall_Procedure_Code> your generated overall procedure workflow code here </Overall_Procedure_Code>
'''
    return [{"role": "system", "content": template}, {"role": "user", "content": prompt}]


def optimize_compiled_skills_prompt(
    task,
    current_skill_names,
    available_skill_summaries,
    compiler_summary="",
    max_skills=6,
):
    template = "You are an expert skill compiler critic focused on execution quality first and efficiency second."
    prompt = f'''
You are reviewing a dynamically compiled task-specific skill package.
Your job is to improve the package so it preserves the skills required to solve the task correctly, while still keeping the package concise.

# Optimization Priorities
1. **Execution Quality First**: Do not drop skills that are needed to locate tools or targets, resolve ambiguity, satisfy prerequisites, or enforce correct phase ordering.
2. **Task Completion Over Compression**: A slightly larger package is acceptable if it materially improves correctness or stability.
3. **Efficiency Second**: After preserving quality-critical skills, remove only skills that are clearly redundant or overly broad.
4. **Grounded Decisions**: Base your decision on the task, current compiled selection, and available candidate skills. Do not invent skill names.
5. **Structured Output**: Return a single JSON object inside the required XML tag.

# Task
{task}

# Current Compiled Selection
{current_skill_names}

# Compiler Summary
{compiler_summary if compiler_summary else "[No compiler summary provided]"}

# Available Candidate Skills
{available_skill_summaries}

# Output Rules
- `must_keep_skills`: exact skill names that must remain in the final package
- `preferred_skill_order`: the recommended final ordered skill list
- `drop_skills`: skills that can be safely excluded
- `task_family`: a short label for the task family
- `reasoning_summary`: one short sentence describing the main correction
- Keep the final package at or below {max_skills} skills when possible, but exceed that only if necessary for correctness.

Keep your output in the format below:
<Analysis> your analysis here </Analysis>
<Compile_Critic_Decision>{{
  "task_family": "short label",
  "must_keep_skills": ["skill-a"],
  "preferred_skill_order": ["skill-a", "skill-b"],
  "drop_skills": ["skill-x"],
  "reasoning_summary": "one short sentence"
}}</Compile_Critic_Decision>
'''
    return [{"role": "system", "content": template}, {"role": "user", "content": prompt}]
