import re


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


def generate_overall_procedure_prompt(
    task,
    overall_procedure_examples,
    skill_contents,
    compiler_summary="",
    procedure_patch_mode: bool = False,
):
    template = "You are a Senior Systems Architect."
    patch_block = ""
    if procedure_patch_mode:
        patch_block = """
# PROCEDURE PATCH MODE (this request follows a runtime failure — read carefully)
- Do **not** rewrite a full manual from scratch if a long procedure already exists in the conversation.
- Output a **compact correction** (aim under ~400 words): start with `## ERRATA` (1–4 bullets: what was wrong or mis-ordered), then `## NEXT STEPS` (3–10 bullets: only immediate fixes).
- If prior guidance is mostly valid, say so in one line and patch only the broken phases.
- Avoid duplicating generic exploration steps already covered earlier; stay environment-agnostic in vocabulary (no benchmark-specific hacks unless the task text requires it).
"""
    prompt = f'''
You are given a complex task description, relevant skill contents, and examples of well-structured procedural guidance texts.
Your task is to create a structured "Procedural Guidance" text. This text will serve as the "brain" for an AI Agent, telling it exactly how to solve the task.
{patch_block}
# CRITICAL INSTRUCTIONS (Follow these strictly)
1.  **Structure with Phases**: You MUST divide the task into logical phases (e.g., Phase 1: Search, Phase 2: Acquire, Phase 3: Place). In PATCH MODE, phases may be limited to ERRATA + NEXT STEPS as specified above.
2.  **Use Exact Syntax**: When describing actions, you MUST quote the exact syntax from the "Available Actions" (e.g., use "go to {{recep}}" instead of "walk to the desk").
3.  **State & Preconditions**: Explicitly mention necessary preconditions tied to **fresh observation** (e.g., after an observation action, if a container is closed, use the documented open action before taking from it).
4.  **Error Handling**: Include a rule for what to do if the agent encounters an error or unexpected situation.
5.  **Observation Grounding**: Do NOT hard-code object colors, component identities, containers, rooms, or mappings unless they are explicitly stated in the task description or must be confirmed from environment observation first.
6.  **No Invented Components**: If a skill mentions example components, treat them as patterns only. Your procedure must tell the agent to inspect the current environment and use the components that actually exist in this variation.
7.  **Executable Multi-Part Syntax**: For tasks that connect, combine, attach, or compare parts, do not emit abstract commands like `connect X to Y`. Require exact action syntax and endpoint/object names from the latest observation or available-action format.
8.  **Separate Tool Search from Target Search**: If the task requires both a tool and a target object, write separate phases for locating/acquiring the tool and locating/acquiring the target. Do not assume they are in the same room.
9.  **Ambiguity Handling**: If the environment returns `Ambiguous request: Please enter the number...`, the next action must be only the selected index, such as `0`, not a restated object name.
10. **Liquid Handling Is Special**: Never tell the agent to `pick up water`, `move water`, or otherwise manipulate a liquid directly. Use a container, `pour`, or a fill sequence such as `move <container> to <source>` -> `activate <source>` -> `deactivate <source>` -> `pick up <container>`.
11. **State-Change Tasks Need Monitoring**: For tasks where an object must change state over time, explicitly mark the intended target if the environment supports such a milestone action, then use legal wait/observe cycles before checking the next stage.
12. **Conciseness**: Prefer the shortest guidance that preserves correctness; avoid repeating the same rule in multiple phases. Skip boilerplate that does not change behavior.
13. **Minimal Action Tokens in Examples**: In quoted example actions, use **short object heads** as the environment lists them (e.g. `pick up tin cup`), not full parenthetical descriptions copied from observations (e.g. avoid `pick up tin cup (containing red paint)` in examples). Teach disambiguation via the ambiguity rule instead.
14. **Closed Vocabulary for Example Actions**: Example commands must use only verbs and patterns that appear in the agent's **Available Actions** / system prompt for the target environment. Do **not** invent actions such as timed waits, unsupported ignition/reading commands, or unsupported placement commands unless they are explicitly listed. For waiting, use only forms the environment documents.
15. **Focus / Milestone Safety**: When the task requires signaling intent on a specific object at a milestone, state that `focus on` must target a name **confirmed in the latest observation** after the prerequisite mixture or state change—not a guessed label.
16. **Candidate Queue for Search**: For any locate/search/browse phase, tell the agent to keep a small tried-candidate queue, avoid repeating failed queries/targets, and relax secondary constraints before abandoning the core goal.
17. **One-at-a-Time Transport**: If the task may involve multiple objects or limited carrying capacity, prescribe a loop that finishes the currently held object's destination/transform before acquiring another object.
18. **Transform Protocol**: For heat/cool/clean/wash/slice/toggle-style tasks, separate the protocol into acquire object -> locate/prepare tool or appliance -> apply one legal transform action -> verify transformed state -> deliver/place. If the legal syntax is `verb {{obj}} with {{tool}}`, keep `{{obj}}` held/available and use that direct `with` action; do not first place the object inside the tool unless the environment explicitly documents that as required.
19. **List-to-Inspect Commitment**: For any task with discovery/list/search results, do not demand perfect surface-text matches before inspection. Maintain a best-so-far candidate with observed hard constraints; after limited horizontal exploration or one relaxed query, inspect the strongest partial candidate before continuing broad exploration.
20. **Soft Descriptor Evidence**: Treat descriptive preferences such as style, fit, use case, quality adjectives, or speed as soft evidence unless the task explicitly marks them mandatory. Do not reject an otherwise strong candidate only because surface text lacks a soft descriptor; inspect the candidate state/details/options first, and commit when hard constraints have evidence and there is no direct contradiction.
21. **Explicit Option Commitment**: If the task states an explicit attribute value (for example color, size, target state, destination, tool, or variant) and the current environment exposes that value as a selectable/actionable option, select or apply that exact option before the final commit. Prior search terms, object labels, or earlier observations are not enough evidence to skip an available explicit option.
22. **Evidence Staging**: On discovery/list/search pages, use only coarse identity/type and directly visible hard constraints to decide whether to inspect. Do not require surface text to contain fine-grained attributes such as color, size, material, care instructions, destination, or other option-like values; verify those on inspection/state pages before rejecting a plausible candidate.

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


def refine_dsc_procedure_prompt(
    task,
    relevant_skill_names,
    skill_contents,
    compiler_summary,
    draft_procedure,
):
    """
    Second-pass prompt for a strong reasoner (e.g. Claude Opus + extended thinking)
    to tighten procedural guidance for higher reward and fewer steps.
    """
    system = """You are the **Procedure Refinement** pass for Dynamic Skill Compilation (DSC).

**Mode: conservative patch**, not rewrite. The draft already passed task+skill context from the compiler. Your output must read like the **same playbook**, edited for clarity and duplicate removal only.

**Hard reject criteria** (if you would violate these, output the draft unchanged inside `<Overall_Procedure>` instead):
- Do not reorder **part-way vs completion** `focus` / milestones relative to the **Task description** text.
- Do not drop `open` / `look at` / cupboard / disambiguation / `pour`→`mix` chains that appear in the draft or are required when observations mention closed containers or ambiguous picks.
- Do not replace benchmark wording with real-world chemistry narratives.

Use extended reasoning internally, then produce a short Analysis and a single refined procedure block."""

    names = relevant_skill_names or []
    names_lines = "\n".join(f"- {n}" for n in names) if names else "- (none listed)"
    cs = compiler_summary if compiler_summary else "[No compiler summary provided]"
    sc = skill_contents if skill_contents else "[No skill contents provided]"
    draft = draft_procedure if draft_procedure else "[Empty draft]"
    draft_body = draft if draft != "[Empty draft]" else ""
    draft_chars = len(draft_body)
    draft_phases = len(re.findall(r"\bPhase\s*\d+", draft_body, re.I))
    lo = max(80, int(0.42 * max(draft_chars, 1)))
    hi = int(max(draft_chars * 1.25, draft_chars + 50))
    min_phases_keep = max(1, draft_phases - 1) if draft_phases else 0

    bounds_line = (
        f"\n# Draft size guard (the downstream system may reject over-compressed output)\n"
        f"- Draft length: **{draft_chars}** characters.\n"
        f"- Detected headings like `Phase 1`, `Phase 2`, …: **{draft_phases}** sections.\n"
        f"- Your `<Overall_Procedure>` should be roughly **{lo}–{hi}** characters (trim repetition only; do not gut content).\n"
    )
    if draft_phases >= 3:
        bounds_line += (
            f"- Keep **at least {min_phases_keep}** phase sections (same ordering). "
            "Merging two phases is allowed only if they repeat the same steps verbatim.\n"
        )

    user_parts = [
        "# Task description (authoritative)\n",
        str(task),
        "\n\n# Retrieved relevant skill names (DSC retrieval)\n",
        names_lines,
        "\n\n# Compiler summary / compiled skill package (context)\n",
        str(cs),
        "\n\n# Full relevant skill contents (preserve intent; skills are evidence for preconditions)\n",
        str(sc),
        "\n\n# Draft procedural guidance (PATCH this; do not replace with a new strategy)\n",
        str(draft),
        bounds_line,
        """

# Milestone fidelity (non‑negotiable)
1. In `<Analysis>`, quote **verbatim** the task sentences that specify **part-way** and **completion** obligations (especially any `focus on …` wording).
2. In `<Overall_Procedure>`, those obligations must appear in the **same relative order** as in the Task description. **Task text beats chemistry intuition** (benchmarks are quirky).
3. When the task names a **thing** (plant, animal, pigment, mercury, …), `focus` / `pick up` must target that **entity** as named in observations—not a parent container—unless the task explicitly points at the container.

# Allowed edits
- Remove **verbatim** duplicate bullets; merge two bullets that prescribe the identical action.
- Shorten prose while keeping the same action sequence and preconditions.
- Add **at most one** bullet of generic error recovery if the draft completely omits ambiguity handling and the skills imply disambiguation.

# Forbidden edits
- Inventing a new multi-phase strategy not present in the draft.
- Collapsing “locate tool / locate target / mix / focus milestones” into one paragraph if the draft separated them and the task requires sequencing.
- Emitting pseudo-code, `ERROR:`, `STOP`, or XML inside instructions meant to become `Action:` lines.

# Simulator hygiene
- Actions: only verbs documented for the current agent/environment; do not invent verbs or action formats.
- After `Ambiguous request…`, the next step is **only** a numeric index.

# Output format
`<Analysis>`: (1) quoted milestone lines from the task, (2) whether the draft matched them, (3) 1–5 bullets listing **only** edits applied.
`<Overall_Procedure>`: tightened draft only.

Do not mention this meta-instruction or model names inside `<Overall_Procedure>`.""",
    ]
    user = "".join(user_parts)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def refine_dsc_procedure_addendum_prompt(
    task,
    draft_procedure,
    relevant_skill_names,
    compiler_summary="",
    max_draft_chars: int = 12000,
):
    """
    Non-destructive refiner: model outputs only a short addendum (bullets) appended to the draft.
    Avoids full-rewrite regressions seen in replace mode.
    """
    system = """You are a **DSC procedure addendum** writer for an embodied simulator agent.

The primary procedural guidance has **already been generated**. You must **NOT** rewrite or summarize it.

Output **only** 3–8 new bullet lines that plug **gaps** the draft might miss: milestone wording vs task text, inspect/open before taking from closed or hidden containers, ambiguity replies (numeric index only), target entity vs container, measured property vs one-shot reading, liquid/container rules if relevant.

Rules:
- Each bullet is one actionable reminder (no full phases, no repeating steps already spelled out in the draft).
- Do not contradict the draft’s phase order or task milestones.
- No pseudo-code or `ERROR:` lines.

Output format (mandatory, single block):
<Refiner_Addendum>
- bullet one
- bullet two
...
</Refiner_Addendum>"""

    names = relevant_skill_names or []
    names_lines = "\n".join(f"- {n}" for n in names) if names else "- (none)"
    cs = compiler_summary if compiler_summary else "[none]"
    body = draft_procedure or ""
    if len(body) > max_draft_chars:
        keep = max_draft_chars - 80
        body = body[:keep] + "\n\n[... draft truncated; do not duplicate content from above ...]\n"

    user = (
        "# Task\n"
        + str(task)
        + "\n\n# Compiler summary (optional context)\n"
        + str(cs)
        + "\n\n# Relevant skill names\n"
        + names_lines
        + "\n\n# Existing draft (read-only; do not replace)\n"
        + body
        + "\n\nWrite `<Refiner_Addendum>` only. If the draft already covers everything, output an empty addendum: `<Refiner_Addendum></Refiner_Addendum>`."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_overall_procedure_code_prompt(task, overall_procedure, procedure_code_template):
    template = "You are a Senior Software Engineer specializing in AI-driven automation and safe code generation."
    recompile_note = ""
    if task and ("[Runtime Recompile]" in task or "PROCEDURE PATCH MODE" in task):
        recompile_note = """
6.  **Patch / Recompile Runs**: If the task text includes a runtime recompile header, treat `overall_procedure` as a **small ERRATA + NEXT STEPS** block. Inject it as **one** concise `messages.append` (do not paste duplicate multi-page guidance). Preserve the template loop; do not nest multiple full copies of the same procedure.
"""
    prompt = f'''
You are given a complex task description, overall procedural guidance, and a strict procedure code template.
Your task is to implement a Python function `overall_procedure_code` that follows the provided procedure code template and incorporates the overall procedure guidance to help Agent solve a complex task.

# Principles
1.  **Adherence to Template**: Strictly follow the provided procedure code template. Input arguments and return types must align with the template. (Most Important)
2.  **Inject Overall Procedural Guidance**: Help the agent accomplish the task by injecting the overall procedure guidance into the function.
3.  **Environment-agnostic actions**: Do not hard-code benchmark-specific strings beyond what the task states; any example action strings should mirror the agent system prompt's Available Actions style.
4.  **Error Handling**: Implement robust error handling for LLM calls only. Do
    **not** wrap `env.step(...)` in `try/except Exception`, because the runtime
    recompile controller intentionally raises control-flow exceptions from
    `env.step(...)` and the outer executor must catch them.
5.  **Action Boundary**: Skills are not environment tools. Never send skill names,
    `[invoke ...]`, `use/call/trigger skill`, `abort`, `done`, `report failure`, or
    placeholder actions like `-` to `env.step(...)`. Skill guidance must be converted
    into a concrete legal environment action before stepping.
6.  **Stateful Local Repair**: On failed actions, repair the smallest local step from
    the latest observation. Preserve completed subgoals, avoid restarting solved phases,
    and avoid repeating the exact same failed action without refreshing state or changing
    the candidate/precondition.
7.  **Protocol Memory**: Maintain compact local memory for tried candidates/queries,
    current carried object, completed placements/transforms, and open/closed or selected
    targets when the template permits ordinary Python variables. Use it to avoid loops
    and to continue one-at-a-time workflows.
8.  **Search and Transform Repair**: If search fails, broaden the query or move to the
    next candidate instead of repeating. If heat/cool/clean or a similar transform fails,
    repair missing preconditions before switching goals. For `verb {{obj}} with {{tool}}`
    action spaces, keep the object held/available for the transform instead of inventing
    a load/place-inside subroutine.
9.  **List-to-Inspect Candidate Commitment**: In tasks with discovery/list/search
    phases, treat surface labels as incomplete evidence. Track best-so-far candidates
    and inspect the strongest partial candidate after limited horizontal exploration or
    one relaxed query; verify state/details/options before rejecting it. Never terminate
    with an empty action when legal candidate or commit actions remain.
10. **Silent Commit Actions**: Some legal selection/commit actions do not visibly change
    the observation text. If a different legal option/action was just selected and no
    explicit error appears, proceed to the next required selection or final commit instead
    of treating the unchanged observation as failure.
11. **Explicit Option Commitment**: If the task specifies an explicit attribute/state and
    the current observation exposes it as a legal selectable option/action, select/apply it
    before the final commit/delivery/placement/action. Do not skip the option just because
    a prior query, label, or observation already mentioned it.
12. **Evidence Staging**: On discovery/list/search pages, filter only by coarse identity/type
    and directly visible hard constraints. Do not reject plausible candidates because
    fine-grained attributes or option-like values are absent from surface text; inspect
    state/details/options first, then reject only with fresh contradictory evidence.
13. **Reward Field Safety**: When reading score/reward metadata from `env.step(...)`,
    use safe `dict.get(...)` access after confirming `info` is a dict. For scalar scores,
    use `info.get("score", task_reward)`. For list-style success flags, use
    `info.get("won", [reward])`. Never use `info["score"]`, `info['score']`,
    `info["won"]`, or `info['won']`, because some environment or abort paths omit them.
14. **Done Field Safety**: Environment done flags may be either a scalar boolean or a
    single-element list/tuple depending on whether the raw environment or a runtime proxy
    is executing the generated code. Normalize with an `isinstance(..., (list, tuple))`
    check before indexing; never assume `task_done[0]` is always valid.
15. **Function Naming**: Name the function `overall_procedure_code` as per the template.
{recompile_note}

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
