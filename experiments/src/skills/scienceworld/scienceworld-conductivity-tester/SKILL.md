---
name: scienceworld-conductivity-tester
description: Determines if an object is electrically conductive by integrating it into a circuit and observing a light bulb. Triggered when classifying an object based on conductivity.
---
# Instructions

## Objective
Classify a target object as **electrically conductive** or **non-conductive** by using it to complete a simple circuit with a battery and a light bulb.

## Core Procedure
1.  **Locate & Acquire Object:** Teleport to the object's location and pick it up.
2.  **Focus on Object:** Use the `focus` action on the object to signal intent.
3.  **Setup in Workshop:** Teleport to the `workshop`. Place the object down.
4.  **Inspect Available Components:** Use `look around` and identify the exact battery, wires, indicator bulb or buzzer, and answer boxes that are actually present in the current workshop variation.
5.  **Construct Test Circuit:** Build the test circuit using the exact component names you observed. Prefer this proven series topology:
    a. Identify:
       - `battery anode` and `battery cathode`
       - a first wire with `terminal 1` and `terminal 2`
       - an indicator component with `anode` and `cathode`
       - a second wire with `terminal 1` and `terminal 2`
       - the target object's `terminal 1` and `terminal 2`
    b. Connect `battery anode` to the first wire `terminal 1`.
    c. Connect the first wire `terminal 2` to the indicator `cathode`.
    d. Connect the indicator `anode` to the second wire `terminal 1`.
    e. Connect the second wire `terminal 2` to the target object's `terminal 1`.
    f. Connect the target object's `terminal 2` directly to `battery cathode`.
    g. Every `connect` action must use the exact observed contact-point syntax, not abstract object-to-object connections.
6.  **Observe & Classify:** After the circuit is complete, use `look around` or `look at <indicator>` to inspect the indicator state.
    - If the indicator is **on/activated**, the object is **conductive**.
    - If the indicator is **off/not activated**, the object is **non-conductive**.
7.  **Place in Correct Answer Box:** Use the task description to decide which observed answer box corresponds to conductive vs. non-conductive, then move the object into that box.

## Key Notes
- Do not assume specific wire colors, bulb colors, or box colors unless they are explicitly present in the current observation.
- The exact circuit layout can vary by task variation; ground every connection in the components actually visible in the workshop.
- The observation after the circuit is built is critical for determining the indicator state.
- This skill assumes the environment's action set and room structure as provided in the trajectory.
