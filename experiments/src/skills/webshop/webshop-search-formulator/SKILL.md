---
name: webshop-search-formulator
description: This skill generates effective search keywords based on parsed product criteria. It is triggered after query parsing, when the agent needs to perform an initial product search on an e-commerce platform. The skill takes structured attributes (e.g., 'women's size 5 patent-beige high heel') and produces a concise, platform-appropriate search string designed to return relevant results, balancing specificity with recall.
---
# Skill: Webshop Search Formulator

## Purpose
You are an expert at formulating the initial search query for an e-commerce product search. Your goal is to translate a structured set of product requirements into a concise, effective search string that will yield relevant results on a platform like Amazon.

## Action-Space Boundary
This skill is reasoning guidance, not a WebShop UI button. Do **not** output
actions such as `click[webshop-search-formulator]` or `activate
webshop-search-formulator`. Convert the selected keywords into a valid
`search[...]` action.

## Core Workflow
1.  **Input:** You receive a parsed query containing key product attributes (e.g., category, type, size, color, material, price limit).
2.  **Process:** Analyze the attributes to identify the most critical, distinguishing features for the initial search. Prioritize attributes that will filter the results meaningfully without being overly restrictive.
3.  **Output:** Generate a single, well-formatted `search[keywords]` action string.

## Key Principles for Search Formulation
*   **Balance Specificity & Recall:** Start with a moderately specific query. Including 2-3 core attributes (e.g., `size 5 patent-beige high heel`) is better than a single generic term (`high heel`) or an overly long list of all attributes.
*   **Prioritize Distinctive Attributes:** Favor attributes that uniquely identify the product variant (e.g., "patent-beige", "size 5") over very common ones (e.g., "women's") in the initial search.
*   **Use Natural Keyword Order:** Place the most important or specific terms first. Mimic how a user might type the query.
*   **Exclude Non-Searchable Filters:** Do not include filters typically applied *after* the search (e.g., price ranges like `< $90`) in the initial keyword string. These are for later refinement.
*   **Standardize Formatting:** Use lowercase, avoid special characters, and separate keywords with spaces.

## EvoSkill Recovery Policy: Staged Search and Commit Budget
For long shopping instructions, use a staged search plan instead of a single
overloaded query.

1. Start with product type plus the two or three most distinctive searchable
   attributes. Do not include price, care instructions, or every material word in
   the initial search.
2. If the first page is weak, issue one revised search that preserves the
   product type and rare attributes while dropping noisy adjectives.
3. Keep a running best candidate across result pages. Track which hard
   constraints it satisfies and which soft constraints still need detail-page
   verification.
4. Do not spend the full budget on pagination. After two result pages, click the
   best candidate seen so far if it is within the displayed price budget or has a
   price range whose low end is within budget.
5. Only conclude "no match" after at least one plausible candidate has been
   inspected on its detail page or all visible candidates fail a hard constraint.

## Example from Trajectory
**Parsed Instruction:** `woman's us size 5 high heel shoe with a rubber sole and color patent-beige, and price lower than 90.00 dollars`
**Effective Search:** `search[size 5 patent-beige high heel]`
**Rationale:** "size 5" and "patent-beige" are the most specific, distinguishing attributes. "high heel" defines the product type. "rubber sole" and price filter are omitted from the initial search to avoid prematurely limiting potentially valid results.

## Action Format
Your final output must be a single action in the exact format:
