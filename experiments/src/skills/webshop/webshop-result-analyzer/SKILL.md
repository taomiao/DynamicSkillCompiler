---
name: webshop-result-analyzer
description: This skill evaluates a list of search results against the target product criteria. Trigger it when a search result page is observed, to identify promising product listings. It analyzes each result's title, price, and brief description to shortlist items that best match the required attributes (like size, color, and price), outputting a prioritized list of candidate product IDs or links for further inspection.
---
# Instructions

Trigger this skill when you observe a search result page (e.g., containing "Page 1 (Total results: 50)" and multiple product listings).

## Action-Space Boundary
This skill is reasoning guidance, not a WebShop UI button. Do **not** output
actions such as `click[webshop-result-analyzer]`, `run webshop-result-analyzer`,
or `activate webshop-result-analyzer`. Apply the analysis mentally, then output
a valid WebShop action such as `search[...]`, `click[<product_id>]`,
`click[Next >]`, `click[< Prev]`, an option click, or `click[Buy Now]`.

## 1. Extract User Requirements
First, parse the user's instruction from the observation. Identify the following key attributes:
- **Product Type:** (e.g., "woman's us size 5 high heel shoe")
- **Specific Attributes:** (e.g., "rubber sole", "color patent-beige")
- **Price Constraint:** (e.g., "price lower than 90.00 dollars")

## 2. Analyze Search Results
For each product listing in the observation (typically formatted as `[ASIN/Product ID] [SEP] [Title] [SEP] [Price Range]`):
1.  Extract the **Product ID** (e.g., B09GXNYJCD).
2.  Extract the **Product Title**.
3.  Extract the **Price**. Convert any range (e.g., "$49.99 to $54.99") to its maximum value for comparison against the budget.
4.  Perform a **textual match** between the title/description and the required attributes (size, color, material like "rubber", product type).

## 3. Score and Prioritize
Use the bundled Python script `analyze_results.py` to perform a consistent, deterministic analysis.
1.  Use the extracted user requirements and the list of product data to build a
    prioritized list of candidate Product IDs, sorted by a match score.
2.  If a helper script is available outside the WebShop UI, it may be used by
    the host system. The agent must still output only valid WebShop UI actions.

## EvoSkill Recovery Policy: Do Not Abstain on a Results Page
When a results page contains visible products, do not end with `Action: none`
unless every visible product violates a hard constraint that cannot be checked
later. Treat the search-results page as a candidate-ranking step:

1. Separate **hard constraints** from **soft constraints**.
   - Hard constraints: product category/type, explicit maximum price, required
     size or variant when shown on the page, and explicit color when shown.
   - Soft constraints: style adjectives, care instructions, material words, fit
     descriptors, and title keywords that may need detail-page verification.
2. Rank products by:
   - hard-constraint compatibility,
   - number of soft constraints matched in the title,
   - lower verified or displayed price,
   - product type similarity.
3. If there is no exact match, click the best partial candidate before
   exhausting the step budget. Use the detail page to verify missing attributes
   instead of repeatedly concluding that no match exists.
4. If the current page has weak matches, keep the best candidate seen so far.
   After at most two result pages, inspect that best candidate rather than
   continuing to paginate.

## 4. Output and Next Action
Present the analysis in this format:
