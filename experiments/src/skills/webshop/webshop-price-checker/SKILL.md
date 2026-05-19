---
name: webshop-price-checker
description: This skill scans product listings or detail pages to verify if an item's price meets a specified budget constraint. It is triggered when evaluating a potential product for purchase. It extracts the price from the displayed information (e.g., '$164.95') and compares it against the maximum allowed budget, outputting a boolean decision on whether the item is within budget.
---
# Instructions
When evaluating a product for purchase, use this skill to check if its price is within the specified budget.

## Action-Space Boundary
This skill is reasoning guidance, not a WebShop UI button. Do **not** output
actions such as `click[webshop-price-checker]`, `run webshop-price-checker`, or
`activate webshop-price-checker`. Inspect the displayed price, reason about the
budget, then output a valid WebShop action such as clicking a product, variant,
navigation control, or `click[Buy Now]`.

## Process
1.  **Identify the Budget:** Extract the maximum allowed price from the user's instruction or the current context. The budget is typically expressed as a dollar amount (e.g., "lower than 200.00 dollars").
2.  **Locate the Price:** On the current web page (search results or product detail page), find the price string. It is usually prefixed with a `$` symbol and may be labeled "Price:".
3.  **Execute Check:** Compare the extracted price string with the budget. If a
    host-side `check_price.py` helper is available, it may be used outside the
    WebShop UI, but never as a WebShop action.
4.  **Make Decision:** Based on the script's boolean output (`True`/`False`):
    *   If `True`: The item is within budget. You may proceed with further evaluation or purchase.
    *   If `False`: The item exceeds the budget. You should continue searching or select a different item.

## Notes
*   The primary action for this skill is to call the bundled script. Do not perform manual price parsing or comparison in your main reasoning.
*   If the price cannot be found on the page, assume the item is not suitable and continue your search.

## EvoSkill Recovery Policy: Search-Result Price Ranges
Search results often display ranges such as `$1.99 to $8.99`.

1. For search-result triage, use the **lowest displayed price** as a provisional
   signal. A range passes triage if its low end is below the user's maximum
   price.
2. For final purchase, use the selected variant's detail-page price as the hard
   check. Do not buy if the selected variant exceeds the budget.
3. Keep price as a hard final constraint, but do not reject a result-page
   candidate solely because the high end of a range exceeds the budget.
4. If several candidates are under budget, prefer the one that satisfies more
   non-price constraints before choosing the cheapest option.
