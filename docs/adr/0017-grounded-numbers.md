# ADR 0017: Every number in an answer must be grounded in executed results

- Status: Accepted
- Date: 2026-11-22

## Context

The core principle of the product is that the LLM never produces numbers. Even when the SQL is right, the explanation can misstate, round wrongly or invent a figure.

## Decision

The explainer returns answer text plus a list of key numbers with the result column each came from. `analyst/pipeline/grounding.py` extracts every number from the text and matches it against result values, the question and stated assumptions, allowing rounding and common formats (1,234.5; 12%). Any unmatched number triggers one regeneration; if that also fails, the user gets a template answer ("Here is the result of the query") with the table. Derived values (differences, ratios) must come from a query, not from the model.

## Consequences

- Zero ungrounded numbers reach users (ADR 0013).
- Some valid answers may fall back to the template; the fallback rate is tracked in the benchmark.
- Numbers spelled as words from eleven up are refused by the explainer, because the check cannot read them. Words up to ten are not looked for, so a made-up "three" is a known gap that the prompt's request for digits narrows; the benchmark will show if it matters.
- The check runs on the text after links, images and addresses are removed from it, so what is checked is what is shown.
