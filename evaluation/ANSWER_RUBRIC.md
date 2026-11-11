# Answer-quality rubric (report §22.2 "Answer quality")

Grade each row of `results/grading_sheet_*.csv` without looking at
`grading_key_*.json` (it maps the system codes S1–S5 to strategies).
Use at least two raters and report agreement (e.g. Cohen's kappa) per criterion.

| Score | Correctness | Completeness (for the requested mark level) | Relevance | Clarity |
|---|---|---|---|---|
| 3 | No factual errors | Covers everything expected at this mark level | Fully on the question | Well organised, correct terminology |
| 2 | Minor imprecision, nothing misleading | Misses one expected point | Mostly on topic | Understandable, small issues |
| 1 | At least one clear error | Misses several expected points | Partly off topic | Hard to follow |
| 0 | Mostly wrong or invented | Does not answer | Off topic | Unreadable |

Rules
- Judge correctness against the cited sources **and** standard textbook knowledge; note
  in *notes* when an answer is correct but its citation does not support it.
- Do not reward length: a 10-mark answer padded with unsupported material scores
  lower on correctness, not higher on completeness (report §15).
- Out-of-domain questions: an answer that correctly says the knowledge base does not
  cover the topic and then answers carefully is acceptable; invented citations score 0
  on correctness.
- Record failures/timeouts as 0 on all criteria rather than skipping them (§22.3).
