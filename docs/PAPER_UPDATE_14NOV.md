# Research paper: edits to apply (results of 12–14 Nov 2026)

**Status: applied to the paper on 14 Nov 2026**, together with the answer-first and saved-cache results. This file is kept as a record.

Paper: "An Adaptive LLM, RAG, Multimodal and Voice-Enabled Framework for Resource-Efficient, Evidence-Grounded Educational Question Answering" (Claude Docs). The Docs connector was unavailable on 14 Nov, so these edits are written out here. Figures come from `evaluation/results/` and `docs/PROGRESS.md` §9e-2 to §9g.

## Abstract: add after the sentence on citation use

> On a laptop CPU with no GPU, the full 140-question benchmark ran without a single failure; 84% of grounded answers cited their sources and all 30 out-of-scope questions were kept ungrounded. Four further mechanisms each improved a measured weakness:
> - conversation memory raised correct retrieval for follow-up questions from 0.3% to 97.3%;
> - agentic retrieval raised evidence coverage for comparisons from 22.5% to 75%;
> - context engineering cut answer time by 34%;
> - exact reasoning tools for image readings raised fully correct image answers from 4/8 to 8/8.

## 1. Introduction: add a contribution to the numbered list

> **Agentic, budgeted retrieval with exact tools.**
> - A bounded plan–retrieve–critique–refine–correct loop over one search tool and an evidence-coverage critic.
> - Token-budgeted context construction, and conversation memory that resolves follow-ups.
> - A semantic answer cache with a content-word guard.
> - Deterministic tools (a truth-table gate checker and a sandboxed snippet interpreter) whose results are handed to the model as checked facts.

## 3. The framework: new subsection 3.7 "Agentic retrieval, context and memory"

> **Agentic retrieval.** Single-shot retrieval embeds the whole question once. A comparison then lands between its two topics, and a multi-part question retrieves for its dominant part only. The agent instead:
> 1. **plans** sub-queries for comparisons and multi-part questions;
> 2. **retrieves** for each sub-query, with scope widening, and interleaves the results;
> 3. **critiques** the evidence by its coverage of the question's key terms;
> 4. **refines**, when coverage is below 0.6, by searching for the missing terms, in at most three steps;
> 5. **corrects** after verification, by fetching evidence for each unsupported claim before the single permitted revision.
>
> Decisions are rules and measurements, not extra model calls: on a CPU every call costs tens of seconds.
>
> **Context engineering.** Sources are budgeted per answer mode: 450, 750 and 1,100 tokens for 2-, 5- and 10-mark answers. Near-duplicate passages are removed, and each passage is compressed to its first sentence plus the sentences most similar to the question, in their original order. Only the prompt copy is compressed: verification checks claims against the full passages. The fixed grounding rules are placed in the system message, so every prompt starts with an identical prefix that the inference server reuses from its cache.
>
> **Memory.**
> - Working memory keeps the recent turns as key sentences within 500 tokens.
> - An episodic line names the topics of older turns.
> - A follow-up ("What are its advantages?") is rewritten into a self-contained search query.
> - Resolved in-domain follow-ups are routed to RAG.
>
> **Answer cache.** A repeated question is answered from a shared cache in two tiers:
> - an exact tier, after normalisation;
> - a semantic tier, at cosine ≥ 0.97 with an **identical set of content words**, so "AC motor" and "DC motor" never match (cosine ≈ 0.97).
>
> The cache is LRU with a 24-hour expiry. It never serves image, document or follow-up questions, nor answers that carried warnings.

## 3.5 Verification: add a paragraph "Exact tools for image readings"

> The 3B vision model transcribed images accurately but its reasoning over the transcription was unreliable. A truth table read exactly as 00→0, 01→0, 10→0, 11→1 was called a NOT gate, and a loop `total += i for i in range(1, 6)` was said to print 21. Two deterministic tools now run on every image reading, and their results are appended as CHECKED FACTS:
> - a gate checker tests 0/1 tables against AND, OR, NAND, NOR, XOR, XNOR, NOT and buffer, for any number of inputs;
> - a ~100-line interpreter runs short snippets. It supports only literals, variables, arithmetic, comparisons, if/for/while and a few built-ins. It has no imports, attributes, function definitions or I/O, and stops at 10,000 steps.

## 6. Results: new subsection 6.6 "CPU-only operation and the new mechanisms"

| Measure | Baseline | With the mechanism |
|---|---|---|
| Full benchmark, 140 questions, laptop CPU (adaptive) | – | 0 failures; 84% of grounded answers cite; 30/30 out-of-scope kept ungrounded; median 125 s |
| Follow-ups, right topic in the top 5 (300 cases) | 0.3% | 97.3% |
| Comparisons, evidence for both sides (40 pairs) | 22.5% | 75% |
| Prompt source tokens, median 2 / 5 / 10-mark | 1,020 | 399 / 644 / 874 |
| Best relevant passage kept in the prompt | – | 100% |
| End to end, same 10 cases, real model: normal questions | 84 s, 1,226 tokens | 56 s, 824 tokens |
| End to end: comparisons covering both sides | 1/3 | 3/3 |
| End to end: follow-ups answered from the topic's sources | 0/3 | 3/3 |
| Repeated question | 9.1 s | 0.0 s |
| Cache false hits between different questions | – | 0 / 4,950 pairs |
| Images fully correct, adaptive route (8 known-answer images) | 4/8 (75% of facts) | 8/8 (100%) |
| Speech recognition, synthetic Indian-English voices | – | 4.3% WER; 1.4 s median on the CPU |

## 7. Measured failures and fixes: add rows to the table

| Run | Failure observed | Fix | Evidence after the fix |
|---|---|---|---|
| Ablation | Memory resolved follow-ups, but the 2-mark shortcut answered them ungrounded | Router rule: in-domain follow-ups go to RAG | 0/3 → 3/3 grounded |
| Tests | Cache matched "AC motor" with "DC motor" (3-letter word minimum) | Every content word must match, short ones included | 0 false hits |
| CPU images | Correct transcription, wrong reasoning (truth table, code) | Exact tools as checked facts | 4/8 → 8/8 |
| CPU run | Read-aloud failed without internet (both TTS engines online) | Offline system voice as the last fallback | Speech without network |

## 8. Discussion: replace the limitation "Verification checks support and written arithmetic, not reasoning"

> **Verification checks support, written arithmetic, truth tables and short code, not open reasoning.** A faulty reasoning step with nothing to recompute still passes. The exact-tool pattern (compute rather than trust) extends to further checkable forms.

## 9. Conclusion: future work, replace the item on 7B models

> Compare the 7B vision model on the same image set (it needs about 8 GB of free memory or a GPU), and extend exact tools to unit analysis of formulas and to circuit equations.
