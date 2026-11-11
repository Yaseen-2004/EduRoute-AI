# Implementation and Evaluation Results

*Draft chapter for the EduRoute AI project report. All numbers are measured;
the result files are named in each section and stored in `evaluation/results/`.
Section 8 (answer quality) is completed once the human grading is done.*

## 1. What was built

EduRoute AI is an adaptive study assistant for B.Tech students. For each
question it decides which processing route to use:
- **direct:** the language model alone;
- **RAG:** retrieval from a knowledge base, then generation with cited sources;
- **vision:** an image question answered by a vision model;
- **hybrid:** image analysis plus retrieval.

It verifies answers selectively. The language model is Qwen2.5-3B-Instruct
(4-bit, served by Ollama) and the vision model is Qwen2.5-VL-3B.

| Component | Implementation |
|---|---|
| Knowledge base | 1,425 Wikipedia articles covering 26 branch groups and 289 subjects; 35,567 chunks of up to 220 tokens; MiniLM-L6 embeddings; FAISS inner-product index |
| Router | Query analysis plus a retrieval probe whose in-domain threshold is calibrated on labelled questions |
| Verification | Claim extraction; evidence mapping (semantic and lexical); figure checks; an NLI contradiction check for claims with figures; at most one revision |
| Interfaces | Streamlit chat with branch and subject filters, 2/5/10-mark answer modes, image upload, voice input (faster-whisper) and spoken answers (Edge-TTS) |
| Privacy | Questions logged only as a hash; images and recordings not stored; 30-day log retention |

## 2. Knowledge-base construction

Topics from the subject catalogues were resolved to Wikipedia articles: 1,370
of 1,370 resolved, 1,352 of them by exact title or a curated hint. Every
resolution is recorded with its score and the rejected candidates, for audit.

Auditing the earlier AIML-only version found 7 of its 92 articles to be wrong
pages: for example "CNN" (the news channel) and several disambiguation pages.
Reference sections had also leaked into chunks. Both defects were corrected
before the engineering-wide build.

**Coverage against the AICTE model curriculum.** The catalogue was compared
with seven AICTE model-curriculum documents:
- AICTE provides a model curriculum for 14 of the 26 groups (167 subjects). **116 of those subjects (69%) match or partly match an AICTE course.**
- 12 groups (122 subjects) have no AICTE model curriculum and need a university syllabus.
- The comparison lists 64 AICTE core courses absent from the catalogue (for example in IT and Robotics). These are candidates for the next corpus update.

## 3. Retrieval

There are two benchmarks of 50 questions each, labelled before the
engineering index existed:
- **AIML:** 50 questions covering all 13 AIML subjects.
- **Engineering:** 50 questions, 2 from each non-AIML group.

A retrieved chunk counts as relevant if it comes from a listed article and
contains a listed evidence term.

| Index / configuration | Benchmark | Success@1 | Success@5 | MRR@10 |
|---|---|---|---|---|
| AIML V3 (proposal baseline) | AIML | 0.80 | 0.92 | 0.855 |
| AIML V4 | AIML | 0.90 | 1.00 | 0.947 |
| Engineering-wide | AIML | 0.90 | 0.98 | 0.939 |
| Engineering-wide | Engineering | 0.90 | 1.00 | 0.950 |
| Engineering-wide, subject filter with fallback | Engineering | 0.92 | 1.00 | 0.960 |

Findings:
- **The larger index barely hurts:** growing the knowledge base twelvefold reduced AIML MRR only from 0.947 to 0.939.
- **Two options did not help:** hybrid BM25 + dense retrieval and a cross-encoder reranker were evaluated and neither improved results (the reranker cost about 120× the latency), so dense retrieval is the default.
- **A strict subject filter made some covered questions ungrounded,** because catalogue subjects overlap. The system therefore widens the search from subject to branch to the whole index when the selected subject has no close match.

## 4. Router

The in-domain threshold (0.491) was fitted on 100 in-domain and 30
out-of-domain questions (AUC 0.999). It was then tested **unchanged** on 60
new questions:

| Question set | Correct |
|---|---|
| In-domain questions, 30 across all branches | 29 / 30 accepted |
| Easy out-of-domain questions (10) | 10 / 10 rejected |
| Hard out-of-domain questions, 16 (history, law, economics, medicine, arts) | 15 / 16 rejected |
| **Overall (4 borderline questions excluded)** | **96.4%** |

The one rejected in-domain question, about a phase-locked loop, has no article
in the knowledge base, so it is a coverage gap rather than a routing error.

## 5. System benchmark: five routing strategies

**Setup:**
- 50 questions: 20 in-domain, 20 out-of-domain and 10 mixed (comparison, numerical, procedure, evidence requests, chat). The same seeded sample was used in every run.
- Each question was answered by five strategies on a Colab T4 GPU: 250 answers.
- All answers were real model output, with 0 failures.
- File: `system_benchmark_20261111_042423.json`.

| Strategy | Median latency | Input tokens / answer | Total tokens | Grounded | Verified | Cites sources | Unsupported claim rate ¹ |
|---|---|---|---|---|---|---|---|
| direct | 3.2 s | 185 | 24.1k | 0% | – | 0% | – |
| fixed_rag | 4.0 s | 1,247 | 79.5k | 100% | – | 86% | – |
| modality | 4.1 s | 1,247 | 79.8k | 100% | – | 82% | – |
| always_verify | 7.8 s | 2,237 | 137.3k | 100% | 100% | 78% | 0.41 |
| **adaptive** | **3.9 s** | **898** | **64.1k** | 48% | 28% | 42% ² | 0.13 |

¹ Mean over verified answers, measured after any revision.
² Adaptive answers out-of-domain questions without retrieval, so they carry no
citations. On in-domain questions, 75% of its answers cite sources.

**Interpretation:**
- **Cost.** Adaptive used **19% fewer tokens than fixed RAG** and **53% fewer than always-verify**, at the same median latency as fixed RAG.
- **Routing behaved as designed.**
  - All 20 out-of-domain questions were answered directly and labelled as not grounded in the knowledge base.
  - 17 of 20 in-domain questions used retrieval.
  - The other 3 were 2-mark definitions, sent direct on purpose to save cost.
- **Grounding everything is not safer.** Fixed RAG reports every answer as "grounded", including out-of-domain questions, where the retrieved passages are irrelevant. When always-verify checked those answers, the unsupported claim rate was **0.76**, against 0.13 in-domain.

## 6. Defects found by measurement, and their fixes

Each benchmark run exposed problems that unit tests had not caught. They are
listed here because the fixes, and the evidence for them, are part of the result.

| Run | Problem found | Fix | Effect in the next run |
|---|---|---|---|
| 8 Nov | Answers cited sources rarely (14–16%) | Citation rule repeated at the end of the prompt | 70–86% of grounded answers cite |
| 8 Nov | A wrong figure was hidden as "partial", giving an unsupported claim rate of 0.0 | Figures not found in relevant evidence make a claim unsupported, and the student is warned | Wrong figures now flagged |
| 10 Nov | The NLI check flagged correct sentences (about 3 real errors in 35 flags) | Contradiction counted only for claims with figures, at probability ≥ 0.95 | Counted contradictions fell from 12% to 2% (adaptive) |
| 10 Nov | 7% of answers echoed prompt instructions | Echo filter plus a prompt rule | 0 echoes |
| 10 Nov | Formulas and sentence fragments were checked as claims | Fragment filter | – |
| 11 Nov | A failed image analysis led to a guessed answer ("XOR gate") | Questions that refer to the image get an explicit "could not analyse" message | – |
| 11 Nov | Hybrid image answers refused to use the image | Image rules in the prompt | The amplitude is now read from the image |

A larger NLI model (`nli-deberta-v3-base`) was also tested. It was *worse*: it
missed both real errors, so the small model, with the rule restricting it to
claims containing figures, was kept.

## 7. Multimodal and voice

**Image questions.** Eight test images were generated with exactly known
answers: a truth table, block diagram, signal plot, bar chart, equation,
series circuit, code snippet and flowchart.

| Route | Images fully correct | Median latency |
|---|---|---|
| Vision model only | 6 / 8 | 2.7 s |
| Vision + knowledge base | 5 / 8 (before the hybrid prompt fix) | 9.5 s |

Both routes read the bar chart, equation, circuit (30 Ω, 0.4 A), code and
flowchart correctly. The 3B vision model misread the AND truth table in every
run, and misread or miscalculated the signal frequency (50 Hz). For questions
whose answer is in the image, the vision-only route is faster and at least as
accurate.

**Voice.** Twenty benchmark questions were read aloud by two Indian-English
synthetic voices and transcribed by faster-whisper `base.en`:
- word error rate **5.7%**, with 65% of transcripts exact;
- median 1.6 s per question.

Most errors are spelling-level, such as "Homan" for Hohmann. Recordings of
student voices are being collected for the definitive measurement.

## 8. Answer quality (human evaluation): *to be completed*

Two raters grade the 250 answers blind, using the system codes S1–S5, on four
0–3 criteria: correctness, completeness, relevance and clarity (rubric in
`evaluation/ANSWER_RUBRIC.md`). Agreement is reported as quadratic-weighted
Cohen's kappa. *Insert the table produced by
`python -m evaluation.analyse_grading` here.*

## 9. Limitations and threats to validity

- **Labels:** the benchmark labels and subject catalogues were drafted during development. They are being reviewed by the team, with the AICTE comparison as a guide for 14 groups.
- **Sample sizes:** the system benchmark uses 50 questions with one generation each, and the image benchmark uses 8 synthetic images. Differences of a few percentage points are within run-to-run variation.
- **Verification limits:** the verifier checks support and figures, not reasoning. An arithmetic error that uses figures found in the sources passes (for example "period 20 ms → 1 Hz"). Claim-level human labels are being collected to calibrate it.
- **Router threshold:** fitted on the benchmark questions. The held-out check (96.4%) reduces, but does not remove, this optimism.
- **Hardware:** results were measured on a T4 GPU. On the development laptop's CPU, generation ran at about 1–4 tokens per second, too slow for interactive use, so the deployment serves the models from a GPU.
