# EduRoute AI — implementation progress against the Detailed Project Report

Status as of 7 November 2026. Every number below was measured on this machine
(Windows, CPU only). The result files are named so they can be re-checked.

## 1. Updated development status (replaces report §23)

| Module | Report status | Now | Evidence |
|---|---|---|---|
| Knowledge base (AIML) | V3: 92 articles, 2,410 chunks | **V4: 96 articles, 3,014 chunks** | `data/metadata/aiml_v4_preparation_report.json`, `aiml_index_report_v4.json` |
| Knowledge base (all engineering) | AIML only; other branches "future" | **engineering_v1: 26 groups, 289 subjects, 1,425 articles, 35,567 chunks.** This is now the app's default index | §7 below; `data/metadata/engineering_corpus_report_v1.json`, `engineering_index_report_v1.json` |
| Retrieval refinement (§19.1) | In progress | **Done.** The Transformer bibliography chunk and the SHAP ranking are fixed, and a 50-question labelled benchmark is in place | `evaluation/results/retrieval_eval_*.json` |
| Hybrid / reranking | – | Implemented and evaluated. **Neither helps**, so both are off by default | §3 below |
| LLM generation (§9) | Pending | **Implemented:** Qwen2.5-3B via an OpenAI-compatible server (Ollama/vLLM) or in-process transformers, with grounded prompts, citations and 2/5/10-mark modes | `src/llm/` |
| Adaptive router (§8) | Pending | **Implemented:** query analysis, calibrated retrieval probe, 5 strategies, decision logging | `src/router/`, `config/router_calibration_<index>.json` |
| Selective verification (§14) | Pending | **Implemented:** claim extraction, evidence mapping, number/citation checks, triggers, one revision pass | `src/verification/` |
| Vision / multimodal RAG (§12) | Pending | **Implemented (code):** image validation, Qwen2.5-VL client, vision and hybrid workflows. **Not yet run against a model** | `src/vision/` |
| Voice (§13) | Pending | **Implemented and tested:** faster-whisper ASR with transcript correction, Edge-TTS with gTTS fallback | `src/voice/speech.py` |
| PDF ingestion (§6) | Pending | **Implemented:** pymupdf4llm extraction, section cleaning, mandatory licence/provenance | `src/data_collection/pdf_ingestion.py` |
| Streamlit app (§20) | Pending | **Implemented:** chat, subject filter, answer modes, image/voice input, citations, verification, developer view, feedback | `app.py` |
| Telemetry (§22.2) | Pending | **Implemented:** per-stage latency, tokens, model calls; question text not logged by default (§21) | `outputs/logs/` |
| Formal evaluation (§22) | Pending | **First run done** with Qwen2.5-3B on a Colab T4: 250 answers, 0 failures. Cost and routing measured; quality awaits human grading | §8; `evaluation/results/system_benchmark_20261108_161042.json` |
| Tests | – | 65 unit tests plus 1 headless app test (run against engineering_v1), all passing | `tests/` |

## 2. Data-quality defects found in V3 (not in the report)

While auditing V3, we found that 7 of the 92 articles were the wrong Wikipedia
page. They were indexed and retrievable:

| V3 title | What it actually was | V4 replacement |
|---|---|---|
| CNN | Cable News Network (29k characters of TV-news history) | removed; *Convolutional neural network* already present |
| Lime | Disambiguation page (the fruit, the material) | covered by *Explainable AI* and *Feature Importance* (Wikipedia has no LIME article) |
| Spark | Disambiguation page | *Apache Spark* |
| RNN | Disambiguation page | *Long short-term memory* (+ existing *Recurrent neural network*) |
| Regression | Disambiguation page | *Regression analysis* |
| Clustering | Disambiguation page | *Cluster analysis* |
| Transparency | Disambiguation page (optics) | *Algorithmic transparency* |

We also added *Statistical classification*, *Gradient descent*, *Attention
(machine learning)*, *Lexical analysis* (tokenization), *Image segmentation*
and *Shapley value* (SHAP) to fill catalogue topics that had no article. The
collector now refuses disambiguation pages.

Other V3 defects fixed in V4:

- **Reference sections leaked into chunks.** V3 truncated text at the first reference heading. That missed "See also", which appears in 73 of 92 articles. It also skipped the two articles integrated after preparation. That caused the report's Transformer issue: a bibliography chunk ranked first. V4 parses the section structure and drops non-content sections wherever they occur (76 removed).
- **Chunk text was degraded.** V3 rebuilt chunk text by decoding token IDs. That lowercased everything and left WordPiece artifacts ("##ns", "dot - product") and `�` characters. This is the text the LLM and the citations would have shown. V4 slices the original text using tokenizer offsets, never cuts mid-word, and has 0 `�` characters.
- **Wrong subject for the Transformer articles.** Both were filed under AIML009 *Big Data Analytics*. They are now under AIML006 *Transformers*.
- **Section metadata.** Each chunk now carries its section path and is embedded with a "Title > Section:" prefix.

## 3. Retrieval evaluation (report §19.1, §22.2)

Benchmark: `evaluation/benchmarks/retrieval_aiml_v1.jsonl`, 50 questions
covering all 13 AIML subjects. A chunk counts as relevant if it comes from a
listed article **and** contains a listed evidence term. Because this
definition doesn't depend on chunk IDs, V3 and V4 are scored with identical
labels. **These labels were drafted during development and must be reviewed
by the team before they are cited in the final report.** They were not
changed after seeing results.

Success@K is the share of questions with at least one relevant chunk in the
top K (the report's Recall@K definition, made exact).

| Config | Questions | S@1 | S@3 | S@5 | MRR@10 | Article@1 | Median latency |
|---|---|---|---|---|---|---|---|
| V3 dense (report baseline) | 50 | 0.80 | 0.90 | 0.92 | 0.855 | 0.88 | 35 ms |
| V3 hybrid | 50 | 0.82 | 0.88 | 0.92 | 0.856 | 0.82 | 36 ms |
| **V4 dense (default)** | 50 | **0.90** | **1.00** | **1.00** | **0.947** | **0.98** | 31 ms |
| V4 hybrid (BM25 + RRF) | 50 | 0.84 | 1.00 | 1.00 | 0.913 | 0.94 | 33 ms |
| V4 dense + cross-encoder rerank | 50 | 0.90 | 0.98 | 1.00 | 0.945 | 0.94 | 2,597 ms |

Reading the results:

- **Coverage.** V3 misses 4 questions outright (least squares, hierarchical clustering, Spark, Shapley) because those articles were disambiguation pages. On the 46 questions both versions can answer, V4 dense scores MRR 0.942 vs 0.929 for V3, and S@3 1.00 vs 0.978. The ranking gain on shared questions is small. Most of V4's improvement comes from correct content.
- **The two report §19 failures are fixed.** Transformer: rank 2 → 1, and the top chunk is now real content, not bibliography. SHAP: rank 2 → 1.
- **Negative results, reported as such.** Hybrid BM25 lowers S@1 from 0.90 to 0.84 on V4. The cross-encoder reranker brings no gain at about 120× the latency. Both remain available as options but are off by default.
- **Known limitation.** On R01 (backpropagation), the *Neural network* article ranks above *Backpropagation*. It is probably relevant but isn't in the labels, so it counts as a miss.

## 4. Router calibration (report §8.2)

The report says a similarity threshold must not be used as confidence without
calibration. `evaluation/calibrate_router.py` compares the top-1 cosine for
the 50 in-domain questions against 40 out-of-domain ones: other B.Tech
branches and general questions (`router_out_of_domain_v1.jsonl`).

| Measure | Value |
|---|---|
| AUC (in-domain vs out-of-domain) | 0.998 |
| In-domain threshold (max Youden J) | 0.512 |
| In-domain questions kept | 96% |
| Out-of-domain falsely accepted | 0% |

**Caveat:** the threshold was fitted on these 90 questions. Confirm it on a
held-out set before reporting router accuracy. Re-run the calibration
whenever the index or embedding model changes.

The table above is for the V4 index. Thresholds are stored per index
(`config/router_calibration_<index>.json`); the engineering index has its own
calibration (§7.4).

## 5. Routing policy (adaptive, balanced)

1. **Image attached:** hybrid multimodal RAG if domain knowledge is needed, otherwise vision LLM. If no vision model is configured, it falls back to RAG on the text and says so.
2. **Small talk / study tips:** direct LLM.
3. **Explicit request for sources or the syllabus:** RAG.
4. **Probe below the calibrated threshold (outside the KB):** direct LLM, with an "ungrounded" notice.
5. **Simple 2-mark definition:** direct LLM, which saves about 1,000 retrieval tokens.
6. **Anything else in-domain:** RAG.

Verification runs only when the workflow has evidence **and** a trigger fires.
Triggers are: weak evidence, a numerical/procedure intent, an ambiguous or
multi-part question, a 10-mark answer, or dates/figures in the answer. If more
than 25% of claims are unsupported, a number doesn't match, or a citation is
invalid, the system makes one revision call.

## 6. Engineering findings

- **TLS.** Python's certifi bundle fails verification on this machine (HTTPS inspection). Network code uses `truststore` (the Windows certificate store) instead. Verification is never disabled.
- **faster-whisper 1.2.1 is incompatible with PyAV 19.** Audio is decoded in `src/voice/speech.py` and passed to Whisper as an array. A TTS → ASR round trip transcribed the test question exactly.
- **Verification cost on CPU.** MiniLM manages about 33 sentences/s here, so mapping 50 evidence windows took 1.5–3 s per answer. Window embeddings are therefore precomputed per index (`src/verification/evidence_cache.py`).
- **A dead LLM server cost about 4 s per request** (Windows localhost IPv6/IPv4 retry). Outages are now cached for 30 s.

## 7. Engineering-wide knowledge base (beyond the report's AIML scope)

The report covered AIML only. The system now covers all 26 catalogue groups
(COMMON, CSE, AIML and 23 other branches, from Aerospace to Textile).

### 7.1 Catalogues

`config/branches/*.json` holds 289 subjects, each with 5 topics. **The 23 new
branch catalogues were drafted for this build and are marked
`"catalogue_status": "illustrative"`. Before the team presents them as a
university syllabus, they must check them against the actual university
syllabus.** `config/topic_hints.json` maps generic topic names (for example
"Functions" or "Teach pendant") to the intended page.

### 7.2 Corpus collection

`src/data_collection/build_engineering_corpus.py` resolved **1,370 of 1,370**
non-AIML topics. Of these, 1,352 matched an exact title or hint; the rest were
semantic matches, scored for relevance and accepted at 0.50 or above. 5
resolutions are flagged as low confidence. Every subject has at least one
article.

The collector rejects disambiguation, list and outline pages and stubs (3
articles were dropped as too short after cleaning). The result is 1,425 unique
articles, 48–96 per group. An article shared by several subjects carries all
of their IDs. 314 long articles are truncated at 30,000 characters.

**Rate limiting.** Wikimedia rate-limited this IP (HTTP 429). A naive design
(one search plus one fetch per topic) needed about 3,900 requests and an
estimated 10 hours. The collector now:
- fetches 20 intros per request, trying exact titles first;
- searches only for unresolved topics;
- downloads full articles as wikitext, 20 per request, and converts them locally (`src/data_collection/wikitext.py`, which keeps `<math>` and formula templates).

Collection then took about 186 requests. It honours Retry-After, uses a
descriptive User-Agent and never parallelises requests. All responses are
cached, so a rerun makes no requests.

**Audit.** `data/metadata/engineering_topic_resolution_v1.csv` lists, for each
topic:
- the page it was resolved to;
- the match score;
- the runner-up candidate;
- the rejected candidates;
- an empty `reviewer_ok` column.

A reviewer should go through at least the low-confidence and non-exact rows.

### 7.3 Index

Built with the V4 pipeline (section-aware cleaning, 220-token chunks with 30
overlap, a "Title > Section:" prefix): **35,567 chunks**, average 169 tokens,
`IndexFlatIP`, 55 MB. Embedding took 3.7 h on this laptop's CPU, which
thermal-throttles. On a Colab T4 this should take minutes.

### 7.4 Results

Benchmarks:
- **AIML:** the 50 AIML questions from §3, used to check that the much larger index does not degrade AIML retrieval.
- **Engineering:** `retrieval_engineering_v1.jsonl`, 50 questions with 2 per non-AIML group. The labels were written before the index existed and are draft labels. One label was corrected after the first run: Wikipedia now titles *Hydraulic fracturing* as *Fracking* (same article), so E41 accepts both titles. This is recorded in the file.

| Index : config | Benchmark | S@1 | S@3 | S@5 | MRR@10 | Article@1 |
|---|---|---|---|---|---|---|
| V4 dense | AIML (50) | 0.90 | 1.00 | 1.00 | 0.947 | 0.98 |
| engineering_v1 dense | AIML (50) | 0.90 | 0.98 | 0.98 | 0.939 | 0.96 |
| engineering_v1 dense, subject filter | AIML (50) | 0.92 | 1.00 | 1.00 | 0.957 | 0.98 |
| engineering_v1 dense, subject filter + widening | AIML (50) | 0.90 | 1.00 | 1.00 | 0.947 | 0.98 |
| engineering_v1 dense | Engineering (50) | 0.90 | 1.00 | 1.00 | 0.950 | 0.94 |
| engineering_v1 dense, subject filter | Engineering (50) | 0.92 | 0.98 | 0.98 | 0.950 | 0.96 |
| **engineering_v1 dense, subject filter + widening (app behaviour)** | Engineering (50) | 0.92 | **1.00** | **1.00** | **0.960** | 0.96 |
| engineering_v1 hybrid ¹ | Engineering (49 answerable) | 0.94 | 0.98 | 1.00 | 0.964 | 0.96 |

¹ Run before the E41 alias fix. Compare with dense on the same 49 questions:
S@1 0.90, MRR 0.949.

Reading the results:

- **Scaling from 3k to 36k chunks barely hurt AIML retrieval.** Without a filter, MRR drops from 0.947 to 0.939. The only lost question is R47 (seq2seq), which falls to rank 9 behind *Sequence alignment* and *Sequence diagram*. Selecting the subject recovers it.
- **Cross-branch retrieval is as good as AIML retrieval:** S@5 1.00 and MRR 0.95 across 25 non-AIML groups.
- **Hybrid differs by index.** On the engineering benchmark it is slightly better at rank 1 (S@1 0.94 vs 0.90, a 2-question difference). On AIML V4 it was worse (§3). The sample is too small to change the default, so dense stays the default; re-test after label review.
- **Latency.** Dense search over 36k vectors takes about 45–70 ms unfiltered. With a filter it takes about 200 ms, because building the ID selector costs more than it saves at this size. Latency on this throttling laptop is noisy.

**Bug found and fixed: strict subject filters made in-KB questions
ungrounded.** Catalogue subjects overlap. For example, the *fractional
distillation* articles are filed under Mass Transfer, Process Equipment Design
and Petroleum Refining, not Petrochemical Technology. A student who selected Petrochemical Technology got
a top score of 0.35, below the in-domain threshold. The router then answered
from the LLM alone as "outside the knowledge base", even though the Chemical
branch had matches at 0.76.

The pipeline now widens the scope when the selected scope has no match at the
calibrated threshold: subject first, then branch, then the whole index
(`scoped_retrieve` in `src/pipeline.py`). It tells the student when it does
this. In the benchmarks, 3 of 100 questions widened (E21, E44, R47), and all
three put a relevant source in the top 2. The cost is one rank-1 hit on AIML
(R47 moves from rank 1 to rank 2). The evaluator measures this exact code path
(`--subject-filter widen`).

**Router calibration for engineering_v1.** In-domain: the 100 AIML and
engineering questions. Out-of-domain: 30 non-engineering questions
(`router_out_of_domain_v2.jsonl`), because the V4 out-of-domain set consisted
of other-branch questions, which are now in domain.

| Measure | Value |
|---|---|
| AUC | 0.999 |
| In-domain threshold (Youden J 0.98) | 0.491 |
| Strong threshold | 0.701 |
| In-domain kept / out-of-domain falsely accepted | 98% / 0% |
| Highest out-of-domain score / lowest in-domain score | 0.481 / 0.454 |

The same caveat applies: the threshold was fitted on these questions, so a
held-out check is needed.

### 7.5 Catalogue compared with the AICTE model curriculum (11 Nov 2026)

**Reference documents.** Seven official AICTE PDFs (aicte.gov.in; stored in `data/raw/aicte/`, not committed):
- the UG model curriculum Vol. I (2018: first year, Civil, Electrical, Mechanical, ECE, CSE) and Vol. II (2018: Chemical, Metallurgical);
- the revised CSE (2022), Civil (2024) and Mechanical curricula;
- Robotics & AI Engineering;
- the course outlines for emerging areas (AI, data science, cyber security, IoT and others).

**Method.** `src/data_collection/compare_aicte.py` extracts 325 course titles,
with document and page for each. It matches every catalogue subject (name plus
topics) to the closest AICTE course for its branch, using the app's MiniLM
embeddings: a score of 0.60 or above is a match and 0.45 or above a partial
match. It then lists the AICTE core courses that no catalogue subject covers.
The parser handles AICTE's own quirks: Vol. II codes its Chemical courses
`PCC-CS1xx`; ECE uses `EC01:`-style codes; and the PDFs contain typos such as
"Complier Design". **This is a screening aid for human review, not a verdict.**

**AICTE has a model curriculum for 14 of the 26 groups: 167 subjects.**
- **116 of those (69%) match or partly match an AICTE course**: 80 matches and 36 partial matches.
- 51 have no close AICTE core course. Most are mainstream electives that AICTE does not list as core, such as Cloud Computing, Web Technologies, Software Engineering, VLSI Design, Wireless Communication and IC Engines.

**12 groups (122 subjects) have no AICTE model curriculum:** aerospace,
agriculture, automobile, biomedical, biotechnology, instrumentation, marine,
mechatronics, mining, ocean, petroleum and textile. They must be validated
against a university syllabus.

**64 AICTE core courses are missing from the catalogue** (`data/metadata/aicte_gaps.csv`):

| Group | Missing core courses | Notes |
|---|---|---|
| Robotics | 18 | |
| IT | 11 | Data structures, operating systems, DBMS and compilers. The IT list duplicates little of CSE, so IT students currently rely on the CSE subjects |
| Civil | 7 | e.g. engineering economy and estimation, disaster preparedness |
| Electrical / EEE | 11 | e.g. microprocessors, signals and systems, analog electronics |
| ECE | 3 | Probability and stochastic processes, control systems, computer networks |
| Other groups | 14 | |

The evaluation studio's *Review syllabus* tab shows the AICTE match next to
every subject, so reviewers can confirm or correct each one. Their decisions
are saved in `config/branches_review/<group>.csv`.

## 8. System benchmark with Qwen2.5-3B (report §22)

**Run setup:**
- Google Colab, T4 GPU, `qwen2.5:3b` via Ollama (4-bit, the same build as the laptop); 8 Nov 2026.
- Engineering index, dense retrieval, temperature 0.3.
- A seeded sample (seed 7) of 20 in-domain questions (AIML and engineering), 20 out-of-domain questions and all 10 extra items.
- 50 questions × 5 strategies = 250 answers.
- **0 failures; all 250 are real model output.**

Files are in `evaluation/results/`:
- `system_benchmark_20261108_161042.json` (summary)
- `system_benchmark_partial_20261108_154626.jsonl` (per-answer rows, with answers)
- `grading_sheet_20261108_161042.csv` (blind; for raters)
- `grading_key_…json` (**do not show to raters**)

| Strategy | Median latency | p90 | Model calls | Input tok | Output tok | Grounded | Verified | Revised | Unsupported claim rate ¹ | Cited |
|---|---|---|---|---|---|---|---|---|---|---|
| direct | 3.2 s | 9.9 s | 1.00 | 161 | 320 | 0% | – | – | – | 0% |
| fixed_rag | 3.1 s | 9.7 s | 1.00 | 1,135 | 323 | 100% | – | – | – | 16% |
| modality | 2.8 s | 9.9 s | 1.00 | 1,135 | 315 | 100% | – | – | – | 16% |
| always_verify | 6.9 s | 22.1 s | 1.58 | 2,143 | 538 | 100% | 100% | 58% | 0.42 | 54% |
| **adaptive** | 4.6 s | 13.0 s | 1.10 | **814** | 388 | 48% | 30% | 10% | 0.17 | 14% |

¹ Mean over verified answers only. It is measured after revision, so it is
not comparable between strategies that verify different questions.

**Routing.** Adaptive behaved as designed on every question:

| Question type | Routing |
|---|---|
| Out-of-domain | **20 of 20 direct** (no retrieval tokens spent) |
| In-domain | 17 of 20 RAG; the 3 direct ones were 2-mark definitions, the intended economy rule |
| Chat | 2 of 2 direct |
| Comparison, procedure, evidence requests | 6 of 6 RAG |
| Numerical | 1 of 2 RAG; X04 (an entropy calculation) had a weak retrieval match and went direct |

**Cost.** Across the 250 answers, adaptive used 60k tokens. Fixed RAG used 73k
(adaptive is 18% lower) and always-verify used 134k (adaptive is 55% lower).
Input tokens were 28% lower than fixed RAG. The cost is latency: the median is
4.6 s against 3.1 s for fixed RAG. Verification ran on half the in-domain
answers, at a median of 1.5 s each, and 10% of answers needed a revision call.

**Why retrieving everything is not "safer".** The fixed-RAG and modality
baselines mark 100% of answers as grounded, including the 20 out-of-domain
questions. There, the sources are irrelevant passages. When always-verify
checked those answers, 18 of 20 needed a revision. **Even after revision, the
mean unsupported claim rate was 0.82**: most claims were not supported by the
sources the answers were "grounded" in. On
in-domain questions, the unsupported claim rate was 0.12–0.15. Routing
out-of-domain questions away from retrieval, and labelling them as
ungrounded, is more honest than showing irrelevant sources.

**Defect found: Qwen2.5-3B rarely cites its sources.** Only 14–16% of RAG
answers contained [n] citations. The citations that do appear come mostly
from revision calls, which demand them. The citation rule came before the
long answer-format instructions, and the small model followed the format and
dropped the rule. Fix: the rule is now repeated as the last line of the
prompt (`CITATION_REMINDER` in `src/llm/prompts.py`). **This has to be
re-measured on Colab**: rebuild the bundle and re-run with the same seed. A
local check was not possible because the laptop ran out of memory.

**Defect found: wrong figures were hidden from the unsupported claim rate.**
In the Colab demo question (three-phase induction motor), the answer included
an invented example: "6 poles … synchronous speed of 1500 RPM … slip 0.02 …
1470 RPM". The physics is wrong: a 6-pole motor at 50 Hz runs at 1000 RPM.
None of those figures appear in the sources, and the number check failed.

The claim was still labelled *partial* because its wording resembled a source
passage. Partial claims are not counted as unsupported, so after the one
allowed revision the student saw "Unsupported claim rate: 0.0".

Fix (`src/verification/claims.py`, with a regression test):
- A claim containing a figure that is in no source is now **unsupported**, however similar its wording is.
- The student sees a warning listing the figures that could not be found in the sources.

The unsupported claim rates in the table above were measured *before* this
fix, so they understate the problem. Limitation: the check only tests whether
a number appears in the evidence. It cannot detect wrong reasoning built from
numbers that do appear there.

### 8.1 Re-run after both fixes (9 Nov 2026)

The same 50 questions (seed 7), model, GPU and settings were used, with the
citation reminder and the stricter number rule added. Again, all 250 answers
were real model output, with 0 failures. Results are in
`system_benchmark_20261109_111552.json` and `grading_sheet_20261109_111552.csv`.

| Strategy | Grounded answers that cite sources | Invalid citations | Total tokens | Median latency |
|---|---|---|---|---|
| fixed_rag | 8/50 → **41/50** | 0 → 0 | 72.9k → 75.3k | 3.1 → 3.4 s |
| modality | 8/50 → **35/50** | 0 → 0 | 72.5k → 73.8k | 2.8 → 2.9 s |
| always_verify | 27/50 → **37/50** | 0 → 0 | 134.0k → 126.0k | 6.9 → 7.5 s |
| **adaptive** | 7/24 → **19/24** | 0 → 0 | 60.1k → 65.5k | 4.6 → 4.1 s |
| direct | – | – | 24.0k → 23.3k | 3.2 → 3.5 s |

- **The citation fix works.** Citation use in grounded answers rose from 16–29% to 70–82%, depending on the strategy. No citation points to a source that does not exist. The reminder costs about 30 input tokens per RAG prompt.
- **The adaptive cost advantage holds:** 13% fewer tokens than fixed RAG and 48% fewer than always-verify. Routing was identical to the first run: 24 RAG and 26 direct, with all 20 out-of-domain questions direct.
- **Unsupported claim rates are higher under the stricter rule, as expected.** Adaptive in-domain went from 0.15 to 0.23. Always-verify went from 0.12 to 0.10 in-domain and from 0.82 to 0.70 out-of-domain. Figures that were previously hidden as *partial* now count. The two runs are therefore not directly comparable on this measure: the rule change and normal generation variance are mixed together. Treat the run-2 figures as the reference.
- Adaptive revised 14% of answers, up from 10%.

### 8.2 Invented worked examples: fixes after the 9 Nov run (not yet measured on Colab)

The induction-motor demo answer invented "6 poles … 1500 RPM … slip 0.02 …
1470 RPM". The source says a **four**-pole motor runs at 1,500 RPM, so the
number exists in the source but was applied to the wrong motor. A number check
cannot catch that. There are now three layers of defence:

1. **Prevention (prompts).** The 5-mark format no longer demands an example the sources may not have. An example rule says: take examples from the sources; otherwise give a qualitative example with no new numbers; never make up numerical worked examples. The revision prompt now requires a figure that is not in the sources to be corrected or removed, and forbids "the knowledge base lacks…" disclaimers for removed content (the demo answer had contradicted itself that way).
2. **Tighter number check.** A figure must appear in an evidence window that is *relevant to the claim* (combined score of at least 0.45), not merely anywhere in the five passages.
3. **NLI contradiction check (optional; `EDUROUTE_NLI_MODEL=cross-encoder/nli-deberta-v3-xsmall`; `src/verification/nli.py`).** Each claim is compared with its top 3 evidence windows:
   - entailed by any of them → supported;
   - contradicted by the best window, or by 2 of the 3 → unsupported, and the student sees "The sources contradict this part of the answer";
   - otherwise → neutral, and the existing heuristics decide.

   The rule exists because a lower-ranked window is often about a different case. In the first version, "no torque is developed when the rotor is stopped" (a passage about single-phase motors) "contradicted" a correct sentence about torque, at probability 0.98. A probability threshold could not separate that false alarm from the real error.

**Check on the demo answer** (11 claims, the Induction motor article as evidence):
- With NLI, exactly the two wrong claims are unsupported: the 6-pole/1500 RPM claim (contradiction 0.96–0.97) and the slip/1470 RPM claim (figures not in the sources). The 9 correct claims pass.
- This is a single answer, not a measurement.

The NLI model is enabled in the Colab notebook. It is off by default on the
laptop, where it adds load time and memory. The benchmark now records
`answers_with_contradicted_claims` and `answers_with_unconfirmed_figures`.

**Known remaining limit:** an arithmetic or reasoning error that uses only
figures the sources support, and that no source contradicts, still passes.
Catching it needs a stronger checker: a larger NLI model, or an LLM judge
evaluated against human claim labels.

**Use the 9 Nov grading sheet for human grading.** It reflects the current
system.

**Example of an error that needs human grading.** One fixed-RAG answer said
the EU AI Act has "three levels of risk: high, medium and low". The Act's
actual tiers are unacceptable, high, limited and minimal risk. The other
strategies did not make this claim; the difference is generation variance.
Fixed RAG does not verify, so nothing caught it. Answer correctness cannot be
read from this table: two raters must grade the blind sheet using
`evaluation/ANSWER_RUBRIC.md`.

**Limits of this run:**
- Sample size: 50 questions, one run each.
- The in-domain labels are draft labels.
- The out-of-domain threshold was fitted on overlapping questions (§7.4).
- Latency is GPU latency; on the laptop CPU, generation runs at about 1.3 tokens/s, so answers take minutes.

### 8.3 Third Colab run (10 Nov 2026): the NLI check over-flagged, and the prompt leaked

**Run setup:**
- The same 50 questions (seed 7) × 5 strategies, all real model output, 0 failures.
- New in this run: the example rule, the tighter number check and NLI (`nli-deberta-v3-xsmall`, contradiction ≥ 0.7, best window or 2 of 3).
- Files: `system_benchmark_20261110_153451.json`, `system_benchmark_partial_colab.jsonl` (with per-claim results) and `grading_sheet_20261110_153451.csv`.

**Results (adaptive):**
- 13% fewer tokens than fixed RAG.
- Median latency 4.0 s.
- In-domain citations 65%; unsupported claim rate 0.17 in-domain.
- All 20 out-of-domain questions answered directly.
- The cost and routing picture is unchanged from §8.1.

**Finding 1: the NLI contradiction check is not reliable on general claims.**
- It flagged 48 of 1,022 checked claims. I reviewed every flag against its evidence.
- Of the 35 flags outside the out-of-domain set, only **3 were real errors**: "C-S-H (*Cement*-Silica-Hydrate)", which appeared twice and is really *calcium*-silicate-hydrate, plus the 6-pole/1500 RPM motor case.
- 11 flags were not claims at all: formula lines, "where y is …" continuations and echoed prompt rules.
- The rest were correct statements, such as ECG electrode placement, the GAN generator's goal and SLAM.
- On those 49 pairs:

  | Model and rule | Real errors caught (of 3) | False alarms |
  |---|---|---|
  | xsmall, any threshold up to 0.95 | 3 | 8–14 |
  | `nli-deberta-v3-base` (larger) | 0–1 | 9–13 |
  | **xsmall at ≥ 0.95, claims with a figure only (adopted)** | 1 (the motor case) | 1 |

  The larger model was **worse**: it missed the motor error (0.34) and the C-S-H error (0.005).

The verifier now treats a contradiction as unsupported only for claims that state a figure, at probability ≥ 0.95. Other contradictions are recorded as `possible_contradiction` for developers and are neither counted nor shown to students. The C-S-H kind of error (a wrong word, no figure) is therefore not caught automatically; it remains for human grading. Because of this, the `answers_with_contradicted_claims` figures in this run (12% adaptive, 42% always_verify) overstate real contradictions and should not be reported.

**Finding 2: fragments were being verified as claims.** Formula lines,
continuation fragments, calculation steps and echoed rules are now skipped
(`is_fragment` in `src/verification/claims.py`). Figures that appear in the
student's question, such as "9 positive and 5 negative examples", no longer
count as unconfirmed.

**Finding 3: the prompt leaked into answers.** 18 of 250 answers (7%)
contained instruction text. Mostly it was a stray "Answer format:" line; in 4
answers it was whole rules such as "Do not invent source titles or URLs.",
mainly on out-of-domain questions forced through retrieval. Two fixes:
- The system prompt now says the instructions must never be repeated.
- `strip_prompt_echo` removes echoed instruction lines before display and verification. Checked on all 250 answers: it cleaned exactly those 18 and changed nothing else.

**Other steps in the run:**
- The router held-out check matched the laptop: accuracy 0.964.
- The synthetic voice WER was 5.7% (median 1.6 s per question).
- The claim-labelling sheet was regenerated without fragments: `claim_labels_20261110_213007.csv`, with 120 claims (40 supported, 40 partial, 40 unsupported). Use this file, not the Colab export, which includes fragments.
- **The image step produced no results file.** The script stopped before writing anything, most likely because the vision model was not being served. It now always writes a file, including the reason for any failure (`image_benchmark_FAILED_*.json`).

### 8.4 Image pipeline checked on the laptop with Qwen2.5-VL-3B (10–11 Nov 2026)

Running a real image question on the laptop found three defects, all now fixed and tested:

1. **The vision client ignored the timeout setting**, so it gave up after 180 s. It now uses `EDUROUTE_VISION_TIMEOUT`, which defaults to the text model's timeout.
2. **When image analysis failed, the question was answered without the image.** The result was a confident wrong answer: "XOR gate" for an AND truth table. Questions that refer to the image ("this truth table", "the circuit shown", "the problem in the image") now get a clear "I could not analyse the image" message instead (`depends_on_image` in `src/pipeline.py`). The same applies when no vision model is configured. General questions with an image still get a knowledge-base answer, with a notice.
3. **The image scoring was too loose.** The accepted answer "and" matched the English word "and", so "XOR gate … inputs and outputs" scored as correct. Matching is now whole-word; all-capitals answers such as AND are case-sensitive; and the bare "30" and "b" alternatives were removed.

**Vision model behaviour.** With a working connection, Qwen2.5-VL-3B (4-bit, CPU):
- read the bar chart (Pump B, 50) and the series circuit (30 Ω, 0.4 A) correctly;
- **misread the AND truth table** three times: as NAND, NOT and NAND.

A "read first, then answer" prompt did not help: its reading did not copy the table rows, and the result was the same on the other images. So the prompt was left unchanged. Reading tabular images is a limitation of the 3B vision model; the Colab run measures it on all 8 starter images.

The Colab image step had produced no file on 10 Nov. It now always writes a result or a failure file. The notebook downloads the vision model with a retry and warms it up before use, and it runs the image step **before** the 30-minute benchmark so a failure shows up early.

### 8.5 Fourth Colab run (11 Nov 2026): final system benchmark and the first image benchmark

The same 50 questions × 5 strategies were run, with all fixes up to 11 Nov.
All 250 answers were real model output, with 0 failures. Files:
`system_benchmark_20261111_042423.json`,
`system_benchmark_partial_colab_20261111.jsonl`,
`grading_sheet_20261111_042423.csv` (**use this sheet for human grading**),
`claim_labels_20261111_042643.csv` and
`image_benchmark_20261111_035749_rescored.json`.

| Measure | 10 Nov | 11 Nov |
|---|---|---|
| Answers containing echoed instructions | 18 / 250 | **0** |
| Adaptive: answers with a counted contradiction | 12% | **2%** |
| always_verify: answers with a counted contradiction | 42% | **6%** |
| Adaptive in-domain citation rate | 65% | **75%** |
| Adaptive unsupported claim rate (all) | 0.19 | **0.13** |
| Adaptive tokens vs fixed RAG | −13% | −15% (897 vs 1,247 input tokens per answer) |

Of the 4 counted contradictions:
- 2 come from always_verify on out-of-domain questions, where the sources are irrelevant.
- 1 is a date about the EU AI Act.
- 1 was a figure the student gave ("32x32 image"), which the number parser could not read. Since fixed: "32x32" is two numbers, "(1,1)" is no longer read as 11, and IP addresses are not figures. Figures that the vision model read off the image also count as known.

**Image benchmark** (Qwen2.5-VL-3B on a T4; 8 starter images). Re-scored
after two scoring bugs were fixed: LaTeX answers such as `30 \, \Omega` were
not recognised, and the stems "encod"/"decod" failed whole-word matching.

| Strategy | Images fully correct | Facts found | Median latency |
|---|---|---|---|
| modality (vision model only) | **6/8** | 88% | 2.7 s |
| adaptive (vision + knowledge base) | 5/8 | 75% | 9.5 s |

- Both strategies were correct on the bar chart, equation, series circuit, code and flowchart; modality was also correct on the block diagram.
- **The truth table was wrong in every run** (NAND or NOT for an AND table): a vision-model limitation.
- **Sine-wave frequency was misread or miscalculated** (0.25 Hz, 1 Hz; the answer is 50 Hz). The amplitude, 5 V, was read correctly.
- **Defect found and fixed:** in the adaptive (hybrid) workflow, the grounding rules made the model answer "the knowledge base does not contain enough information" even when the image analysis held the answer. That is why adaptive scored 0/2 on the sine wave. `IMAGE_RULES` now tells the model to take facts about the image from the image analysis. Re-checked on the laptop: the hybrid answer now states the 5 V amplitude from the image. The wrong frequency (1 Hz) is flagged to the student as an unconfirmed figure.
- Conclusion for the report: for questions whose answer is *in* the image, the vision-only route is faster and at least as accurate. The hybrid route adds value only when background concepts are needed. Routing image questions that point at the image (`depends_on_image`) to the vision-only workflow is a sensible next refinement, to be confirmed on real student images.

## 9. Tools and checks added before the next Colab run (9 Nov 2026)

| Item | Status | Result / how to use |
|---|---|---|
| Router held-out check (`evaluation/check_router_heldout.py`, `router_heldout_v1.jsonl`) | **Run on the laptop** | See the breakdown below |
| Voice WER (`evaluation/evaluate_asr.py`) | **Synthetic baseline run** | See the breakdown below |
| Image questions (`evaluation/run_image_benchmark.py`, `make_image_starter_set.py`) | Ready; runs on Colab with Qwen2.5-VL | 8 synthetic images with exactly known answers in `evaluation/image_eval/`. Add real photos to the same folder and CSV |
| Claim labelling (`evaluation/claim_labels.py`) | Ready | The benchmark now stores every verified claim. `export` writes a stratified sample for raters; `score` measures the verifier and suggests a threshold |
| Grading analysis (`evaluation/analyse_grading.py`) | Ready | Per-strategy quality per criterion (in- and out-of-domain), the share of wrong answers, and rater agreement (exact, within 1, quadratic-weighted kappa) |
| Benchmark resume and keep-awake | Done | `--resume <partial.jsonl>` skips finished answers; `--keep-awake` stops Windows sleeping. `EDUROUTE_LLM_TIMEOUT` was raised from a fixed 180 s, which would have cut off long CPU answers |
| Log retention and privacy notice | Done | Logs older than `EDUROUTE_LOG_RETENTION_DAYS` (default 30) are deleted, at most once a day. A privacy notice in the app sidebar states what is logged (a hash, not text), what is never kept, retention and where data goes |
| Colab notebook | Rebuilt | One session: the system benchmark (resumable), image questions, router check, voice baseline, claim export; results come back as one zip |
| Tests | 44 unit tests + 1 app test, all passing | |

**Router held-out check.** The threshold (0.491) was applied unchanged to 60
new questions. Accuracy, excluding 4 borderline questions, was **0.964**:

| Question set | Result |
|---|---|
| 30 in-domain, one or two per branch | 96.7% kept (29/30) |
| 10 easy out-of-domain | 100% rejected |
| 16 hard out-of-domain (history, law, economics, medicine, arts) | 93.8% rejected (15/16) |

The two errors are informative:
- *Phase-locked loop* (H09, score 0.476) has no article in the corpus. It is a **coverage gap**, not a router fault; add it in the next corpus update.
- *Vaccines* (N22) was accepted, but the corpus does contain a *Vaccine* article (Biotechnology), so the answer would be grounded. The label, not the router, is debatable.

Borderline questions, reported but not scored: the greenhouse effect (0.74) and earthquakes (0.54) were accepted; Pythagoras (0.489) and bees (0.44) were rejected.

**Voice WER, synthetic baseline.** Twenty benchmark questions were read aloud
by Edge-TTS *NeerjaNeural* (en-IN). The *Prakash* voice failed and gTTS
(Indian English) replaced it; the files are named after the engine actually
used.
- WER **5.7–7.1%** across two generations; 65–70% of transcripts exact; median 2 s per question; real-time factor 0.57.
- Most errors are spelling-level: "back propagation", "Homan" for Hohmann, "base theorem" for Bayes theorem. The exception is "What is" heard as "Which is" with gTTS.
- **This is a pipeline check, not a student-voice measurement.** Real recordings are needed (see §10).

**CPU run.** A CPU benchmark was started on the laptop and stopped after 13
answers in favour of Colab. On this laptop, Qwen2.5-3B wrote about 3.9
tokens/s once enough RAM was free, so a 10-mark answer took 2–4 minutes.
Its partial file, `system_benchmark_partial_cpu_run1.jsonl`, is kept.

## 9a. Improvements after the evaluation (12 Nov 2026)

| Area | Change | Measured effect |
|---|---|---|
| Quick response | Answers stream onto the screen as they are generated; models and Whisper are loaded in the background when the app starts | Laptop CPU: first words after 3.2 s, against 10.4 s for the full answer. A cold model load (about 48 s on the CPU) no longer delays the first question |
| Image routing | Questions about the image itself ("this truth table", "the circuit shown") go to the vision model alone | On the 11 Nov benchmark, the vision-only route was at least as accurate on every such image and about 3.5× faster than hybrid |
| Image input | Camera photo button; the Colab notebook compares Qwen2.5-VL 3B and 7B | Preprocessing (2× upscale, contrast) did **not** help: 1 of 6 tries correct. The 3B model reads the truth-table rows correctly but names the gate wrongly, a reasoning error, so the 7B model is the candidate fix |
| Voice | A technical-term corrector checks transcripts against the knowledge-base vocabulary (it never changes small words or plurals; corrections are shown to the student); beam size 1 | Synthetic set: WER 7.1% → **4.3%**, median 2.2 s per question |
| Documents | Students upload PDF, Word or text files; passages are cited by page, and summaries cover the whole document | End to end: questions about a lab manual were answered from the right page; unrelated questions still used the knowledge base |
| Diagrams | Workflow diagrams (Graphviz) for "draw / flowchart / workflow" requests and 10-mark process answers | Qwen2.5-3B often draws disconnected fragments. These are detected, and the step-by-step flowchart built from the answer is used instead |

Tests: 65 unit tests plus 1 app test, all passing.

**Housekeeping (12 Nov):**
- C: had filled to 0.2 GB free, mainly because Windows' page file grew to 14.9 GB under memory pressure.
- Removed unused downloads: a rejected NLI model, partial model files and the Ollama installer. This brought C: back to 4.7 GB free.
- Restarting the laptop lets Windows shrink the page file.

## 9b. Gap filling and arithmetic checking (11 Nov 2026)

**Knowledge-base gaps filled** (`src/data_collection/extend_corpus.py`):
- **Inputs:** the 64 AICTE core courses missing from the catalogue (§7.5), plus two gaps found in evaluation: *Phase-locked loop* (router held-out miss H09) and *Bayes' theorem* (absent from the speech term vocabulary).
- **Resolution:** the gaps went through the same audited resolver as the main corpus. A first dry run matched compound course titles to journals, a company and unrelated pages, so authoritative hints were added for 23 titles.
- **Result:** 66 of 66 topics resolved; 15 already covered; **37 new articles, 800 new passages**. Audit file: `data/metadata/engineering_supplement_v1.csv`.
- **Index:** only the new passages were embedded (87 s, against 3.7 h for a full rebuild) and appended. The index now holds **36,367 passages** (before: 35,567). Backups are kept as `*.bak`.

| Measure | Before | After |
|---|---|---|
| Router held-out accuracy (τ unchanged) | 96.4% (29/30 in-domain kept) | **98.2%** (30/30 kept; phase-locked loop now covered) |
| Engineering retrieval S@1 / MRR | 0.90 / 0.950 | 0.88 / 0.940 |
| AIML retrieval S@1 / S@5 / MRR | 0.90 / 0.98 / 0.939 | 0.88 / 0.98 / 0.924 |

Only two questions changed rank, and in both a new, relevant article now ranks first that the original labels do not list:
- **E04** ("TCP/IP protocol suite") now returns *Computer network* first.
- **R06** ("Explain Bayes theorem") now returns the article *Bayes' theorem* itself.

The labels were left unchanged, since editing labels after seeing results would bias the benchmark. The small drop is therefore a labelling artefact, and the team's label review (§10) should decide on it.

**Arithmetic and unit check** (`src/verification/arithmetic.py`). This addresses the verifier's documented blind spot: a calculation with plausible-looking figures was never checked.
- **Equations:** every written equation is recomputed: each side that is pure arithmetic with known units (prefixes converted: 400 mA = 0.4 A) must match the next. Sides containing variables are skipped, never guessed.
- **Period and frequency:** stated in the same sentence, they must satisfy f × T ≈ 1.
- **Scope:** it runs on every answer, whatever the route, in microseconds.
- **Action:** a wrong calculation triggers one revision call (unless one was already made); anything still wrong is shown to the student as a warning. The benchmark records `answers_with_arithmetic_issues`.
- **Tests:** on 13 test statements it catches the three known errors: 12 V / 30 Ω = 0.5 A; "20 ms → 1 Hz"; and the 6-pole motor's 120 × 50 / 6 = 1500 RPM (it is 1,000). It stays silent on correct working, LaTeX, rounding and variable names.

Tests: 79 unit tests plus 1 app test, all passing.

## 9c. NVIDIA GPU server kit (11 Nov 2026)

`deploy/` holds everything needed to run on the team's NVIDIA servers (guide: `docs/GPU_SERVER.md`):

- **Packaging:** the committed project ships as one ~85 MB package. Secrets and personal data are git-ignored, so they cannot be included.
- **Pre-flight:** a read-only check of the server.
- **Setup:** no root needed. It installs CUDA PyTorch matched to the driver, Ollama in user space, qwen2.5:3b, qwen2.5vl:3b and qwen2.5vl:7b; the NLI check and Whisper run on the GPU.
- **Services:** start and stop the app on 127.0.0.1 only, reached through an SSH tunnel.
- **Evaluations:** one script runs them all (tests, retrieval, router, the **full** system benchmark, images with 3B vs 7B, voice, claim export) and returns a zip. The system benchmark resumes after an interruption.
- **Docker:** an alternative route.
- **Code changes:**
  - Whisper's device is now a setting (`EDUROUTE_WHISPER_DEVICE`, `EDUROUTE_WHISPER_COMPUTE_TYPE`).
  - `pymupdf` was missing from `requirements.txt`, so PDF uploads would have failed on a fresh install. It has been added.

**Production hardening** (the app will run on a pod whose link is public):

- **Access codes:** an access code for students and a separate studio code for the team (the studio holds grading data). Setup generates random codes; comparison is constant-time, and each wrong attempt slows the next.
- **Load limits:**
  - 60 questions an hour per session;
  - at most 4 answers generated at once, with the rest queued for up to 120 s;
  - questions capped at 4,000 characters.
- **Hidden from students:** the developer view (routing internals, strategy switch).
- **Errors:** an unexpected failure shows the student a short reference code; the full traceback goes to the server log.
- **Pod support:** `pod_start.sh` installs on the persistent volume on first boot, then starts the services and watches them; `services.sh watch` restarts anything that stops. On any address other than localhost, the app refuses to start without an access code.
- **Smoke test:** `deploy/smoke_test.py` asks real questions on every route (knowledge base with citations, general, image, voice) within time limits, and exits non-zero on failure.
- **Fixed:** the front page showed a stale passage count (35,567); it now reads it from the index.

Status: ready. The DGX login node (172.19.0.11) is reachable only from the lab network, so GPU work moved to Colab (12 Nov).

## 9d. Product completeness for real use (12 Nov 2026)

- **Full app on Colab:** `colab/EduRoute_Colab_App.ipynb` runs the complete app on a T4 GPU behind a public HTTPS link (Cloudflare quick tunnel), with random student and team codes. It runs the smoke test before opening the link, restarts anything that stops, and downloads the session's data (ratings, logs, studio work) at the end.
- **Full evaluation on Colab:** `colab/EduRoute_Colab_Benchmark.ipynb` defaults to the complete run (smoke test, all 130 questions × 5 strategies, 3B vs 7B images, router, voice, claim export) and resumes from Google Drive.
- **Operations dashboard** (`pages/3_Operations_Dashboard.py`, team code): questions per day, route mix, answer time by route (median, 95th percentile), quality signals (verified, unsupported-claim rate, revised, citing), helpful-rating rate, failed answers with reference codes. Failures are now logged to `errors.jsonl` (no question text).
- **Conversation export:** students download their questions, answers and cited sources as Markdown study notes. The app test caught that the button first appeared one interaction late; it is now filled at the end of each run.
- **README:** rewritten as a product page (what it does, architecture, measured results, three ways to run, roles, configuration, privacy). GitHub Actions CI runs the static check, the shell-script parse and the tests.
- **Static check:** pyflakes is clean; 4 minor warnings fixed.

Tests: 90 passing.

## 9e. CPU-only operation (12 Nov 2026)

Colab could not be used, so everything now runs on the laptop (i7-8665U, 4 cores, 16 GB, no GPU).

- **Measured speed:** prompt reading 18.5 → 20.9 tokens/s; writing 4.4 → 5.3 tokens/s (+20%) after tuning Ollama for the CPU:
  - one request slot;
  - flash attention with an 8-bit KV cache;
  - models kept loaded for 2 h.

  A repeated system prompt is now read from cache at ~250 tokens/s. A knowledge-base answer takes ~2.5–3 min; a general answer ~10 s.
- **Laptop `.env`** (not committed): vision on (`qwen2.5vl:3b`), NLI on, 30-minute timeouts, one answer at a time, Hugging Face offline mode. When the Wi-Fi dropped, model loading had been stalling on download retries.
- **Image fix (read, then reason):** the CPU smoke test failed the truth-table image.
  - The 3B vision model *transcribed* the table exactly, but answering in one step it called it XOR, then NOT.
  - Image-only questions now go in two steps: the vision model transcribes (tables row by row, graph axes and values), then the text model answers from the transcription.
  - Result: "AND gate", with the right reason, in 248 s on the CPU (175 s before, one call fewer).
  - `EDUROUTE_VISION_TWO_STEP=false` restores one-step answers. Tests use mock models.
- **Unattended evaluation:** `run_cpu_evaluation.cmd` (`evaluation/run_cpu_evaluation.py`) runs these stages, keeping Windows awake and skipping finished stages on restart:
  - the smoke test;
  - the adaptive strategy on all 140 benchmark questions (resumable answer by answer);
  - the image benchmark with the 3B model; 7B only with `--with-7b` and at least 8 GB free RAM;
  - voice word error rate;
  - the claim export.

  The 5-strategy comparison stays the 11 Nov GPU run: repeating it on 140 questions would take ~30 h on this CPU.

Tests: 92 passing.

## 9e-2. Full 140-question benchmark on the laptop CPU (12–13 Nov 2026)

Adaptive strategy, all 140 questions (100 in-domain, 30 out-of-domain, 10 extra), qwen2.5:3b on the i7 CPU, NLI on, extended index. File: `evaluation/results/system_benchmark_20261113_111439.json` (blind grading sheet `grading_sheet_20261113_111439.csv`).

| Adaptive | GPU 11 Nov (50-question sample) | CPU 13 Nov (all 140) |
|---|---|---|
| Failures | 0 | 0 |
| In-domain answered from sources | 85% | 79% |
| Out-of-domain kept ungrounded | 100% | 100% (30/30) |
| Grounded answers citing a source | 88% | 84% |
| Invalid citations | 0 | 0 |
| Mean unsupported-claim rate (verified answers) | 0.133 | 0.150 |
| Answers with a counted contradiction | – | 0.7% (1/140) |
| Median answer time | 4 s | 125 s (CPU) |

The full question set confirms the 50-question sample within a few points. The one flagged answer (X05, the 32×32 convolution calculation) failed the arithmetic check after its revision and carried a contradiction, so the student would see both warnings rather than silent wrong working.

The run was interrupted twice and resumed without loss. Modern Standby (lid closed) paused it for 54 min and 80 min, and a restart stopped it overnight. The answer affected by standby was re-run so its timing is clean.

## 9f. Agentic RAG, context engineering, inference optimisation, memory (12 Nov 2026)

| Feature | Module | Mechanism |
|---|---|---|
| Agentic RAG | `src/agent/agentic_rag.py` | Plan sub-queries (comparisons, multi-part questions) → retrieve each → critique (coverage of the question's key terms) → refine with the missing terms (bounded at 3 steps; optional one-call LLM query rewrite) → after verification, retrieve evidence for each unsupported claim before the one revision. Every step is traced in the developer view. |
| Context engineering | `src/rag/context.py`, `build_rag_messages(stable_prefix=True)` | Token budget per answer mode (450/750/1,100/600); near-duplicate removal; sentence-level compression to the most relevant sentences (first sentence kept, original order); only the prompt copy is compressed, so verification sees the full passages; fixed rules moved into the system message so the prompt prefix is identical and reused by the server's cache. |
| Inference optimisation | `src/inference/answer_cache.py`, `EduRouteRetriever.embed_query` | Answer cache shared by all sessions: exact tier, plus a semantic tier at cosine ≥ 0.97 with an identical content-word set. Same mode, scope and strategy only. No images, documents, follow-ups or answers with warnings. LRU, 256 entries, 24 h TTL. Query-embedding LRU (512). |
| Memory management | `src/agent/memory.py` | Working memory (recent turns as key sentences, within 500 tokens), an episodic summary of older topics, and follow-up resolution into a self-contained search query. Bounded at 20 turns. |

All four are settings (`EDUROUTE_AGENTIC_RAG`, `_CONTEXT_ENGINEERING`, `_CONVERSATION_MEMORY`, `_ANSWER_CACHE`). They stayed off until the CPU benchmark had finished, so it measured one system throughout, and are now on by default (see below).

**Measured without the language model** (`python -m evaluation.evaluate_agentic`, engineering_v1 index, 100 benchmark questions; `evaluation/results/agentic_eval_20261112_200024.json`):

| Measure | Baseline | New |
|---|---|---|
| Follow-ups ("What are its advantages?" etc., 300 cases): topic in the top 5 | 0.3% | 97.3% |
| Comparisons ("Compare A and B", 40 same-branch pairs): evidence for both sides | 22.5% (top 5) | 75% (agent's evidence, up to 6 passages; ~4 steps) |
| Source tokens in the prompt (median), 2 / 5 / 10-mark | 1,020 | 399 / 644 / 874 (−60% / −36% / −14%) |
| Best relevant passage kept in the prompt | – | 100% in every mode |
| Prompt-reading time saved on this CPU (~20 tokens/s) | – | ~31 / 19 / 7 s |
| Cache: exact repeats / reworded repeats | – | 100/100 / 98/100 |
| Cache: wrong hits between different questions | – | 0 (4,950 pairs) |

The tests caught two bugs before release:
- The cache matched "advantages of **AC** motor" with "of **DC** motor", because a 3-letter minimum dropped "AC" and "DC". Every word now counts.
- The planner ignored one-word sub-questions ("What is a transformer?").

**With the language model** (14 Nov 2026; `evaluation/run_feature_ablation.py`; qwen2.5:3b on the laptop CPU; the same 10 cases, features off vs on; `evaluation/results/feature_ablation_20261114_103600.json`):

| Cases | Features off | Features on |
|---|---|---|
| Normal questions (3): mean time / prompt tokens | 84 s / 1,226 | **56 s / 824** (−34% / −33%) |
| Comparisons (3): evidence covers both sides | 1/3 | **3/3** (citations 9 → 12) |
| Follow-ups (3): answered from the topic's sources | 0/3 | **3/3** (9 citations; 44 s instead of 13 s, because they now retrieve) |
| Repeated question | 9.1 s | **0.0 s** (cache) |

The first run found a gap. Memory resolved each follow-up's topic, but the router's "simple 2-mark definition → direct answer" shortcut then answered it ungrounded. A follow-up resolved from the conversation now goes to RAG when its topic is in the knowledge base (router rule 4b; an out-of-domain follow-up still goes direct). Re-run: 3/3 grounded.

**The four features are now on by default.** Each can be switched off with its `EDUROUTE_*` setting.

## 9g. Exact reasoning tools for images (14 Nov 2026)

The CPU image benchmark (`image_benchmark_20261114_091650.json`) gave: image-only route 7/8 images with all facts right (94% of facts), adaptive 4/8 (75%). In both lasting failures the vision model **read the image exactly**, and the 3B model then reasoned wrongly from its own transcription:
- a truth table read as 00→0, 01→0, 10→0, 11→1 was called NOT (it is AND);
- `total = 0; for i in range(1, 6): total += i; print(total)` was said to print 21 (it prints 15).

`src/vision/reasoning_tools.py` works like the arithmetic check: it computes, rather than trusting the model.
- **Truth tables:** the rows of any 0/1 table are tested against AND, OR, NAND, NOR, XOR, XNOR, NOT and buffer, with any number of inputs.
- **Code:** short snippets are run by a ~100-line interpreter that knows only numbers, strings, variables, arithmetic, comparisons, if/for/while and a few built-ins. There are no imports, attributes, function definitions or files, and runs stop at 10,000 steps.
- **Hand-off:** the result is appended to the image reading as a CHECKED FACT for the text model.

Run over every saved transcription, the tools gave AND and "15" for the two failures, and fired on none of the other six images. Live on the real models, both answers are now correct ("AND gate"; "prints 15"). Tests cover all gates; unsafe, endless and huge code is never run.

**Image benchmark re-run with the tools** (CPU, `image_benchmark_20261114_120214.json`):

| Route | Before (14 Nov 09:16) | After (14 Nov 12:02) |
|---|---|---|
| Adaptive: images with all facts right | 4/8 (75% of facts) | **8/8 (100%)** |
| Image-only: images with all facts right | 7/8 (94%) | 7/8 (94%) |

The one remaining miss is on the image-only route: the block-diagram answer leaves out one of the five expected elements, a recall gap, not a reasoning error.

Tests: 129 fast + 2 slow.

Tests: 114 passing (fast suite 112 + 2 slow).

## 9h. Faster answers on the CPU, and past papers (14 Nov 2026)

**Saved answer cache and question bank** (`src/inference/answer_cache.py`, `src/inference/question_bank.py`, `run_question_bank.cmd`):
- The cache is saved to disk and survives restarts. Each record carries the index, model and format version, so stale answers are dropped automatically.
- Student questions are stored only as a hash, hashed words and an embedding, never as text (a test checks that no wording reaches the file).
- Matching rule: the same content words, plus intent markers (compare, why, derive, calculate, example) and negations, and cosine ≥ 0.65.
  - Simple rewordings ("What is X?", "Tell me about X", "X") scored as low as 0.69 against "Explain X."
  - On the 100 benchmark questions: 500/500 rewordings found, 0 wrong matches.
  - Tests also caught "Compare A and B" matching "Explain A and B" (fixed with the intent markers).
- The question bank pre-answers "Explain <topic>." breadth-first over all 289 subjects and pins the answers (never expired or evicted), under the subject, branch and no-selection scopes.
  - Live: "Can you explain calculus" was answered in **0.03 s** instead of 68–94 s.
  - Build: 578 answers overnight (`run_question_bank.cmd`).
- The app picks up answers saved by the builder while it runs.

**Answer first, check afterwards** (`ask(defer_checks=True)`, `draft.complete()`):
- The student sees the answer as soon as it is written. Verification, any revision, the arithmetic check and the diagram run afterwards, and the answer is replaced (with a notice) only if a check changed it.
- The final answers are identical to before (tested).
- On the 140-question CPU run, 50 answers had checks after writing. For those, the wait until readable was a mean of 364 s → 224 s. Across all answers: **182 s → 131 s (−28%)**.

**Past papers** (`pages/4_Past_Papers.py`, `src/data_collection/past_papers.py`):
- Upload a previous-year paper. It is split into questions (Q1. (a) …, [10], (5M); instructions dropped), with the format taken from the marks, and each question is matched to a catalogue subject and topic.
- Any question can be answered (instantly if banked). The team can queue the whole paper into the question bank.
- A 5-question sample paper matched all five to the right subject and topic.

**Memory:** the vision model is no longer warmed at app start. Idle, it held 2.9 GB and, together with the bank build and other jobs, exhausted the laptop's RAM (MemoryError). It now loads in ~10 s on the first image question (`EDUROUTE_WARM_VISION_MODEL=true` to opt in).

**Shared resources:** `src/ui/resources.py` gives every page one copy of the index, models and cache.

**Research paper:** the edits in `docs/PAPER_UPDATE_14NOV.md` and the results above are applied (section 3.7 and results table 6.6 added).

## 9i. Better speech-to-text (17 Nov 2026)

Measured on the 20 synthetic Indian-English clips, plus a degraded copy of each: the volume at −26 dB (a quiet laptop microphone) with fan noise and 50/100 Hz mains hum at 10 dB SNR. The numbers are word error rates, about 140 words per set, so one word is about 0.7 points.

| Change | Clean | Noisy and quiet | Median time | Kept? |
|---|---|---|---|---|
| Before (generic vocabulary prompt) | 4.3% | 7.1% | 2.0 s | – |
| **Subject found automatically, second pass with its vocabulary** | **3.6%** | **6.4%** | 4.1 s | **yes**, when no subject is chosen |
| Subject known (the student chose it) | 3.6% | 6.4% | 2.0 s | already in the app |
| Recording clean-up (80 Hz high-pass, level normalisation) | 5.0% | 7.9% | 1.9 s | no: worse on its own |
| Subject topics as Whisper hotwords | 3.6% | 5.0% | 2.0 s | no gain over the prompt |
| Per-word confidence (word timestamps) | 29% | 19% | 12 s | no: breaks decoding |
| `small.en` instead of `base.en` (clean) | 3.6% | – | 5.8 s | no: 2.5× slower |

- **Automatic subject:** the first-pass transcript is searched in the index the app has already loaded, and the top five passages vote for a subject. Below a similarity of 0.45 no subject is used, so off-syllabus speech is not pushed towards engineering words. It picked the right subject in 39 of 40 clips. Cost: about 2 s more, and only when no subject is chosen (`EDUROUTE_WHISPER_SUBJECT_SECOND_PASS=false` turns it off). The student sees "Recognised with the vocabulary of *Thermodynamics*" above the transcript.
- **Always on:**
  - segments that Whisper itself rates as non-speech, and stock subtitle phrases ("Thank you for watching"), are dropped;
  - decoder loops (the same letters 4+ times in a row) are dropped;
  - one misheard phrase is not carried into the next segment (`condition_on_previous_text=False`).
- **Not measurable here:** clean-up together with the subject prompt scored 4.3% on the noisy set, against 6.4% for the prompt alone. The gap is 3 words and clean-up alone was worse, so it stays off until real recordings (Evaluation studio, "Record voice") can settle it.

**Multilingual mode (removed the same day at the team's request; kept in git history, commit `ebdb069`):**
- NLLB-200 translation passed a round-trip quality gate for 8 Indian languages (chrF 77–85).
- Whisper `small` speech was usable for Tamil (CER 19%), weak for Hindi (59%, mostly spelling variants) and unusable for Kannada and Telugu (CER > 100%).

## 10. What remains (report §25)

**Done by me (12–14 Nov):**
- Corpus gap fill (AICTE core gaps; *Phase-locked loop*): 37 articles, 800 passages; no rebuild needed (§9b).
- The full benchmark on all 140 questions (CPU, §9e-2).
- Image, voice and feature measurements (§9f, §9g).
- Production deployment kits (§9c, §9d).
- Offline read-aloud.
- The non-AI "course handbook" redesign.

**Needs the team.** Done in the **evaluation studio** (`streamlit run app.py`, then *Evaluation studio* in the sidebar, team code). Each person works under their own name, and the files are saved in the formats the analysis scripts read.

| Item | Studio tab | Who | Then run |
|---|---|---|---|
| Grade answers blind, 2 raters: the 140 answers of the 13 Nov CPU run (current system) or the 250 of the 11 Nov GPU run (5-strategy comparison) | Grade answers | Team (2 people, ~2–4 h each) | `python -m evaluation.analyse_grading --key evaluation/results/grading_key_20261113_111439.json --benchmark evaluation/results/system_benchmark_20261113_111439.json evaluation/results/grading/*.csv` |
| Label the exported claims (`claim_labels_20261114_091735.csv`) | Label claims | Team (~1 h) | `python -m evaluation.claim_labels score evaluation/results/claim_labelling/<file>.csv` |
| Record 20–30 questions per speaker (with consent) | Record voice | Team | `python -m evaluation.evaluate_asr evaluation/voice_eval` |
| Add real images with their expected facts | Add images | Team | `run_cpu_evaluation.cmd --stages images --redo` |
| Confirm or correct each subject; the AICTE match is shown (§7.5) | Review syllabus | Team; branches without an AICTE reference need the university syllabus | – |
| Push to GitHub; host it (Streamlit Cloud, a GPU server or the DGX) | – | Team: say the word and I prepare and run it | `docs/DEPLOYMENT.md`, `docs/GPU_SERVER.md` |

**Blocked by hardware or access:**
- **qwen2.5vl:7b image comparison:** needs ~8 GB of free RAM on this laptop (it has ~3 GB with the app open) or a GPU. Run `run_cpu_evaluation.cmd --with-7b --stages images7b` with other programs closed, or on Colab or the DGX.
- **Optional:** a larger NLI model or an LLM judge for reasoning errors.
