# EduRoute AI

An adaptive LLM + RAG + multimodal + voice assistant for B.Tech students.
For what is implemented and the measured results, see [docs/PROGRESS.md](docs/PROGRESS.md).

## Quick start (Windows)

```powershell
venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env          # then edit if needed
venv\Scripts\streamlit run app.py
```

The app works without a language model. In that mode it shows clearly
labelled source excerpts. To get generated answers, start Qwen2.5-3B-Instruct
using one of the options below.

### Option A: Ollama on this laptop (CPU, simplest)

```powershell
winget install Ollama.Ollama
ollama pull qwen2.5:3b            # ~2 GB, 4-bit quantised
ollama pull qwen2.5vl:3b          # optional: image questions
```

The `.env` defaults already point to `http://localhost:11434/v1`. For image
questions, also set:

```
EDUROUTE_VISION_BASE_URL=http://localhost:11434/v1
EDUROUTE_VISION_MODEL=qwen2.5vl:3b
```

### Option B: vLLM on Google Colab (T4 GPU, report §9)

```python
!pip install vllm
!vllm serve Qwen/Qwen2.5-3B-Instruct --dtype half --max-model-len 8192 &
# expose port 8000 with a tunnel (e.g. cloudflared) and copy the https URL
```

Then in `.env`:

```
EDUROUTE_LLM_BASE_URL=https://<tunnel>/v1
EDUROUTE_LLM_MODEL=Qwen/Qwen2.5-3B-Instruct
```

### Option C: system benchmark on Colab (recommended on a laptop CPU)

On this laptop, Qwen2.5-3B runs at about 1.3 tokens/s, so one answer takes
minutes. To run the benchmark on a free T4 GPU instead:

1. `venv\Scripts\python colab\make_bundle.py` creates `colab/eduroute_colab_bundle.zip` (code, configs, benchmarks, index; about 60 MB, no `.env`).
2. Open `colab/EduRoute_Colab_Benchmark.ipynb` in Google Colab, set *Runtime → T4 GPU* and run all cells.
3. Copy the downloaded `eduroute_benchmark_results.zip` into `evaluation/results/`.

## Pipelines

The app uses the **engineering-wide index** (all 26 catalogue groups, 289 subjects)
when `vector_store/engineering_faiss_v1.index` exists, otherwise the AIML-only V4
index. Override with `EDUROUTE_RAG_VERSION=v4|engineering_v1`.

| Task | Command |
|---|---|
| Collect the engineering knowledge base (all branches; cached and resumable) | `venv\Scripts\python -m src.data_collection.build_engineering_corpus` |
| Build the engineering index (~35k chunks; about 1 h on a laptop CPU, minutes on a GPU) | `venv\Scripts\python -m src.rag.build_index_v4 --dataset engineering_v1` |
| Rebuild the AIML V4 dataset (cached fetch) | `venv\Scripts\python -m src.data_collection.prepare_wikipedia_v4` |
| Rebuild the AIML V4 index (~5 min) | `venv\Scripts\python -m src.rag.build_index_v4 [--include-documents]` |
| Cross-branch retrieval benchmark | `venv\Scripts\python -m evaluation.evaluate_retrieval --benchmark retrieval_engineering_v1.jsonl --configs engineering_v1:dense [--subject-filter [widen]]` (`widen` = the app's subject → branch → all fallback) |
| Router calibration for the engineering index | `venv\Scripts\python -m evaluation.calibrate_router --version engineering_v1` |
| Precompute verification evidence (~15 min) | `venv\Scripts\python -m src.verification.evidence_cache` |
| Ingest a permitted PDF | `venv\Scripts\python -m src.data_collection.pdf_ingestion notes.pdf --title ... --subject-id AIML005 --topic CNN --license "..."` |
| Retrieval smoke test | `venv\Scripts\python -m src.rag.retrieval` |
| Retrieval benchmark | `venv\Scripts\python -m evaluation.evaluate_retrieval` |
| Router calibration | `venv\Scripts\python -m evaluation.calibrate_router` |
| System benchmark (5 strategies) | `venv\Scripts\python -m evaluation.run_system_benchmark` |
| Router check on held-out questions | `venv\Scripts\python -m evaluation.check_router_heldout` |
| Image questions (needs a vision model) | `venv\Scripts\python -m evaluation.run_image_benchmark` (starter set: `-m evaluation.make_image_starter_set`) |
| Voice word error rate | `venv\Scripts\python -m evaluation.evaluate_asr <folder>` or `--make-synthetic 20` |
| Export / score claim labels | `venv\Scripts\python -m evaluation.claim_labels export <partial.jsonl>` then `score <labels.csv>` |
| Analyse human grading | `venv\Scripts\python -m evaluation.analyse_grading --key <key.json> --benchmark <run.json> rater1.csv rater2.csv` |
| Resume an interrupted benchmark | add `--resume evaluation\results\<partial>.jsonl` (and `--keep-awake` on a laptop) |
| Delete old logs now | `venv\Scripts\python -m src.utils.retention` (automatic in the app; `EDUROUTE_LOG_RETENTION_DAYS`) |
| Tests | `venv\Scripts\python -m pytest -q` (add `-m "not slow"` to skip the app test) |

## Layout

```
app.py                      Streamlit UI
config/settings.py          configuration (reads .env)
config/disciplines.json     26 catalogue groups; config/branches/*.json = subjects and topics
config/topic_hints.json     preferred source page for generic topic names
config/router_calibration_<index>.json   thresholds per index (v4, engineering_v1)
src/data_collection/        Wikipedia collector, V4 preparation, PDF ingestion
src/rag/                    cleaning, V4 chunk/embed/index build, retrieval
src/llm/                    backends (Ollama/vLLM, transformers, extractive) and prompts
src/router/                 query analysis and adaptive router
src/verification/           claims, evidence mapping, selective verification
src/vision/                 image validation, vision model client
src/voice/                  Whisper ASR, Edge-TTS / gTTS
src/pipeline.py             orchestration and telemetry
evaluation/                 benchmarks, evaluators, rubric, results
tests/                      unit and app tests
```

V2/V3 artifacts and scripts are kept unchanged for reproducibility of the
report's numbers.
