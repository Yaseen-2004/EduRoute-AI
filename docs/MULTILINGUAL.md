# Multilingual EduRoute

Students can ask in, and get answers in, Indian languages: by typing in their own
script or by speaking. EduRoute still **searches, routes and checks in English**: only
the question going in and the already-checked answer coming out are translated.
Grounding, citations and verification are therefore exactly as in English.

```
question (हिन्दी / ಕನ್ನಡ / తెలుగు / தமிழ் / …)  ──NLLB──►  English question
        ──► routing · retrieval · generation · verification (English, unchanged)
        ──► checked English answer  ──NLLB──►  answer in the student's language
                                    └── "English original" kept one click away
```

## Components

| Part | Model | Runs | Notes |
|---|---|---|---|
| Translation | Meta NLLB-200 distilled 600M, CTranslate2 int8 (623 MB) | offline, CPU | Licence **CC-BY-NC 4.0** (research and non-commercial use) |
| Speech in Indian languages | Whisper `small` (multilingual) via faster-whisper | offline, CPU | The chosen language is given to Whisper as a hint |
| Speech in English | Whisper `base.en` | offline, CPU | 4.3% WER; `small.en` was 3.6% but 2.5× slower (17 Nov) |
| Read-aloud | Edge neural voices per language (e.g. hi-IN-SwaraNeural), then gTTS | online | The offline Windows voice speaks English only, so it is not used for Indian languages |

No downloaded model code runs: the translation model is plain weights executed by
CTranslate2, and the tokenizer is the standard NLLB tokenizer in `transformers`.
(AI4Bharat's IndicTrans2 scores higher on Indian languages but needs its own model
code to run, `trust_remote_code`. It was not used without explicit approval.)

## What translation protects

The translation keeps the Markdown structure: headings, bullets, numbered lists, tables and code blocks. It also never sends these to the model:

- **Citations** `[1]`, `[2][3]`, whether they sit before or after the full stop.
- **Inline code, formulas and URLs** (`$\eta = W/Q$`, `` `pip install x` ``).
- **Technical terms**, which stay in English inside the sentence, as in Indian engineering classrooms. The terms come from the knowledge base's own 1,462 article titles and 1,435 syllabus topics.
  - Without this, NLLB turned "steam" into "fog" (Tamil) and "deadlock" into "passivity" (Kannada) on 17 Nov.

## Setup (once, about 2.5 GB download, about 5 min conversion)

```powershell
venv\Scripts\ct2-transformers-converter --model facebook/nllb-200-distilled-600M ^
    --output_dir models\translation\nllb-200-600M-ct2-int8 --quantization int8 ^
    --copy_files tokenizer.json sentencepiece.bpe.model tokenizer_config.json special_tokens_map.json
```

The app offers other languages only when `models/translation/nllb-200-600M-ct2-int8/model.bin`
exists. Whisper `small` downloads on the first spoken question in an Indian language.

## Quality gate

```powershell
venv\Scripts\python -m evaluation.evaluate_translation --answers 8 --keep-terms on
venv\Scripts\python -m evaluation.evaluate_asr --make-synthetic 10 --language hi   # speech, per language
```

A language is offered (`ENABLED` in `src/language/languages.py`) only if real answers survive translation into it and back. The thresholds were fixed before looking at the results:

- **chrF ≥ 50:** the round-trip text still matches the original.
- **Technical-term survival ≥ 0.6:** most technical terms come back intact.

Speech is scored by character error rate (CER), the usual measure for Indian scripts. Results are in `docs/PROGRESS.md`.
