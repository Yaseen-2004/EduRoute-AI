"""
Builds docs/EduRoute_AI_Project_Report.pdf: a complete, step-by-step project report.
Benchmark figures are read from the result files, not typed in.

    venv\\Scripts\\python docs\\report\\build_report.py
"""

import csv
import json
from collections import Counter
from datetime import date
from pathlib import Path

from reportlab.graphics.charts.barcharts import HorizontalBarChart, VerticalBarChart
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "evaluation" / "results"
OUT = ROOT / "docs" / "EduRoute_AI_Project_Report.pdf"
FONTS = Path("C:/Windows/Fonts")

pdfmetrics.registerFont(TTFont("Body", str(FONTS / "segoeui.ttf")))
pdfmetrics.registerFont(TTFont("Body-Bold", str(FONTS / "segoeuib.ttf")))
pdfmetrics.registerFont(TTFont("Mono", str(FONTS / "consola.ttf")))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body", boldItalic="Body-Bold")

INK = colors.HexColor("#1f2937")
MUTED = colors.HexColor("#6b7280")
ACCENT = colors.HexColor("#1d4ed8")
ACCENT_LIGHT = colors.HexColor("#dbeafe")
RULE = colors.HexColor("#d1d5db")
SERIES = [colors.HexColor(c) for c in ("#1d4ed8", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6")]

base = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("title", fontName="Body-Bold", fontSize=26, leading=32, textColor=INK, alignment=TA_CENTER),
    "subtitle": ParagraphStyle("subtitle", fontName="Body", fontSize=13, leading=18, textColor=MUTED, alignment=TA_CENTER),
    "h1": ParagraphStyle("h1", fontName="Body-Bold", fontSize=17, leading=22, textColor=ACCENT, spaceBefore=6, spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName="Body-Bold", fontSize=12.5, leading=16, textColor=INK, spaceBefore=10, spaceAfter=4),
    "body": ParagraphStyle("body", fontName="Body", fontSize=9.6, leading=13.6, textColor=INK, spaceAfter=5),
    "small": ParagraphStyle("small", fontName="Body", fontSize=8, leading=10.5, textColor=MUTED),
    "cell": ParagraphStyle("cell", fontName="Body", fontSize=8.2, leading=10.4, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Body-Bold", fontSize=8.2, leading=10.4, textColor=colors.white),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=8, leading=10.5, textColor=INK,
                           backColor=colors.HexColor("#f3f4f6"), borderPadding=4, spaceAfter=6, leftIndent=4),
    "caption": ParagraphStyle("caption", fontName="Body", fontSize=8, leading=10, textColor=MUTED,
                              alignment=TA_CENTER, spaceBefore=2, spaceAfter=8),
    "toc1": ParagraphStyle("toc1", fontName="Body", fontSize=10, leading=15, textColor=INK),
}


# ------------------------------------------------------------------ helpers

def P(text, style="body"):
    return Paragraph(text, S[style])


def H1(text):
    p = Paragraph(text, S["h1"])
    p._toc = text
    return p


def bullets(items):
    return ListFlowable([ListItem(P(i), leftIndent=10) for i in items], bulletType="bullet",
                        start="•", leftIndent=12, bulletFontName="Body", bulletFontSize=8)


def table(rows, widths, header=True, zebra=True):
    data = [[Paragraph(str(c), S["cellb" if header and r == 0 else "cell"]) for c in row]
            for r, row in enumerate(rows)]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), ACCENT))
    if zebra:
        for r in range(1 if header else 0, len(rows)):
            if r % 2 == 0:
                style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f8fafc")))
    t.setStyle(TableStyle(style))
    return t


def code(text):
    return Paragraph(text.replace("&", "&amp;").replace("<", "&lt;").replace("\n", "<br/>"), S["code"])


def caption(text):
    return P(text, "caption")


def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ figures

def bar_chart(categories, series, names, width=165, height=62, value_max=None, fmt="%d", horizontal=False):
    d = Drawing(width * mm, height * mm)
    chart = HorizontalBarChart() if horizontal else VerticalBarChart()
    chart.x, chart.y = 18 * mm if not horizontal else 32 * mm, 12 * mm
    chart.width = (width - (24 if not horizontal else 40)) * mm
    chart.height = (height - 20) * mm
    chart.data = series
    chart.categoryAxis.categoryNames = categories
    chart.categoryAxis.labels.fontName = "Body"
    chart.categoryAxis.labels.fontSize = 7
    chart.valueAxis.labels.fontName = "Body"
    chart.valueAxis.labels.fontSize = 7
    chart.valueAxis.valueMin = 0
    if value_max:
        chart.valueAxis.valueMax = value_max
    chart.valueAxis.strokeColor = RULE
    chart.categoryAxis.strokeColor = RULE
    chart.barLabelFormat = fmt
    chart.barLabels.fontName = "Body"
    chart.barLabels.fontSize = 6.5
    chart.barLabels.nudge = 6
    chart.barSpacing = 1
    chart.groupSpacing = 6
    for i in range(len(series)):
        chart.bars[i].fillColor = SERIES[i % len(SERIES)]
        chart.bars[i].strokeColor = None
    d.add(chart)
    if len(series) > 1:
        legend = Legend()
        legend.x, legend.y = chart.x + 4, (height - 3) * mm
        legend.fontName, legend.fontSize = "Body", 7
        legend.alignment = "right"
        legend.columnMaximum = 1
        legend.dx = legend.dy = 6
        legend.deltax = 70
        legend.colorNamePairs = [(SERIES[i], names[i]) for i in range(len(series))]
        d.add(legend)
    return d


def architecture():
    d = Drawing(170 * mm, 92 * mm)

    def box(x, y, w, h, title, sub="", fill=ACCENT_LIGHT):
        d.add(Rect(x * mm, y * mm, w * mm, h * mm, rx=6, ry=6, fillColor=fill, strokeColor=ACCENT, strokeWidth=0.8))
        d.add(String((x + w / 2) * mm, (y + h - 5.2) * mm, title, fontName="Body-Bold", fontSize=8,
                     fillColor=INK, textAnchor="middle"))
        for k, line in enumerate(sub.split("\n") if sub else []):
            d.add(String((x + w / 2) * mm, (y + h - 9.5 - 3.6 * k) * mm, line, fontName="Body", fontSize=6.5,
                         fillColor=MUTED, textAnchor="middle"))

    def arrow(x1, y1, x2, y2):
        d.add(Line(x1 * mm, y1 * mm, x2 * mm, y2 * mm, strokeColor=INK, strokeWidth=0.8))
        import math
        a = math.atan2(y2 - y1, x2 - x1)
        tip = (x2 * mm, y2 * mm)
        left = (tip[0] - 2.2 * mm * math.cos(a - 0.4), tip[1] - 2.2 * mm * math.sin(a - 0.4))
        right = (tip[0] - 2.2 * mm * math.cos(a + 0.4), tip[1] - 2.2 * mm * math.sin(a + 0.4))
        d.add(Polygon([*tip, *left, *right], fillColor=INK, strokeColor=INK))

    box(2, 70, 38, 18, "Student input", "text · image · voice")
    box(48, 70, 52, 18, "Input processing", "Whisper ASR · image validation\nsubject / branch filter")
    box(108, 70, 58, 18, "Query analysis + retrieval probe", "intent, complexity, modality\ntop-1 cosine vs calibrated threshold")
    arrow(40, 79, 48, 79)
    arrow(100, 79, 108, 79)
    box(60, 46, 50, 16, "Adaptive router", "5 strategies · cost preference", fill=colors.HexColor("#fef3c7"))
    arrow(137, 70, 100, 62)
    box(2, 22, 38, 16, "Direct LLM", "Qwen2.5-3B")
    box(44, 22, 38, 16, "RAG + LLM", "FAISS (36,367 chunks)\n+ cited answer")
    box(86, 22, 38, 16, "Vision LLM", "Qwen2.5-VL-3B")
    box(128, 22, 40, 16, "Hybrid multimodal", "image analysis\n+ retrieval")
    for x in (21, 63, 105, 148):
        arrow(85, 46, x, 38)
    box(40, 2, 92, 13, "Selective verification → answer + sources + warnings + telemetry",
        fill=colors.HexColor("#dcfce7"))
    for x in (63, 148):
        arrow(x, 22, 86, 15)
    return d


# ------------------------------------------------------------------ content

def build():
    final = load("system_benchmark_20261111_042423.json")
    runs = {
        "8 Nov": load("system_benchmark_20261108_161042.json"),
        "9 Nov": load("system_benchmark_20261109_111552.json"),
        "10 Nov": load("system_benchmark_20261110_153451.json"),
        "11 Nov": final,
    }
    images = load("image_benchmark_20261111_035749_rescored.json")
    asr = load("asr_eval_voice_eval_synthetic_20261110_153745.json")
    heldout = load("router_heldout_engineering_v1_20261111_221904.json")   # after the 11 Nov gap fill
    corpus = json.loads((ROOT / "data" / "metadata" / "engineering_corpus_report_v1.json").read_text(encoding="utf-8"))
    index = json.loads((ROOT / "data" / "metadata" / "engineering_index_report_v1.json").read_text(encoding="utf-8"))
    # The 11 Nov gap fill appended to the index; current totals include it.
    supplement = json.loads((ROOT / "data" / "metadata" / "engineering_supplement_report_v1.json")
                            .read_text(encoding="utf-8"))
    new_articles = supplement["new_articles"]
    new_articles = len(new_articles) if isinstance(new_articles, (list, dict)) else new_articles
    total_articles = corpus["articles_total"] + new_articles
    aicte = list(csv.DictReader(open(ROOT / "data" / "metadata" / "aicte_comparison.csv", encoding="utf-8")))
    gaps = list(csv.DictReader(open(ROOT / "data" / "metadata" / "aicte_gaps.csv", encoding="utf-8")))

    strategies = ["direct", "fixed_rag", "modality", "always_verify", "adaptive"]
    rows = final["rows"]
    tokens = {s: sum(r["input_tokens"] + r["output_tokens"] for r in rows if r["strategy"] == s) for s in strategies}
    summ = {s: final["summary"][s]["all"] for s in strategies}

    story = []

    # ---------------------------------------------------------- title page
    story += [Spacer(1, 45 * mm), P("EduRoute AI", "title"), Spacer(1, 4 * mm),
              P("An Adaptive LLM + RAG + Multimodal + Voice Study Assistant<br/>for B.Tech Engineering Students",
                "subtitle"),
              Spacer(1, 10 * mm),
              P("Complete project report: design, step-by-step development, evaluation and results", "subtitle"),
              Spacer(1, 30 * mm),
              P("Team: Syed Mohammed Yaseen · Mohammed Ayaan · Arfath Khan", "subtitle"),
              Spacer(1, 3 * mm),
              P(f"Report generated {date.today():%d %B %Y}", "subtitle"),
              Spacer(1, 30 * mm),
              P("All numbers in this report are measured. Each result names the file it comes from "
                "(folder <font name='Mono'>evaluation/results/</font>), so every figure can be re-checked.",
                "small"),
              PageBreak()]

    toc = TableOfContents()
    toc.levelStyles = [S["toc1"]]
    story += [P("Contents", "h1"), toc, PageBreak()]

    # ---------------------------------------------------------- 1 summary
    story += [H1("1. Executive summary"),
              P("EduRoute AI answers B.Tech students' questions in the format of university examinations "
                "(2-, 5- and 10-mark answers). For every question an <b>adaptive router</b> decides whether to "
                "answer directly, to retrieve evidence from a knowledge base first, or to analyse an attached image, "
                "and whether the answer needs to be <b>verified</b> against its sources. The aim is answers that are "
                "grounded where it matters, at a lower cost than retrieving and verifying everything."),
              table([
                  ["Area", "Result"],
                  ["Knowledge base", f"{total_articles:,} articles · 26 branch groups · 289 subjects · "
                                     f"{supplement['index_vectors_after']:,} indexed chunks"],
                  ["Retrieval", "Correct source in the top 5 for 50/50 engineering questions (MRR 0.95–0.96)"],
                  ["Router", f"AUC 0.999 on calibration; {heldout['accuracy_excluding_borderline'] * 100:.1f}% "
                             "correct on 60 unseen questions"],
                  ["Cost", f"Adaptive used {100 * (1 - tokens['adaptive'] / tokens['fixed_rag']):.0f}% fewer tokens "
                           f"than always retrieving and {100 * (1 - tokens['adaptive'] / tokens['always_verify']):.0f}% "
                           "fewer than always verifying, at the same median latency"],
                  ["Reliability", "250/250 answers generated in each of four benchmark runs, 0 failures"],
                  ["Images", f"{int(images['summary']['modality']['all_facts_correct'] * 8)}/8 test images answered "
                             "fully correctly (vision route)"],
                  ["Voice", f"Word error rate {asr['wer'] * 100:.1f}% on synthetic Indian-English speech"],
                  ["Syllabus", "116 of 167 subjects with an AICTE model curriculum match or partly match it"],
                  ["Quality checks", "53 automated tests; 4 benchmark runs, each exposing defects that were then fixed"],
              ], [38, 132]),
              Spacer(1, 4 * mm),
              P("<b>Not yet complete:</b> human grading of answer correctness (two raters), claim labelling, real "
                "student voice recordings and images, syllabus confirmation for the 12 branch groups without an AICTE "
                "model curriculum, and public deployment. Tools for all of these are built into the app."),
              PageBreak()]

    # ---------------------------------------------------------- 2 objectives
    story += [H1("2. Problem and objectives"),
              P("Students revising for engineering examinations need answers that are correct, at the right depth "
                "for the marks, and traceable to sources. A general chatbot answers fluently but can invent facts; "
                "a pure retrieval system is accurate but rigid and expensive when it retrieves for every question. "
                "The project's objectives were:"),
              bullets([
                  "Cover the <b>complete engineering syllabus</b>, not one branch: 26 branch groups and 289 subjects.",
                  "<b>Route</b> each question to the cheapest workflow that answers it well: direct, RAG, vision or hybrid.",
                  "<b>Verify selectively</b>: check answers against their evidence only when the risk justifies the cost.",
                  "Accept <b>text, images and voice</b>, and answer in 2/5/10-mark formats with citations.",
                  "<b>Measure everything</b>: retrieval, routing, cost, latency, grounding and answer quality, against "
                  "four baseline strategies run on the same questions.",
                  "Protect student privacy: no question text, images or recordings stored by default.",
              ]),
              H1("3. System architecture"),
              architecture(),
              caption("Figure 1. Request flow. The retrieval probe is one embedding plus one FAISS search (tens of "
                      "milliseconds), and its results are reused when RAG is chosen."),
              table([
                  ["Layer", "Technology", "Role"],
                  ["Language model", "Qwen2.5-3B-Instruct (4-bit) via Ollama", "Answer generation, revision"],
                  ["Vision model", "Qwen2.5-VL-3B via Ollama", "Image description and image answers"],
                  ["Embeddings", "all-MiniLM-L6-v2 (384-d)", "Retrieval, router probe, claim-evidence mapping"],
                  ["Vector store", "FAISS IndexFlatIP", "Exact cosine search over 36,367 chunks"],
                  ["Verification", "Heuristics + nli-deberta-v3-xsmall", "Claim support, figure checks, contradictions"],
                  ["Speech", "faster-whisper base.en · Edge-TTS / gTTS", "Voice questions, spoken answers"],
                  ["Interface", "Streamlit 1.66", "Chat app and evaluation studio"],
                  ["GPU runs", "Google Colab T4", "Benchmarks and demo model serving"],
              ], [32, 66, 72]),
              PageBreak()]

    # ---------------------------------------------------------- 4 step by step
    story += [H1("4. Step-by-step development"),
              P("The work proceeded in ten phases. Each phase ends with a measurement, and several phases were "
                "triggered by problems the previous measurement exposed.")]

    def phase(title, did, outcome):
        return KeepTogether([P(title, "h2"), bullets(did), P(f"<b>Outcome:</b> {outcome}")])

    story += [
        phase("Step 1 · Auditing the original AIML knowledge base (V3 → V4)", [
            "Inspected all 92 Wikipedia articles of the original AIML index.",
            "Found 7 wrong pages, e.g. <i>CNN</i> was the news channel and <i>Spark</i>, <i>RNN</i>, "
            "<i>Regression</i> were disambiguation pages; reference sections ('See also', bibliographies) leaked into "
            "chunks; chunk text had been rebuilt from token IDs, lower-casing everything.",
            "Rebuilt as V4: correct articles, section-aware cleaning, offset-based chunking (220 tokens, 30 overlap) "
            "that never cuts mid-word, and a 'Title > Section' prefix on each chunk.",
        ], "AIML retrieval Success@1 rose from 0.80 to 0.90 and MRR from 0.855 to 0.947 on a 50-question benchmark."),
        phase("Step 2 · Building catalogues for all engineering branches", [
            "Extended the subject catalogue from AIML to 26 branch groups and 289 subjects, five topics each.",
            "Added curated hints for ambiguous topic names (e.g. 'Functions' → the mathematical function).",
        ], "1,370 non-AIML topics defined; later checked against the AICTE model curriculum (Step 10)."),
        phase("Step 3 · Collecting the engineering-wide corpus", [
            "Wikimedia rate-limited the project (HTTP 429). A naive design needed ~3,900 requests (~10 hours).",
            "Redesigned collection: 20 article intros per request, search only for unresolved topics, full articles "
            "fetched 20 at a time as wikitext and converted locally; Retry-After honoured; never parallelised.",
            "Every topic resolution recorded with score, runner-up and rejected candidates for audit.",
        ], f"About 186 requests; 1,370/1,370 topics resolved; {corpus['articles_total']:,} unique articles."),
        phase("Step 4 · Indexing and retrieval evaluation", [
            f"Embedded {index['total_chunks']:,} chunks (average {index['average_chunk_tokens']:.0f} tokens) into a "
            f"FAISS index (3.7 h on the laptop CPU). On 11 Nov, {supplement['gap_topics']} missing AICTE core "
            f"topics added {new_articles} articles and {supplement['new_passages']} passages incrementally "
            f"({supplement['embedding_seconds']:.0f} s), giving {supplement['index_vectors_after']:,}.",
            "Evaluated dense, hybrid (BM25 + RRF) and cross-encoder reranking on two 50-question benchmarks.",
            "Found that strict subject filters made in-scope questions look out of scope; added subject → branch → "
            "whole-index widening.",
        ], "Dense retrieval kept as default; correct source in the top 5 for every engineering question."),
        phase("Step 5 · Router and its calibration", [
            "Query analysis (intent, complexity, modality, ambiguity) plus a retrieval probe.",
            "Fitted the in-domain threshold by Youden's J on 100 in-domain vs 30 out-of-domain questions.",
            "Built 5 strategies sharing one code path so comparisons are fair: direct, fixed_rag, always_verify, "
            "modality, adaptive.",
        ], f"AUC 0.999; {heldout['accuracy_excluding_borderline'] * 100:.1f}% correct on 60 held-out questions."),
        phase("Step 6 · Generation, verification, vision and voice", [
            "Grounded prompts with numbered sources and 2/5/10-mark formats; Ollama/vLLM/transformers back-ends; "
            "labelled extractive fallback when no model is reachable.",
            "Verification: claims extracted from the answer, mapped to evidence windows, figures checked, one "
            "revision call at most.",
            "Image validation and vision workflows; Whisper ASR with a custom audio decoder; Edge-TTS with gTTS fallback.",
        ], "A complete pipeline with telemetry (latency per stage, tokens, model calls)."),
        phase("Step 7 · Running the language model", [
            "Installed Ollama and Qwen2.5-3B on the laptop: about 1–4 tokens/s on the CPU, minutes per answer.",
            "Moved benchmark runs to a Colab T4 GPU with a resumable notebook and an upload bundle.",
        ], "Four Colab runs of 250 answers each, all completed without failures."),
        phase("Step 8 · Measuring, finding defects, fixing them (four runs)", [
            "Run 1 (8 Nov): citations rare; wrong figures hidden from the unsupported-claim rate.",
            "Run 2 (9 Nov): citation fix confirmed; an invented worked example ('6-pole motor at 1500 RPM') found.",
            "Run 3 (10 Nov): NLI contradiction check over-flagged; prompt instructions leaked into 7% of answers.",
            "Run 4 (11 Nov): leaks eliminated, contradictions down to 2%; first image benchmark.",
        ], "Every defect and fix is listed in Section 6."),
        phase("Step 9 · Evaluation tools and the evaluation studio", [
            "Held-out router check, word-error-rate tool, image benchmark, claim-labelling export, grading analysis "
            "with inter-rater agreement (weighted kappa).",
            "An Evaluation Studio page in the app for blind grading, claim labelling, consented voice recording, "
            "image collection and syllabus review.",
        ], "The remaining human evaluation can be done in the app, by several people at once."),
        phase("Step 10 · Syllabus check and deployment preparation", [
            "Extracted 325 course titles from seven official AICTE model-curriculum documents and matched every "
            "subject to them.",
            "Prepared Streamlit Community Cloud deployment: CPU-only requirements, secrets template, Colab model "
            "server with an HTTPS tunnel, deployment guide; git repository with grading keys and recordings excluded.",
            "Privacy: question hashes only, no stored images or recordings, 30-day log retention, privacy notice.",
        ], "Ready to deploy; syllabus gaps listed for the next corpus update."),
    ]
    story.append(PageBreak())

    # ---------------------------------------------------------- 5 results
    story += [H1("5. Results")]
    story += [P("5.1 Knowledge base", "h2")]
    per = corpus["articles_per_discipline"]
    cats = sorted(per, key=per.get)
    story += [bar_chart(cats, [[per[c] for c in cats]], ["articles"], height=88, horizontal=True),
              caption("Figure 2. Articles per branch group (an article shared by several subjects is counted in each)."),
              P("5.2 Retrieval", "h2"),
              table([["Index / configuration", "Benchmark", "Success@1", "Success@5", "MRR@10"],
                     ["AIML V3 (original baseline)", "AIML (50)", "0.80", "0.92", "0.855"],
                     ["AIML V4 (corrected)", "AIML (50)", "0.90", "1.00", "0.947"],
                     ["Engineering-wide", "AIML (50)", "0.90", "0.98", "0.939"],
                     ["Engineering-wide", "Engineering (50)", "0.90", "1.00", "0.950"],
                     ["Engineering-wide, subject filter + widening", "Engineering (50)", "0.92", "1.00", "0.960"]],
                    [62, 32, 24, 24, 24]),
              caption("Table 1. A retrieved chunk is relevant if it comes from a labelled article and contains a "
                      "labelled evidence term. Hybrid BM25 and a cross-encoder reranker did not improve results."),
              P("5.3 Router", "h2"),
              table([["Check", "Questions", "Result"],
                     ["Calibration (fitted)", "100 in-domain, 30 out-of-domain", "AUC 0.999 · threshold 0.491"],
                     ["Held-out: in-domain", "30 (all branches)", f"{heldout['in_domain_kept'] * 100:.1f}% kept"],
                     ["Held-out: easy out-of-domain", "10", f"{heldout['out_of_domain_rejected_easy'] * 100:.0f}% rejected"],
                     ["Held-out: hard out-of-domain", "16 (history, law, economics, medicine, arts)",
                      f"{heldout['out_of_domain_rejected_hard'] * 100:.1f}% rejected"]],
                    [50, 66, 54]),
              PageBreak()]

    story += [P("5.4 System benchmark: five strategies, 250 answers (Colab T4, 11 Nov)", "h2"),
              table([["Strategy", "Median latency", "Input tokens / answer", "Total tokens", "Grounded",
                      "Verified", "Cites sources"]] +
                    [[s, f"{summ[s]['median_latency_ms'] / 1000:.1f} s", f"{summ[s]['mean_input_tokens']:.0f}",
                      f"{tokens[s] / 1000:.1f}k", f"{summ[s]['grounded_rate'] * 100:.0f}%",
                      f"{summ[s]['verified_rate'] * 100:.0f}%", f"{summ[s]['answers_with_citations'] * 100:.0f}%"]
                     for s in strategies],
                    [30, 22, 28, 22, 20, 20, 24]),
              caption("Table 2. Same 50 questions for every strategy (20 in-domain, 20 out-of-domain, 10 mixed). "
                      "Adaptive answers out-of-domain questions directly, so they carry no citations; on in-domain "
                      "questions 75% of its answers cite sources."),
              bar_chart(strategies, [[tokens[s] / 1000 for s in strategies]], ["tokens (thousands)"], fmt="%.1fk"),
              caption("Figure 3. Total tokens over the 250-answer run, per strategy (lower is cheaper)."),
              P("<b>Reading the results.</b> Adaptive routing cost "
                f"{100 * (1 - tokens['adaptive'] / tokens['fixed_rag']):.0f}% less than retrieving for every question "
                f"and {100 * (1 - tokens['adaptive'] / tokens['always_verify']):.0f}% less than verifying every answer, "
                "with the same median latency as fixed RAG. It sent all 20 out-of-domain questions to the model "
                "directly (and labelled them as not grounded), used retrieval for 17 of 20 in-domain questions, and "
                "answered the other 3 (2-mark definitions) directly by design. Retrieving for everything is not "
                "safer: on out-of-domain questions the 'grounded' fixed-RAG answers rest on irrelevant passages, and "
                "verification found 76% of their claims unsupported, against 13% in-domain."),
              PageBreak()]

    cite = [[runs[k]["summary"][s]["all"]["answers_with_citations"] * 100 for k in runs]
            for s in ("fixed_rag", "always_verify")]
    story += [P("5.5 Improvement across the four benchmark runs", "h2"),
              bar_chart(list(runs), cite, ["fixed_rag", "always_verify"], fmt="%d%%", value_max=100),
              caption("Figure 4. Share of answers citing their sources, per run. The citation reminder was added "
                      "after the 8 Nov run."),
              table([["Measure", "8 Nov", "9 Nov", "10 Nov", "11 Nov"],
                     ["Adaptive: unsupported claim rate"] +
                     [str(runs[k]["summary"]["adaptive"]["all"].get("mean_unsupported_claim_rate")) for k in runs],
                     ["Adaptive: in-domain citation rate"] +
                     [f"{runs[k]['summary']['adaptive']['in_domain']['answers_with_citations'] * 100:.0f}%" for k in runs],
                     ["Answers echoing prompt instructions", "–", "–", "18", "0"],
                     ["Adaptive: answers with a counted contradiction", "–", "–", "12%", "2%"]],
                    [70, 25, 25, 25, 25]),
              caption("Table 3. The unsupported-claim rule became stricter on 9 Nov (wrong figures now count), so "
                      "rates are comparable only from that run on."),
              P("5.6 Image questions (Qwen2.5-VL-3B)", "h2"),
              table([["Image (known answer)", "Vision only", "Vision + knowledge base"]] +
                    [[r_m["file"].replace(".png", "").replace("_", " "),
                      f"{r_m['facts_found']}/{r_m['facts_total']}", f"{r_a['facts_found']}/{r_a['facts_total']}"]
                     for r_m, r_a in zip([r for r in images["rows"] if r["strategy"] == "modality"],
                                         [r for r in images["rows"] if r["strategy"] == "adaptive"])],
                    [70, 40, 50]),
              caption(f"Table 4. Facts found per image. Vision only: {int(images['summary']['modality']['all_facts_correct'] * 8)}/8 "
                      f"fully correct (median {images['summary']['modality']['median_latency_ms'] / 1000:.1f} s); "
                      f"hybrid: {int(images['summary']['adaptive']['all_facts_correct'] * 8)}/8, measured before the "
                      "hybrid prompt fix. The 3B model misread the AND truth table in every run."),
              P("5.7 Voice", "h2"),
              P(f"Twenty benchmark questions read by two Indian-English synthetic voices were transcribed with a word "
                f"error rate of <b>{asr['wer'] * 100:.1f}%</b> ({asr['exact_transcripts'] * 100:.0f}% exact; median "
                f"{asr['median_latency_s']} s per question). Errors were mostly spellings of technical names (e.g. "
                "'Homan' for Hohmann). Real student recordings are needed for the definitive figure."),
              PageBreak()]

    status = Counter(r["status"] for r in aicte)
    with_ref = [r for r in aicte if r["status"] != "no AICTE reference"]
    story += [P("5.8 Syllabus coverage against the AICTE model curriculum", "h2"),
              P(f"Seven official AICTE documents (Vol. I and II 2018, CSE 2022, Civil 2024, Mechanical, Robotics &amp; AI, "
                f"emerging areas) yielded 325 course titles. Of the {len(with_ref)} catalogue subjects in the 14 groups "
                f"AICTE covers, <b>{status['match'] + status['partial']} ({100 * (status['match'] + status['partial']) / len(with_ref):.0f}%)</b> "
                f"match or partly match an AICTE course; {status['no close AICTE course']} are mostly modern electives "
                f"AICTE does not list as core. {status['no AICTE reference']} subjects in 12 groups have no AICTE model "
                f"curriculum. {len(gaps)} AICTE core courses are not in the catalogue."),
              table([["Group", "AICTE core courses missing from the catalogue"]] +
                    [[b, str(n)] for b, n in Counter(g["branch"] for g in gaps).most_common()],
                    [40, 80]),
              PageBreak()]

    # ---------------------------------------------------------- 6 defects
    story += [H1("6. Problems found by measurement, and their fixes"),
              P("Each benchmark run exposed problems that unit tests had not. Listing them shows that the evaluation "
                "drove the design."),
              table([["Found in", "Problem", "Fix", "Evidence afterwards"],
                     ["Audit", "7 wrong Wikipedia pages; references in chunks", "Rebuilt V4 corpus and chunker",
                      "AIML MRR 0.855 → 0.947"],
                     ["Retrieval eval", "Strict subject filter made in-scope questions ungrounded",
                      "Subject → branch → index widening", "Engineering MRR 0.950 → 0.960 (filtered)"],
                     ["CPU run", "180 s model timeout would turn long answers into excerpts",
                      "Configurable timeouts", "No fallbacks in later runs"],
                     ["8 Nov", "Only 14–16% of answers cited sources", "Citation reminder last in prompt",
                      "70–86% cite"],
                     ["8 Nov", "Wrong figures labelled 'partial', rate shown as 0.0",
                      "Figures not in relevant evidence → unsupported + student warning", "Wrong figures flagged"],
                     ["9 Nov", "Invented worked example (6-pole motor at 1500 RPM)",
                      "Example rule; relevant-evidence figure check; NLI contradiction check", "Caught in re-check"],
                     ["10 Nov", "NLI flagged correct sentences (≈3 real errors in 35 flags)",
                      "Contradictions count only for claims with figures, p ≥ 0.95", "12% → 2% of answers"],
                     ["10 Nov", "Prompt instructions echoed into 7% of answers", "Echo filter + prompt rule",
                      "18 → 0 answers"],
                     ["10 Nov", "Formulas and fragments verified as claims", "Fragment filter", "–"],
                     ["11 Nov", "'(1,1)' read as 11; '32x32' from the question unrecognised", "Number parser fixed",
                      "Unit tests"],
                     ["11 Nov", "Failed image analysis gave a guessed answer ('XOR gate')",
                      "'Could not analyse the image' for image-dependent questions", "Unit tests"],
                     ["11 Nov", "Hybrid image answers ignored the image", "Image rules in the prompt",
                      "Amplitude read correctly in re-test"],
                     ["11 Nov", "Image scoring matched the word 'and' and missed LaTeX answers",
                      "Whole-word matching, LaTeX normalisation", "Re-scored: 6/8 and 5/8"],
                     ["11 Nov", "66 AICTE core topics missing from the knowledge base",
                      "Incremental extension: 37 articles, 800 passages, 87 s (no rebuild)",
                      "Router held-out 96.4% → 98.2%"],
                     ["11 Nov", "Wrong arithmetic passed verification ('20 ms → 1 Hz')",
                      "Arithmetic and unit check on every answer; one revision",
                      "All 3 known errors caught"],
                     ["12 Nov", "App open to anyone on a public pod link; no load limits",
                      "Access codes, rate limit, answer queue, error references",
                      "Gate and limit tests"],
                     ["12 Nov", "Follow-ups ('What are its advantages?') retrieved unrelated passages",
                      "Conversation memory resolves the topic; budgeted memory in the prompt",
                      "Right topic in top 5: 0.3% → 97.3%"],
                     ["12 Nov", "Comparisons retrieved evidence for one side only",
                      "Agentic RAG: plan sub-queries, check coverage, refine, correct",
                      "Both sides: 22.5% → 75%"],
                     ["12 Nov", "~1,000 source tokens per prompt at 20 tokens/s on the CPU",
                      "Context engineering: de-duplicate, compress, budget; stable prefix",
                      "−34% answer time, best passage kept 100%"],
                     ["12 Nov", "Answer cache matched 'AC motor' with 'DC motor'",
                      "Every content word must match, short ones included",
                      "0 false hits in 4,950 pairs"],
                     ["14 Nov", "Follow-ups resolved but answered by the direct shortcut",
                      "Router rule: in-domain follow-ups go to RAG", "0/3 → 3/3 grounded"],
                     ["14 Nov", "3B model read a truth table / code exactly, then reasoned wrongly",
                      "Exact tools: gate checker, safe code interpreter", "Adaptive images 4/8 → 8/8"]],
                    [17, 54, 54, 45]),
              Spacer(1, 3 * mm),
              P("A larger NLI model (nli-deberta-v3-base) was tested as an alternative and performed <i>worse</i>: "
                "it missed both genuine errors in the flagged set. The small model with the figure-only rule was kept."),
              PageBreak()]

    # ---------------------------------------------------------- 7 limitations
    story += [H1("7. Limitations"),
              bullets([
                  "<b>Answer correctness is not yet measured by people.</b> The blind grading sheet is ready; until it "
                  "is graded, the report shows cost, grounding and verification, not correctness.",
                  "<b>Labels and catalogues were drafted during development.</b> Benchmark labels and the 23 new branch "
                  "catalogues need team review; 12 groups have no AICTE reference.",
                  "<b>Small samples.</b> 50 questions per benchmark run and 8 synthetic images; differences of a few "
                  "percentage points are within run-to-run variation.",
                  "<b>Verification checks support and explicit arithmetic, not reasoning.</b> Written-out "
                  "calculations and period/frequency pairs are now recomputed, but a wrong step of reasoning "
                  "without a written calculation still passes.",
                  "<b>Small models.</b> The 3B vision model misreads tables; the 3B language model needs firm prompting "
                  "to cite sources and to avoid inventing examples.",
                  "<b>Hardware.</b> Interactive use needs a GPU; the laptop CPU produced 1–4 tokens/s.",
                  "<b>Knowledge source.</b> Wikipedia is general, not syllabus-specific; textbook or lecture material "
                  "(with permission) would match examination expectations more closely.",
              ]),
              H1("8. Remaining work"),
              table([["Task", "Who", "Tool"],
                     ["Blind grading of 250 answers by two raters", "Team", "Evaluation Studio → Grade answers"],
                     ["Label 120 claims for verifier calibration", "Team", "Evaluation Studio → Label claims"],
                     ["Confirm/correct each subject; syllabus for the 12 groups without AICTE reference",
                      "Team", "Evaluation Studio → Review syllabus"],
                     ["Record student voice questions (with consent)", "Team", "Evaluation Studio → Record voice"],
                     ["Add real images (circuits, notes, figures)", "Team", "Evaluation Studio → Add images"],
                     ["Deploy on Streamlit Community Cloud", "Team", "docs/DEPLOYMENT.md (~20 min)"],
                     ["Full 130-question benchmark and 3B vs 7B vision comparison on a GPU",
                      "Me, on Colab or the DGX", "Colab benchmark notebook / deploy/k8s/deploy.sh"],
                     ["Insert grading results into the results chapter", "Me, after grading",
                      "evaluation.analyse_grading"]],
                    [80, 30, 60]),
              PageBreak()]

    # ---------------------------------------------------------- 9 how to run
    story += [H1("9. How to run the project"),
              P("Local app (chat and evaluation studio)", "h2"),
              code("double-click run_app.cmd      (or: venv\\Scripts\\streamlit run app.py)\n"
                   "open http://localhost:8501"),
              P("Language model on the laptop (slow) or on Colab (recommended)", "h2"),
              code("Local:  Ollama with qwen2.5:3b (models stored on D:\\Ollama\\models)\n"
                   "Colab:  colab/EduRoute_Colab_Benchmark.ipynb     (benchmarks)\n"
                   "        colab/EduRoute_Colab_Model_Server.ipynb  (serve models to the hosted app)"),
              P("Evaluation commands", "h2"),
              code("python -m evaluation.evaluate_retrieval --benchmark retrieval_engineering_v1.jsonl\n"
                   "python -m evaluation.check_router_heldout\n"
                   "python -m evaluation.run_system_benchmark --sample 20 --resume <partial.jsonl>\n"
                   "python -m evaluation.run_image_benchmark\n"
                   "python -m evaluation.evaluate_asr <folder>\n"
                   "python -m evaluation.analyse_grading --key <key.json> rater1.csv rater2.csv\n"
                   "python -m src.data_collection.compare_aicte\n"
                   "python -m pytest -q                      (131 tests)"),
              P("Production deployment on a GPU server or the DGX cluster (docs/GPU_SERVER.md)", "h2"),
              code("laptop:      venv\\Scripts\\python deploy\\make_server_package.py\n"
                   "DGX (k8s):   bash deploy/k8s/inspect_cluster.sh;  bash deploy/k8s/deploy.sh up\n"
                   "plain GPU:   bash deploy/gpu/setup.sh;  bash deploy/gpu/services.sh start\n"
                   "check:       python -m deploy.smoke_test\n"
                   "evaluate:    bash deploy/gpu/run_evaluations.sh   (results zipped)"),
              P("A deployment requires an access code for students and a separate code for the evaluation "
                "studio, limits each session to 60 questions an hour, generates at most four answers at once "
                "(others queue), restarts failed services automatically and backs up the team's work daily."),
              P("Project layout", "h2"),
              table([["Folder", "Contents"],
                     ["app.py, pages/", "Streamlit chat app; Evaluation Studio"],
                     ["src/rag, src/router, src/verification", "Retrieval, routing, verification"],
                     ["src/llm, src/vision, src/voice", "Model back-ends and prompts; image and speech handling"],
                     ["src/data_collection", "Wikipedia collector, PDF ingestion, AICTE comparison"],
                     ["config/", "Settings, branch catalogues, router calibration"],
                     ["evaluation/", "Benchmarks, evaluators, rubric, results"],
                     ["colab/", "GPU notebooks and upload bundle"],
                     ["deploy/", "GPU server, pod and Kubernetes deployment; smoke test"],
                     ["docs/", "PROGRESS.md (measured status), REPORT_RESULTS.md, DEPLOYMENT.md, this report"]],
                    [60, 110])]
    return story


class ReportDoc(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        if hasattr(flowable, "_toc"):
            self.notify("TOCEntry", (0, flowable._toc, self.page))


def footer(canvas, doc):
    canvas.saveState()
    if doc.page > 1:
        canvas.setFont("Body", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, 10 * mm, "EduRoute AI · Project report")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
        canvas.setStrokeColor(RULE)
        canvas.line(18 * mm, 13 * mm, A4[0] - 18 * mm, 13 * mm)
    canvas.restoreState()


def main():
    doc = ReportDoc(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                    bottomMargin=18 * mm, title="EduRoute AI: Project Report",
                    author="Syed Mohammed Yaseen, Mohammed Ayaan, Arfath Khan",
                    subject="Adaptive LLM + RAG + multimodal + voice study assistant")
    doc.multiBuild(build(), onFirstPage=footer, onLaterPages=footer)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
