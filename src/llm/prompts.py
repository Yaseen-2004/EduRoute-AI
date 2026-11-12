"""Prompt templates (report §9.2 grounded generation, §15 answer modes)."""

import re

SYSTEM_PROMPT = (
    "You are EduRoute AI, an academic tutor for B.Tech engineering students. "
    "Explain clearly and accurately at an undergraduate level. Define technical "
    "terms when you first use them and give a short example where it helps. "
    "If you are not sure about something, say so instead of guessing. "
    "Never invent references, URLs, statistics, quotations or numerical values. "
    "The instructions you are given are for you only: never repeat or quote them "
    "in your answer."
)

ANSWER_MODES = {
    "2-mark": {
        "label": "2-mark (short answer)",
        "max_new_tokens": 160,
        "instructions": (
            "Write a short examination answer of about 40-70 words: a direct "
            "definition or answer, followed by one or two key points. Use the "
            "essential terminology. No headings."
        ),
    },
    "5-mark": {
        "label": "5-mark (structured)",
        "max_new_tokens": 420,
        "instructions": (
            "Write a structured examination answer of about 150-250 words with: "
            "a one-line introduction; a structured explanation using short "
            "bullet points for the key concepts, features or steps; and one "
            "relevant example (see the example rule)."
        ),
    },
    "10-mark": {
        "label": "10-mark (detailed)",
        "max_new_tokens": 900,
        "instructions": (
            "Write a detailed examination answer of about 400-600 words using "
            "these markdown headings where relevant: Introduction; Explanation; "
            "Components / Stages; Examples / Applications; Advantages; "
            "Limitations; Conclusion. If a diagram would normally be drawn, "
            "describe it in words under 'Diagram description'."
        ),
    },
    "conversational": {
        "label": "Conversational",
        "max_new_tokens": 512,
        "instructions": "Answer conversationally and concisely, as a helpful tutor.",
    },
}

# Answer length is a presentation preference, not a target to pad (report §15).
NO_PADDING = (
    "Do not pad the answer with unsupported details just to reach the length; "
    "a shorter accurate answer is better."
)

GROUNDING_RULES = (
    "Use the numbered SOURCES below as your primary evidence.\n"
    "- Cite the sources that support each factual statement with bracketed "
    "numbers such as [1] or [2][3]. Only cite numbers that exist below.\n"
    "- If you add general background knowledge that is not in the sources, "
    "do not attach a citation to it.\n"
    "- If the sources do not contain enough information to answer, say "
    "clearly: \"The knowledge base does not contain enough information about "
    "this\" and answer only what the sources support.\n"
    "- Never invent source titles or URLs.\n"
    "- Example rule: take examples from the SOURCES. If they contain no suitable "
    "example, give a short qualitative example without new numbers. Do not make "
    "up numerical worked examples; use only figures that appear in the SOURCES, "
    "and for a calculation show each step so it can be checked."
)


# Qwen2.5-3B cited sources in only 14-16% of RAG answers when the citation
# rule came before the (long) answer-format instructions (Colab benchmark,
# 8 Nov 2026). Small models follow the last instruction best, so it is repeated
# at the very end of the prompt.
CITATION_REMINDER = (
    "Important: end every sentence that uses information from the SOURCES with "
    "its source number in square brackets, for example [1] or [2][3]."
)


def _instruction_sentences():
    texts = [GROUNDING_RULES, NO_PADDING, CITATION_REMINDER, SYSTEM_PROMPT, IMAGE_RULES]
    texts += [m["instructions"] for m in ANSWER_MODES.values()]
    sentences = set()
    for text in texts:
        for part in re.split(r"(?<=[.:;])\s+|\n", text):
            words = re.findall(r"[a-z0-9]+", part.lower())
            if len(words) >= 6:
                sentences.add(" ".join(words))
    return sentences


_ECHO_PREFIX = re.compile(r"^\s*(answer format:|sources:|question:|draft answer:)", re.IGNORECASE)


def strip_prompt_echo(answer):
    """Remove lines that repeat the prompt's own instructions.

    On the 10 Nov Colab run, 4 of 250 answers (mostly out-of-domain questions
    forced through retrieval) echoed lines such as "Answer format: ..." and
    "Do not invent source titles or URLs." into the answer.
    """
    if not answer:
        return answer
    rules = _INSTRUCTION_SENTENCES
    kept = []
    for line in answer.splitlines():
        words = " ".join(re.findall(r"[a-z0-9]+", line.lower()))
        if _ECHO_PREFIX.match(line) or (words and any(rule in words for rule in rules)):
            continue
        kept.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def format_sources(passages, max_chars=1200):
    blocks = []
    for number, passage in enumerate(passages, start=1):
        title = passage.get("article_title", "Unknown source")
        section = passage.get("section")
        heading = f"{title} - {section}" if section and section != "Introduction" else title
        text = passage.get("prompt_text") or passage["text"]     # compressed by context engineering
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + " ..."
        blocks.append(f"[{number}] {heading}\n{text}")
    return "\n\n".join(blocks)


def _history_block(history):
    if not history:
        return ""
    if isinstance(history, str):          # already rendered by conversation memory
        return history
    lines = []
    for turn in history[-3:]:
        lines.append(f"Student: {turn['question']}")
        lines.append(f"Tutor: {turn['answer'][:600]}")
    return "Earlier in this conversation:\n" + "\n".join(lines) + "\n\n"


def _context_block(subject_context):
    if not subject_context:
        return ""
    parts = [f"{k}: {v}" for k, v in subject_context.items() if v]
    return ("Student context - " + "; ".join(parts) + "\n\n") if parts else ""


def build_direct_messages(question, answer_mode="5-mark", history=None, subject_context=None):
    mode = ANSWER_MODES[answer_mode]
    user = (
        f"{_context_block(subject_context)}{_history_block(history)}"
        f"Question: {question}\n\n"
        f"Answer format: {mode['instructions']} {NO_PADDING}"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


# Hybrid image + knowledge-base answers. Without this, the grounding rules
# (which talk only about SOURCES) made the model answer "the knowledge base does
# not contain enough information" for questions whose answer was in the image
# (11 Nov Colab image benchmark: sine-wave amplitude, truth table).
IMAGE_RULES = (
    "The IMAGE ANALYSIS describes the student's own image. Take facts about the image "
    "(values, labels, table rows, connections, structure) from the IMAGE ANALYSIS and "
    "answer them directly, without a citation. Use the SOURCES only for background "
    "concepts, and cite them for those. Never say the knowledge base lacks information "
    "when the answer is in the IMAGE ANALYSIS; if the image analysis is incomplete, give "
    "what it shows and say what could not be read."
)


def build_image_messages(question, visual_description, answer_mode="5-mark", history=None,
                         subject_context=None):
    """Answer an image question with the TEXT model from the vision model's reading.

    "Read, then reason": the 3B vision model transcribed the truth-table image
    exactly but, answering in one step, called it XOR / NOT; the text model
    reasoning over the transcription gets it right (12 Nov CPU tests).
    """
    mode = ANSWER_MODES[answer_mode]
    user = (
        f"{_context_block(subject_context)}{_history_block(history)}"
        f"IMAGE ANALYSIS (read from the student's image by a vision model):\n{visual_description}\n\n"
        f"Question about the image: {question or 'Explain this image.'}\n\n"
        "Work from the IMAGE ANALYSIS: for a table, check each row before concluding; for a "
        "graph, use the values read off it. Facts about the image need no citation. If the "
        "analysis marks something uncertain or incomplete, say what could not be read.\n\n"
        f"Answer format: {mode['instructions']} {NO_PADDING}"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def build_rag_messages(question, passages, answer_mode="5-mark", history=None,
                       subject_context=None, visual_description=None, stable_prefix=False):
    """`stable_prefix`: put the fixed grounding rules in the system message, so every
    RAG prompt starts with the same text and the server reuses its cached reading
    (Ollama/llama.cpp prefix cache) instead of re-reading ~250 tokens per question.
    The citation reminder stays last: the 3B model follows the final instruction best."""
    mode = ANSWER_MODES[answer_mode]
    visual = ""
    if visual_description:
        visual = (
            "IMAGE ANALYSIS (from a vision model; may contain recognition errors - "
            "treat items marked uncertain as uncertain):\n"
            f"{visual_description}\n\n"
        )
    system = f"{SYSTEM_PROMPT}\n\n{GROUNDING_RULES}" if stable_prefix else SYSTEM_PROMPT
    user = (
        f"{_context_block(subject_context)}{_history_block(history)}"
        f"{visual}"
        f"SOURCES:\n{format_sources(passages)}\n\n"
        f"{'' if stable_prefix else GROUNDING_RULES + chr(10) * 2}"
        f"{IMAGE_RULES + chr(10) * 2 if visual_description else ''}"
        f"Question: {question}\n\n"
        f"Answer format: {mode['instructions']} {NO_PADDING}\n\n"
        f"{CITATION_REMINDER}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


VISION_EXTRACTION_PROMPT = (
    "You are analysing an image uploaded by an engineering student.\n"
    "Student question: {question}\n\n"
    "Describe what the image contains in a structured way:\n"
    "1. Type of image (circuit, block diagram, graph, equation, code, flowchart, photo, ...).\n"
    "2. VISIBLE: labels, text, symbols, values, axes and connections you can read clearly.\n"
    "3. UNCERTAIN: anything you cannot read reliably - mark each item '(uncertain)'.\n"
    "4. Relationships/structure (what connects to what, trends in a graph, steps in a flow).\n"
    "TABLES (truth tables, data tables): copy EVERY row, cell by cell, as a Markdown table "
    "with its header. Never summarise a table as 'values 0 and 1'.\n"
    "GRAPHS: read the axis labels, units and the tick values at the peaks, crossings and "
    "periods.\n"
    "Do not answer the student's question yet and do not guess unreadable values."
)

VISION_ANSWER_PROMPT = (
    "Student question about the attached image: {question}\n\n"
    "Answer format: {instructions} {no_padding}\n"
    "If the image contains a table (for example a truth table), first copy EVERY row, cell "
    "by cell, as a Markdown table, then answer using those rows only. For a graph, first "
    "state the axis labels, units and the values you read.\n"
    "Distinguish clearly between what is visible in the image and what you are "
    "inferring. If part of the image is unreadable, say so."
)


def build_revision_messages(question, draft_answer, passages, unsupported_claims, answer_mode):
    mode = ANSWER_MODES[answer_mode]
    claims = "\n".join(f"- {c}" for c in unsupported_claims)
    user = (
        f"SOURCES:\n{format_sources(passages)}\n\n"
        f"Question: {question}\n\n"
        f"DRAFT ANSWER:\n{draft_answer}\n\n"
        "A verification step found that these statements in the draft are not "
        f"supported by the sources:\n{claims}\n\n"
        "Rewrite the answer. For each listed statement either (a) correct it using "
        "the sources, (b) clearly qualify it as general background knowledge without "
        "a citation, or (c) remove it if it may be wrong. A statement with a figure "
        "that is not in the sources must be corrected from the sources or removed; "
        "never keep the figure. Do not add sentences about what the knowledge base "
        "lacks for content you removed; simply leave it out. Keep supported content "
        f"and its citations. {GROUNDING_RULES}\n\nAnswer format: {mode['instructions']}\n\n"
        f"{CITATION_REMINDER}"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def build_spoken_summary_messages(answer):
    user = (
        "Rewrite the following answer as a short spoken explanation of at most 90 "
        "words for text-to-speech. Plain sentences only: no markdown, bullet points, "
        "citation numbers, URLs or symbols; write formulas in words.\n\n"
        f"{answer}"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


_INSTRUCTION_SENTENCES = _instruction_sentences()
