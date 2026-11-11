"""
Minimal wikitext -> plain text conversion for knowledge-base building.

Why: TextExtracts returns full text for only one page per request, and the
Wikimedia API rate-limits this client (HTTP 429). Fetching raw wikitext for
20 pages per request and converting it locally cuts article requests ~20x.

Keeps:   section headings (== H ==), paragraphs, list items, link text,
         formulas from <math> and {{math}}/{{mvar}} templates.
Removes: references, comments, other templates (infoboxes, citations,
         navboxes), tables, files/images, categories, HTML tags, bold/italic marks.
"""

import html
import re

_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_REF_SELF = re.compile(r"<ref[^>/]*/>", re.IGNORECASE)
_REF = re.compile(r"<ref[^>]*>.*?</ref>", re.IGNORECASE | re.DOTALL)
_DROP_TAGS = re.compile(r"<(gallery|timeline|score|syntaxhighlight|source|references)[^>]*>.*?</\1>",
                        re.IGNORECASE | re.DOTALL)
_MATH = re.compile(r"<math[^>]*>(.*?)</math>", re.IGNORECASE | re.DOTALL)
_INLINE_TEMPLATES = re.compile(r"\{\{\s*(?:math|mvar|tmath|nowrap|var|sub|sup|ill|lang)\s*\|([^{}]*)\}\}",
                               re.IGNORECASE)
_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_TABLE = re.compile(r"\{\|.*?\|\}", re.DOTALL)
_FILE_LINK_START = re.compile(r"\[\[(?:File|Image|Category|Media):", re.IGNORECASE)
_LINK = re.compile(r"\[\[([^\[\]|]*\|)?([^\[\]]*)\]\]")
_EXT_LINK = re.compile(r"\[https?://[^\s\]]+\s*([^\]]*)\]")
_TAG = re.compile(r"</?[a-zA-Z][^>]*>")


def _remove_balanced(text, start_pattern, open_token="[[", close_token="]]"):
    """Remove constructs like [[File:...[[nested]]...]] that regex cannot balance."""
    out, i = [], 0
    while True:
        match = start_pattern.search(text, i)
        if not match:
            out.append(text[i:])
            break
        out.append(text[i:match.start()])
        depth, j = 0, match.start()
        while j < len(text):
            if text.startswith(open_token, j):
                depth += 1
                j += 2
            elif text.startswith(close_token, j):
                depth -= 1
                j += 2
                if depth == 0:
                    break
            else:
                j += 1
        i = j
    return "".join(out)


_FORMULA_WRAPPER_START = re.compile(r"\{\{\s*(?:NumBlk|EquationRef|Equation box 1|chem2|chem|ce)\s*\|+",
                                    re.IGNORECASE)
_FORMULA_WRAPPER_END = re.compile(r"\|*\s*(?:\{\{EquationRef\|[^{}]*\}\})?\s*\|*\}\}\s*$", re.MULTILINE)


def tidy_formula_wrappers(text):
    """
    Remove leftover wrapper templates around formulas whose LaTeX braces
    prevented normal template removal ("{{NumBlk|| x^2 |}}" -> "x^2").
    Safe to run on already-converted text.
    """
    lines = []
    for line in text.split("\n"):
        if _FORMULA_WRAPPER_START.search(line):
            line = _FORMULA_WRAPPER_START.sub("", line)
            line = _FORMULA_WRAPPER_END.sub("", line)
        lines.append(line.strip() if line.strip() else line)
    return "\n".join(lines)


def wikitext_to_text(wikitext):
    text = _COMMENT.sub("", wikitext)
    text = _REF_SELF.sub("", text)
    text = _REF.sub("", text)
    text = _DROP_TAGS.sub("", text)
    text = _MATH.sub(lambda m: " " + m.group(1).strip() + " ", text)
    # Keep the visible content of inline formatting templates, then drop the rest
    # innermost-first so nested templates disappear completely.
    for _ in range(10):
        new = _INLINE_TEMPLATES.sub(lambda m: m.group(1).split("|")[-1], text)
        new = _TEMPLATE.sub("", new)
        if new == text:
            break
        text = new
    text = _TABLE.sub("", text)
    text = _remove_balanced(text, _FILE_LINK_START)
    text = _LINK.sub(lambda m: m.group(2), text)
    text = _EXT_LINK.sub(lambda m: m.group(1), text)
    text = _TAG.sub("", text)
    text = re.sub(r"'{2,}", "", text)
    text = html.unescape(text)

    lines = []
    for line in text.splitlines():
        line = line.strip()
        if re.match(r"^(={2,6}).*\1$", line):
            lines.append("\n" + line)
            continue
        line = re.sub(r"^[*#:;]+\s*", "", line)
        if line in ("|}", "{|") or line.startswith(("|", "!", "__")):
            continue
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\(\s*[,;]?\s*\)", "", text)          # "( )" left by removed templates
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return tidy_formula_wrappers(text).strip()
