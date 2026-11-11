"""
Workflow / process diagrams for answers (report §15: 10-mark answers normally
include a diagram).

A diagram is drawn when the student asks for one ("draw", "flowchart",
"workflow", "block diagram"...) or for a 10-mark process/explanation answer.
The finished answer is turned into a Graphviz DOT flowchart by one extra model
call, so the diagram always matches the answer the student reads. Without a
language model, a simple top-to-bottom flowchart is built from the answer's
numbered or bulleted steps. The app renders DOT with st.graphviz_chart.
"""

import re

DIAGRAM_REQUEST = re.compile(
    r"\b(draw|diagram|flow ?chart|work ?flow|block diagram|schematic|sketch|"
    r"illustrate|visuali[sz]e|show (the )?(steps|process|stages|flow))\b", re.IGNORECASE)
MAX_NODES = 14

DIAGRAM_PROMPT = (
    "Turn the answer below into a clear flowchart in Graphviz DOT.\n"
    "Rules:\n"
    "- Output only a ```dot code block containing one digraph.\n"
    "- Use rankdir=TB, node [shape=box, style=rounded]; use shape=diamond only for decisions.\n"
    f"- At most {MAX_NODES} nodes; node labels of at most 6 words, taken from the answer.\n"
    "- Show the order of steps, stages or components exactly as the answer describes them; "
    "do not add anything that is not in the answer.\n"
    "- The diagram must be ONE connected flow: every node is linked, each step has an arrow "
    "to the next step (A -> B -> C ...), with no separate disconnected pieces.\n\n"
    "Question: {question}\n\nAnswer:\n{answer}"
)


def wants_diagram(question, answer_mode="5-mark", intent=None):
    if DIAGRAM_REQUEST.search(question or ""):
        return True
    return answer_mode == "10-mark" and intent in ("procedure", "explanation")


def build_diagram_messages(question, answer):
    from src.llm.prompts import SYSTEM_PROMPT
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": DIAGRAM_PROMPT.replace("{question}", question)
             .replace("{answer}", answer[:4000])}]


def extract_dot(text):
    """The digraph from a model reply, or None if it is missing or unusable."""
    if not text:
        return None
    block = re.search(r"```(?:dot|graphviz)?\s*(digraph[\s\S]*?\})\s*```", text, re.IGNORECASE)
    dot = block.group(1) if block else None
    if dot is None:
        bare = re.search(r"(digraph\s*\w*\s*\{[\s\S]*\})", text)
        dot = bare.group(1) if bare else None
    if not dot or dot.count("{") != dot.count("}") or "->" not in dot:
        return None
    if re.search(r"\b(shell|exec|import)\b|<script", dot, re.IGNORECASE):   # nothing but a graph
        return None
    nodes = set(re.findall(r'(?:^|[;{\s])("?[\w ]+"?)\s*(?:\[|->)', dot))
    if len(nodes) > MAX_NODES * 2:
        return None
    return dot.strip()


_EDGE = re.compile(r'("[^"]+"|[\w]+)\s*->\s*("[^"]+"|[\w]+)')


def is_connected(dot):
    """True when all nodes that appear in edges form one connected flow."""
    edges = [(a.strip('"'), b.strip('"')) for a, b in _EDGE.findall(dot or "")]
    if not edges:
        return False
    nodes = {n for e in edges for n in e}
    neighbours = {n: set() for n in nodes}
    for a, b in edges:
        neighbours[a].add(b)
        neighbours[b].add(a)
    seen, stack = set(), [next(iter(nodes))]
    while stack:
        n = stack.pop()
        if n not in seen:
            seen.add(n)
            stack.extend(neighbours[n] - seen)
    return seen == nodes


_STEP = re.compile(r"^\s*(?:\d+[.)]|[-*•]|step\s*\d+[:.)]?)\s+(.*)$", re.IGNORECASE)


def _label(text, words=6):
    text = re.sub(r"\[\d+\]|\*\*|__|`", "", text).strip()
    text = re.split(r"[:.;]\s", text, maxsplit=1)[0]          # the step's heading part
    clipped = " ".join(text.split()[:words]).rstrip(".,;: ")
    return clipped.replace('"', "'") + ("…" if len(text.split()) > words else "")


def heuristic_dot(answer, title=None):
    """A linear flowchart from the answer's numbered/bulleted steps (no model needed)."""
    steps = []
    for line in (answer or "").splitlines():
        m = _STEP.match(line)
        if m and len(m.group(1).split()) >= 2:
            steps.append(_label(m.group(1)))
    steps = list(dict.fromkeys(s for s in steps if s))[:MAX_NODES]
    if len(steps) < 2:
        return None
    lines = ["digraph G {", "  rankdir=TB;", '  node [shape=box, style="rounded,filled", fillcolor="#eef2ff"];']
    if title:
        lines.append(f'  start [label="{_label(title, 8)}", shape=ellipse, fillcolor="#e0f2fe"];')
    for i, step in enumerate(steps):
        lines.append(f'  s{i} [label="{step}"];')
    order = (["start"] if title else []) + [f"s{i}" for i in range(len(steps))]
    lines += [f"  {a} -> {b};" for a, b in zip(order, order[1:])]
    lines.append("}")
    return "\n".join(lines)
