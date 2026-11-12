"""
Arithmetic and unit-consistency check for answers (no model, no sources).

The evidence-based verifier checks whether figures appear in the sources,
not whether a calculation is right: "period 20 ms, so a frequency of 1 Hz"
passed it (11 Nov image benchmark). This module recomputes what an answer
computes and flags what does not hold:

1. Equations written out: every side of "12 V / 30 Ω = 0.4 A" or
   "10 Ω + 20 Ω = 30 Ω" that is pure arithmetic (numbers, operators and known
   units) is evaluated, with unit prefixes converted (400 mA = 0.4 A), and
   adjacent sides must agree within rounding.
2. Period and frequency stated in the same sentence must satisfy f × T ≈ 1.

Sides containing a variable (R_total, x, R1) are skipped, never guessed, so
a variable name is not mistaken for a number. It runs on every answer and
costs microseconds.
"""

import ast
import operator
import re
from dataclasses import dataclass

PREFIX = {"G": 1e9, "M": 1e6, "k": 1e3, "m": 1e-3, "µ": 1e-6, "u": 1e-6, "n": 1e-9}
BASE_UNITS = {"V", "A", "Ω", "ohm", "ohms", "Hz", "s", "m", "g", "N", "J", "W", "Pa", "K", "F", "H", "C",
              "rpm", "RPM", "rad", "%", "°C", "°", "bar", "L", "B", "bit", "bits", "Wh", "VA", "VAR"}
WORD = re.compile(r"[A-Za-zΩµ°%]+")

_LATEX = [(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))"), (r"\\times|\\cdot|×|·", "*"), (r"÷", "/"),
          (r"−|–", "-"), (r"\\Omega", "Ω"), (r"\\mu", "µ"), (r"\\text\{([^{}]*)\}", r"\1"),
          (r"\\mathrm\{([^{}]*)\}", r"\1"), (r"\\[,;:! ]|\\quad", " "), (r"\\[\[\]()]|\$", " "),
          (r"(?<=\d),(?=\d{3}\b)", "")]


@dataclass
class ArithmeticIssue:
    statement: str
    expected: float
    stated: float

    def describe(self):
        return f"{self.statement.strip().rstrip('.,;:')} (works out to {_fmt(self.expected)}, not {_fmt(self.stated)})"


def _fmt(x):
    return f"{x:.4g}"


def _plain(text):
    for pattern, repl in _LATEX:
        text = re.sub(pattern, repl, text)
    return text


def _unit_scale(word):
    """Multiplier for a unit word, or None if the word is not a unit (a variable)."""
    if word in BASE_UNITS:
        return 1.0
    if len(word) > 1 and word[0] in PREFIX and word[1:] in BASE_UNITS:
        return PREFIX[word[0]]
    return None


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow}


def _safe_eval(expr):
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            right = ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 12:
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](ev(node.left), right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = ev(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        raise ValueError("not plain arithmetic")
    return ev(ast.parse(expr, mode="eval"))


def evaluate_side(side):
    """Value of one side of an equation, or None when it is not pure arithmetic."""
    side = side.strip().strip(".,;:")
    if not re.search(r"\d", side):
        return None
    out = []
    for m in re.finditer(r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|[A-Za-zΩµ°%]+|\^|[-+*/()]|\S", side):
        tok = m.group(0)
        if re.fullmatch(r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", tok):
            # a digit glued to letters ("R1", "x2") is part of a variable name
            if m.start() > 0 and side[m.start() - 1].isalpha():
                return None
            out.append(tok)
        elif WORD.fullmatch(tok):
            scale = _unit_scale(tok)
            if scale is None:
                return None                     # a variable or a word: not evaluable
            if out and re.fullmatch(r"[\d.eE+-]+", out[-1]) and scale != 1.0:
                out[-1] = f"({out[-1]}*{scale!r})"
        elif tok == "^":
            out.append("**")
        elif tok in "+-*/()":
            out.append(tok)
        else:
            return None
    expr = "".join(out)
    if not expr or not re.search(r"\d", expr):
        return None
    try:
        return _safe_eval(expr)
    except Exception:
        return None


def _agrees(a, b):
    """Equal within what the stated precision allows (rounding to 2-3 figures)."""
    if a == b:
        return True
    scale = max(abs(a), abs(b))
    return abs(a - b) <= max(0.02 * scale, 1e-9)


PERIOD = re.compile(r"\bperiod\b[^.\n]{0,60}?(\d+(?:\.\d+)?)\s*(ms|µs|us|ns|s)\b", re.IGNORECASE)
FREQUENCY = re.compile(r"\bfrequency\b[^.\n]{0,60}?(\d+(?:\.\d+)?)\s*(GHz|MHz|kHz|Hz)\b", re.IGNORECASE)
_TIME = {"s": 1.0, "ms": 1e-3, "µs": 1e-6, "us": 1e-6, "ns": 1e-9}
_FREQ = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}


def check_answer(answer):
    """All arithmetic problems found in an answer (empty list = nothing wrong found)."""
    issues = []
    text = _plain(answer or "")
    for line in re.split(r"\n|(?<=[.!?])\s+(?=[A-Z])", text):
        if "=" in line:
            sides = re.split(r"=|≈", line)
            values = [evaluate_side(s) for s in sides]
            for (s_left, v_left), (s_right, v_right) in zip(zip(sides, values), zip(sides[1:], values[1:])):
                if v_left is not None and v_right is not None and not _agrees(v_left, v_right):
                    issues.append(ArithmeticIssue(f"{s_left.strip()} = {s_right.strip()}", v_left, v_right))
        p, f = PERIOD.search(line), FREQUENCY.search(line)
        if p and f:
            period = float(p.group(1)) * _TIME[p.group(2).replace("µ", "µ")]
            freq = float(f.group(1)) * _FREQ[f.group(2).lower()]
            if period > 0 and not _agrees(1 / period, freq):
                issues.append(ArithmeticIssue(
                    f"frequency {f.group(1)} {f.group(2)} for a period of {p.group(1)} {p.group(2)}",
                    1 / period, freq))
    return issues
