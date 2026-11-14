"""
Exact reasoning tools for what the vision model reads off an image.

The 3B vision model transcribes images well but the 3B models reason over
the transcription unreliably (13-14 Nov CPU image benchmark):
- a truth table read exactly as 00->0, 01->0, 10->0, 11->1 was called a
  NOT gate (it is AND);
- `total = 0; for i in range(1, 6): total += i; print(total)` read exactly
  was said to print 21 (it prints 15).

Like the arithmetic checker, these tools compute the answer instead of
trusting the model, and their result is handed to the model as a checked fact:

  analyse_truth_table(text)  which standard gate the rows of a 0/1 table match
  run_code_snippet(text)     what a short Python snippet prints, run by a tiny
                             interpreter that only knows numbers, strings,
                             variables, arithmetic, comparisons, if/for/while,
                             range/len/sum/min/max/abs/print; no imports,
                             attributes, functions, files or anything else;
                             at most 10,000 steps.
"""

import ast
import operator
import re

GATES = {
    "AND": lambda a, b: a & b, "OR": lambda a, b: a | b, "NAND": lambda a, b: 1 - (a & b),
    "NOR": lambda a, b: 1 - (a | b), "XOR": lambda a, b: a ^ b, "XNOR": lambda a, b: 1 - (a ^ b),
}
OUTPUT_NAMES = {"y", "q", "f", "out", "output", "z"}
_BASE = {"NAND": "AND", "NOR": "OR", "XNOR": "XOR"}


def _gate(name, inputs):
    """A gate over any number of inputs: AND/OR/XOR fold; NAND/NOR/XNOR negate them."""
    base = _BASE.get(name, name)
    acc = inputs[0]
    for v in inputs[1:]:
        acc = GATES[base](acc, v)
    return 1 - acc if name in _BASE else acc
ROW = re.compile(r"^\s*\|(.+)\|\s*$")


def _tables(text):
    """Markdown tables in the text: lists of rows (lists of cell strings)."""
    tables, current = [], []
    for line in (text or "").splitlines():
        m = ROW.match(line)
        if m and not set(m.group(1).replace("|", "").strip()) <= set("-: "):
            current.append([c.strip() for c in m.group(1).split("|")])
        elif not m and current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    return tables


def analyse_truth_table(text):
    """'The rows match the AND function exactly.' for a 0/1 table, else None."""
    for table in _tables(text):
        header, rows = table[0], table[1:]
        if len(rows) < 2 or any(len(r) != len(header) for r in rows):
            continue
        if not all(cell in ("0", "1") for r in rows for cell in r):
            continue
        out = next((i for i, h in enumerate(header) if h.lower() in OUTPUT_NAMES), len(header) - 1)
        inputs = [i for i in range(len(header)) if i != out]
        names = [header[i] for i in inputs]
        values = [[int(r[i]) for i in inputs] for r in rows]
        y = [int(r[out]) for r in rows]
        matches = []
        if len(inputs) == 1:
            if all(o == 1 - v[0] for v, o in zip(values, y)):
                matches.append("NOT")
            if all(o == v[0] for v, o in zip(values, y)):
                matches.append("BUFFER (identity)")
        elif len(inputs) >= 2:
            for name in GATES:
                if [_gate(name, v) for v in values] == y:
                    matches.append(name)
        complete = len({tuple(v) for v in values}) == 2 ** len(inputs)
        if matches:
            return (f"Checked by computing the table: with inputs {', '.join(names)} and output "
                    f"{header[out]}, every row matches the {matches[0]} function"
                    + (" (all input combinations are present)." if complete else
                       " for the rows shown (not every input combination is listed)."))
        return (f"Checked by computing the table: the rows do not match any single standard gate "
                f"(AND, OR, NAND, NOR, XOR, XNOR, NOT) for inputs {', '.join(names)}.")
    return None


# ---------------------------------------------------------------- code

class _Stop(Exception):
    pass


_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_CMP = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
        ast.Gt: operator.gt, ast.GtE: operator.ge}
_FUNCS = {"range": range, "len": len, "sum": sum, "min": min, "max": max, "abs": abs,
          "int": int, "float": float, "str": str}
MAX_STEPS = 10_000


class _Interpreter:

    def __init__(self):
        self.vars, self.out, self.steps = {}, [], 0

    def tick(self):
        self.steps += 1
        if self.steps > MAX_STEPS:
            raise _Stop("too many steps")

    def expr(self, node):
        self.tick()
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str, bool)):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in self.vars:
                raise _Stop(f"unknown name {node.id}")
            return self.vars[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            left, right = self.expr(node.left), self.expr(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 64:
                raise _Stop("exponent too large")
            return _BIN[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd, ast.Not)):
            value = self.expr(node.operand)
            return -value if isinstance(node.op, ast.USub) else (not value if isinstance(node.op, ast.Not) else value)
        if isinstance(node, ast.Compare) and all(type(o) in _CMP for o in node.ops):
            left = self.expr(node.left)
            for op, comp in zip(node.ops, node.comparators):
                right = self.expr(comp)
                if not _CMP[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.BoolOp):
            values = [self.expr(v) for v in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self.expr(e) for e in node.elts]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            args = [self.expr(a) for a in node.args]
            if node.func.id == "print":
                self.out.append(" ".join(str(a) for a in args))
                return None
            if node.func.id in _FUNCS:
                result = _FUNCS[node.func.id](*args)
                if isinstance(result, range) and len(result) > MAX_STEPS:
                    raise _Stop("range too large")
                return list(result) if isinstance(result, range) else result
        raise _Stop(f"unsupported expression {type(node).__name__}")

    def run(self, statements):
        for node in statements:
            self.tick()
            if isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets):
                value = self.expr(node.value)
                for target in node.targets:
                    self.vars[target.id] = value
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and type(node.op) in _BIN:
                self.vars[node.target.id] = _BIN[type(node.op)](self.expr(node.target), self.expr(node.value))
            elif isinstance(node, ast.Expr):
                self.expr(node.value)
            elif isinstance(node, ast.If):
                self.run(node.body if self.expr(node.test) else node.orelse)
            elif isinstance(node, ast.For) and isinstance(node.target, ast.Name) and not node.orelse:
                for item in self.expr(node.iter):
                    self.vars[node.target.id] = item
                    self.run(node.body)
            elif isinstance(node, ast.While) and not node.orelse:
                while self.expr(node.test):
                    self.run(node.body)
            elif isinstance(node, ast.Pass):
                continue
            else:
                raise _Stop(f"unsupported statement {type(node).__name__}")


def _code_candidates(text):
    """Code blocks in fences, or the lines that look like Python statements."""
    fenced = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "", re.DOTALL)
    if fenced:
        return fenced
    # Vision transcriptions list code as bullets: "- `total = 0`". Only lines that are
    # entirely one code span count; prose mentioning `total` is left out.
    lines = []
    for line in (text or "").splitlines():
        m = re.match(r"^\s*(?:[-*]|\d+\.)?\s*`([^`]+)`\s*$", line)
        if m:
            lines.append(m.group(1).rstrip())
    return ["\n".join(lines)] if lines else []


def run_code_snippet(text):
    """'Running this code prints: 15' when a short, simple snippet can be run safely, else None."""
    for code in _code_candidates(text):
        lines = [l for l in code.splitlines() if l.strip()]
        if not lines or len(lines) > 40:
            continue
        indent = min(len(l) - len(l.lstrip()) for l in lines)
        source = "\n".join(l[indent:] for l in lines)
        try:
            tree = ast.parse(source)
            interpreter = _Interpreter()
            interpreter.run(tree.body)
        except (_Stop, SyntaxError, TypeError, ValueError, ZeroDivisionError, OverflowError,
                IndexError, KeyError, RecursionError):
            continue
        if interpreter.out:
            printed = "\n".join(interpreter.out)
            return f"Checked by running the code in a safe interpreter: it prints {printed!r}."
    return None


def check_image_reading(visual_description):
    """Checked facts to append to the vision model's reading (empty string if none)."""
    facts = [f for f in (analyse_truth_table(visual_description), run_code_snippet(visual_description)) if f]
    return ("\n\nCHECKED FACTS (computed exactly; trust these over your own reasoning):\n- "
            + "\n- ".join(facts)) if facts else ""
