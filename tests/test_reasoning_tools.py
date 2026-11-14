"""Exact tools for image readings: truth tables and short code snippets (14 Nov fix)."""

import pytest

from src.vision.reasoning_tools import analyse_truth_table, check_image_reading, run_code_snippet


def table(rows, header=("A", "B", "Y")):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


@pytest.mark.parametrize("outputs,gate", [
    ((0, 0, 0, 1), "AND"), ((0, 1, 1, 1), "OR"), ((1, 1, 1, 0), "NAND"),
    ((1, 0, 0, 0), "NOR"), ((0, 1, 1, 0), "XOR"), ((1, 0, 0, 1), "XNOR"),
])
def test_two_input_gates(outputs, gate):
    rows = [(a, b, y) for (a, b), y in zip([(0, 0), (0, 1), (1, 0), (1, 1)], outputs)]
    assert f"matches the {gate} function" in analyse_truth_table(table(rows))


def test_not_gate_three_inputs_and_unknown_tables():
    assert "NOT" in analyse_truth_table(table([(0, 1), (1, 0)], header=("A", "Y")))
    three = [(a, b, c, a & b & c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    assert "AND" in analyse_truth_table(table(three, header=("A", "B", "C", "Y")))
    assert "do not match any single standard gate" in analyse_truth_table(
        table([(0, 0, 0), (0, 1, 0), (1, 0, 1), (1, 1, 1)]))                 # Y = A: no 2-input gate
    assert analyse_truth_table("| Pump | Flow |\n|---|---|\n| A | 120 |\n| B | 95 |") is None   # not 0/1


def test_the_benchmark_transcription_is_recognised():
    reading = ("2. VISIBLE:\n   - `total = 0`\n   - `for i in range(1, 6):`\n   - `    total += i`\n"
               "   - `print(total)`\n4. The variable `total` is updated in the loop.")
    assert "prints '15'" in run_code_snippet(reading)


@pytest.mark.parametrize("code", [
    "import os\nos.remove('x')", "open('secret.txt').read()", "x = [].__class__",
    "while True:\n    pass", "for i in range(10**9):\n    pass", "print(2 ** 100000)",
    "def f():\n    return 1\nprint(f())",
])
def test_unsafe_or_unbounded_code_is_never_run(code):
    assert run_code_snippet(f"```python\n{code}\n```") is None


def test_checked_facts_block():
    assert check_image_reading("A photo of a motor.") == ""
    block = check_image_reading(table([(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]))
    assert block.startswith("\n\nCHECKED FACTS") and "AND" in block
