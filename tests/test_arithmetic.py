"""Arithmetic and unit-consistency checks on answers."""

import pytest

from src.verification.arithmetic import check_answer, evaluate_side


@pytest.mark.parametrize("text", [
    "Using Ohm's law, I = 12 V / 30 Ω = 0.4 A.",
    "R_total = 10 Ω + 20 Ω = 30 Ω.",
    "I = 12 / 30 = 400 mA",                                      # unit prefixes converted
    r"\( R_{total} = 10 \, \Omega + 20 \, \Omega = 30 \, \Omega \)",
    r"\( I = \frac{12 \, V}{30 \, \Omega} = 0.4 \, A \)",
    "With a period of 20 ms, the frequency is 50 Hz.",
    "R1 = 10 Ω and V = I × R.",                                  # variables are never guessed
    "H = -(9/14 × log 9/14) = 0.940",                            # a function: not evaluated
    "p = 9/14 ≈ 0.643",                                          # rounding allowed
    "n = 120 × 50 / 4 = 1,500 RPM",                              # thousands separator
])
def test_correct_working_raises_nothing(text):
    assert check_answer(text) == []


@pytest.mark.parametrize("text,expected,stated", [
    ("The current is I = 12 V / 30 Ω = 0.5 A.", 0.4, 0.5),
    ("The graph shows a period of 20 ms and a frequency of 1 Hz.", 50, 1),   # 11 Nov image benchmark
    ("n = 120 × 50 / 6 = 1500 RPM", 1000, 1500),                             # the 6-pole motor example
])
def test_wrong_calculations_are_caught(text, expected, stated):
    issues = check_answer(text)
    assert len(issues) == 1
    assert issues[0].expected == pytest.approx(expected) and issues[0].stated == pytest.approx(stated)
    assert "works out to" in issues[0].describe() and not issues[0].describe().endswith(".")


def test_sides_with_variables_are_not_evaluated():
    assert evaluate_side("R_total") is None
    assert evaluate_side("x2 + 3") is None
    assert evaluate_side("12 V / 30 Ω") == pytest.approx(0.4)
    assert evaluate_side("400 mA") == pytest.approx(0.4)
