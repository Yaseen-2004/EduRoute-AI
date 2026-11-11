"""Workflow diagrams: when they are drawn, and that only safe, well-formed DOT is shown."""

from src.llm.diagrams import extract_dot, heuristic_dot, wants_diagram


def test_when_a_diagram_is_drawn():
    assert wants_diagram("Draw the workflow of the TCP three-way handshake")
    assert wants_diagram("Explain the steps with a flowchart", "5-mark")
    assert wants_diagram("Describe the working of a blast furnace", "10-mark", "explanation")
    assert not wants_diagram("What is Ohm's law?", "2-mark", "definition")
    assert not wants_diagram("Define entropy", "10-mark", "definition")


def test_extract_dot_from_a_model_reply():
    reply = ("Here is the flowchart:\n```dot\ndigraph G {\n rankdir=TB;\n a [label=\"SYN\"];\n"
             " b [label=\"SYN-ACK\"];\n a -> b;\n}\n```")
    dot = extract_dot(reply)
    assert dot.startswith("digraph") and "a -> b" in dot
    assert extract_dot("no diagram here") is None
    assert extract_dot("```dot\ndigraph G { a -> b;\n```") is None          # unbalanced braces
    assert extract_dot("```dot\ndigraph G { a [label=\"<script>\"]; a -> b; }\n```") is None


def test_heuristic_flowchart_from_numbered_steps():
    answer = ("The k-means algorithm:\n1. Choose k initial centroids at random.\n"
              "2. Assign every point to its nearest centroid.\n3. Recompute each centroid as the mean.\n"
              "4. Repeat until assignments stop changing.")
    dot = heuristic_dot(answer, title="Steps of k-means")
    assert dot.count("->") == 4 and 'label="Choose k initial centroids at random"' in dot
    assert heuristic_dot("A single sentence without steps.") is None


def test_disconnected_diagrams_are_detected():
    from src.llm.diagrams import is_connected
    chained = 'digraph G { "SYN" -> "SYN-ACK"; "SYN-ACK" -> "ACK"; }'
    pieces = 'digraph G { "Client sends SYN" -> "Server receives SYN"; "Client sends ACK" -> "Server receives ACK"; }'
    assert is_connected(chained) and not is_connected(pieces)
