"""
Synthetic starter set for image questions (report §12, §22.2).

Draws 8 simple engineering images whose correct answers are known exactly
(truth table, block diagram, signal plot, bar chart, equation, resistor
circuit, code, flowchart) into evaluation/image_eval/, with questions.csv.
It tests the image pipeline end to end; it is NOT a substitute for real
photos of circuits, handwritten notes and textbook figures, which the team
should add to the same folder (same CSV format).

    venv\\Scripts\\python -m evaluation.make_image_starter_set

questions.csv columns:
    file, question, answer_mode, expected, kind, source
`expected`: required facts separated by ';' with alternatives separated by
'|', e.g. "and gate|and;1" means the answer must mention "AND gate" (or
"and") and the figure 1.
"""

import csv
import math

from PIL import Image, ImageDraw, ImageFont

from config import settings

OUT = settings.BASE_DIR / "evaluation" / "image_eval"


def font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def canvas(w=900, h=600):
    img = Image.new("RGB", (w, h), "white")
    return img, ImageDraw.Draw(img)


def box(d, xy, text, f):
    d.rectangle(xy, outline="black", width=3)
    x0, y0, x1, y1 = xy
    tw = d.textlength(text, font=f)
    d.text(((x0 + x1 - tw) / 2, (y0 + y1) / 2 - f.size / 2), text, fill="black", font=f)


def arrow(d, a, b):
    d.line([a, b], fill="black", width=3)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    for s in (-0.45, 0.45):
        d.line([b, (b[0] - 16 * math.cos(ang + s), b[1] - 16 * math.sin(ang + s))], fill="black", width=3)


def truth_table():
    img, d = canvas(700, 520)
    f, fb = font(34), font(30)
    d.text((40, 20), "Truth table", fill="black", font=fb)
    rows = [("A", "B", "Y"), ("0", "0", "0"), ("0", "1", "0"), ("1", "0", "0"), ("1", "1", "1")]
    for r, row in enumerate(rows):
        for c, v in enumerate(row):
            x, y = 120 + c * 160, 90 + r * 80
            d.rectangle([x, y, x + 160, y + 80], outline="black", width=2)
            d.text((x + 65, y + 22), v, fill="black", font=f)
    return img


def block_diagram():
    img, d = canvas(1200, 360)
    f = font(28)
    labels = ["Source", "Encoder", "Channel", "Decoder", "Destination"]
    for i, t in enumerate(labels):
        x = 30 + i * 235
        box(d, (x, 130, x + 190, 230), t, f)
        if i:
            arrow(d, (x - 45, 180), (x - 2, 180))
    d.text((480, 270), "Noise", fill="black", font=f)
    arrow(d, (520, 265), (520, 233))
    return img


def sine_plot():
    img, d = canvas(1000, 600)
    f = font(24)
    x0, y0, x1, y1 = 110, 60, 950, 520
    mid = (y0 + y1) / 2
    d.line([(x0, y0), (x0, y1)], fill="black", width=2)
    d.line([(x0, mid), (x1, mid)], fill="black", width=2)
    span_ms = 40.0                       # two periods of a 20 ms (50 Hz) signal
    amp_px = (y1 - y0) / 2 - 20          # 5 V
    pts = [(x0 + (x1 - x0) * t / 400, mid - amp_px * math.sin(2 * math.pi * (span_ms * t / 400) / 20))
           for t in range(401)]
    d.line(pts, fill="blue", width=3)
    for ms in range(0, 41, 10):
        x = x0 + (x1 - x0) * ms / span_ms
        d.line([(x, mid - 6), (x, mid + 6)], fill="black", width=2)
        d.text((x - 12, mid + 12), str(ms), fill="black", font=f)
    for v in (5, -5):
        y = mid - amp_px * v / 5
        d.line([(x0 - 6, y), (x0 + 6, y)], fill="black", width=2)
        d.text((x0 - 60, y - 12), f"{v} V", fill="black", font=f)
    d.text((x1 - 140, mid + 45), "Time (ms)", fill="black", font=f)
    d.text((x0 + 10, y0 - 40), "Voltage (V)", fill="black", font=f)
    return img


def bar_chart():
    img, d = canvas(900, 600)
    f = font(26)
    data = [("Pump A", 30), ("Pump B", 50), ("Pump C", 20)]
    base = 520
    d.line([(100, 60), (100, base)], fill="black", width=2)
    d.line([(100, base), (850, base)], fill="black", width=2)
    for i, (name, v) in enumerate(data):
        x = 170 + i * 230
        top = base - v * 8
        d.rectangle([x, top, x + 140, base], fill="#4a7ab5", outline="black")
        d.text((x + 50, top - 34), str(v), fill="black", font=f)
        d.text((x + 15, base + 12), name, fill="black", font=f)
    d.text((110, 20), "Flow rate (litres per minute)", fill="black", font=f)
    return img


def equation():
    img, d = canvas(900, 380)
    d.text((60, 60), "Ohm's law:  V = I × R", fill="black", font=font(48))
    d.text((60, 170), "I = 2 A,   R = 5 Ω", fill="black", font=font(44))
    d.text((60, 270), "V = ?", fill="black", font=font(44))
    return img


def series_circuit():
    img, d = canvas(1000, 560)
    f = font(28)
    d.rectangle([150, 120, 850, 440], outline="black", width=3)        # wire loop
    d.rectangle([300, 95, 460, 145], fill="white", outline="black", width=3)
    d.text((310, 40), "R1 = 10 Ω", fill="black", font=f)
    d.rectangle([560, 95, 720, 145], fill="white", outline="black", width=3)
    d.text((570, 40), "R2 = 20 Ω", fill="black", font=f)
    d.rectangle([120, 240, 180, 320], fill="white", outline="white")     # battery gap
    d.line([(110, 260), (190, 260)], fill="black", width=5)
    d.line([(130, 300), (170, 300)], fill="black", width=5)
    d.text((200, 262), "12 V", fill="black", font=f)
    d.text((140, 220), "+", fill="black", font=f)
    return img


def code_snippet():
    img, d = canvas(800, 360)
    f = font(34)
    lines = ["total = 0", "for i in range(1, 6):", "    total += i", "print(total)"]
    d.rectangle([30, 30, 770, 330], outline="#888888", width=2)
    for i, line in enumerate(lines):
        d.text((60, 60 + i * 62), line, fill="black", font=f)
    return img


def flowchart():
    img, d = canvas(900, 900)
    f = font(28)
    d.ellipse([350, 20, 550, 90], outline="black", width=3)
    d.text((415, 38), "Start", fill="black", font=f)
    arrow(d, (450, 90), (450, 140))
    d.polygon([(330, 140), (590, 140), (560, 210), (300, 210)], outline="black")
    d.text((380, 160), "Read n", fill="black", font=f)
    arrow(d, (450, 210), (450, 270))
    d.polygon([(450, 270), (620, 370), (450, 470), (280, 370)], outline="black")
    d.text((365, 352), "n mod 2 = 0 ?", fill="black", font=font(24))
    arrow(d, (280, 370), (150, 370)); d.text((190, 330), "Yes", fill="black", font=f)
    arrow(d, (620, 370), (750, 370)); d.text((650, 330), "No", fill="black", font=f)
    box(d, (40, 400, 260, 480), 'Print "Even"', font(24))
    box(d, (640, 400, 870, 480), 'Print "Odd"', font(24))
    arrow(d, (150, 480), (150, 600)); arrow(d, (755, 480), (755, 600))
    d.line([(150, 600), (755, 600)], fill="black", width=3)
    arrow(d, (450, 600), (450, 660))
    d.ellipse([350, 660, 550, 730], outline="black", width=3)
    d.text((425, 678), "End", fill="black", font=f)
    return img


ITEMS = [
    ("truth_table.png", truth_table, "Which logic gate does this truth table represent? Explain.",
     "2-mark", "AND gate|AND|and gate|and-gate", "truth table"),
    ("block_diagram.png", block_diagram, "What system does this block diagram show? Explain the role of each block.",
     "5-mark", "communication;encoder|encoding|encodes|encode;decoder|decoding|decodes|decode;channel;noise|noisy", "block diagram"),
    ("sine_plot.png", sine_plot, "What are the amplitude and frequency of this signal?",
     "2-mark", "5 v|5v|5 volt;50 hz|50hz|50 hertz", "graph"),
    ("bar_chart.png", bar_chart, "Which pump has the highest flow rate, and what is it?",
     "2-mark", "pump b;50", "chart"),
    ("equation.png", equation, "Solve the problem shown in the image.",
     "2-mark", "10 v|10v|10 volt", "equation"),
    ("series_circuit.png", series_circuit, "Find the total resistance and the current in this circuit.",
     "5-mark", "30 Ω|30Ω|30 ohm|30 ohms|30-ohm;0.4 A|0.4A|0.4 amp|0.4 ampere|400 mA", "circuit"),
    ("code_snippet.png", code_snippet, "What does this code print? Explain why.",
     "2-mark", "15", "code"),
    ("flowchart.png", flowchart, "What does this flowchart do?",
     "2-mark", "even;odd", "flowchart"),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, draw, question, mode, expected, kind in ITEMS:
        draw().save(OUT / name)
        rows.append({"file": name, "question": question, "answer_mode": mode, "expected": expected,
                     "kind": kind, "source": "synthetic starter set"})
    with (OUT / "questions.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "question", "answer_mode", "expected", "kind", "source"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} images + questions.csv -> {OUT}")


if __name__ == "__main__":
    main()
