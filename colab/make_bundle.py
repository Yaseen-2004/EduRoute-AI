"""
Package what the Colab benchmark needs into colab/eduroute_colab_bundle.zip:
code, configs, benchmarks and the engineering index. Excludes the venv, raw
data, logs, old results and anything secret (.env).

Run: venv\\Scripts\\python colab\\make_bundle.py
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "colab" / "eduroute_colab_bundle.zip"

INCLUDE_DIRS = ["config", "src", "evaluation"]
INCLUDE_FILES = [
    "app.py", "requirements.txt", ".env.example", "pytest.ini",
    "vector_store/engineering_faiss_v1.index",
    "vector_store/engineering_faiss_v1_metadata.jsonl",
]
SKIP_PARTS = {"__pycache__", "results"}
SKIP_NAMES = {".env"}


def files():
    for d in INCLUDE_DIRS:
        for path in (ROOT / d).rglob("*"):
            rel = path.relative_to(ROOT)
            if path.is_file() and not SKIP_PARTS & set(rel.parts) and path.name not in SKIP_NAMES:
                yield rel
    for f in INCLUDE_FILES:
        yield Path(f)


def main():
    count = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in files():
            zf.write(ROOT / rel, f"eduroute/{rel.as_posix()}")
            count += 1
    print(f"{count} files -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
