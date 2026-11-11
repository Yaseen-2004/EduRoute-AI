"""
Package the committed project for a Linux GPU server:
deploy/eduroute_server_package.tar.gz (code, configs, the FAISS indexes,
benchmarks, deploy scripts).

Only committed files go in, so .env, grading keys, voice recordings, logs and
backups (all git-ignored) can never be included. Raw and processed corpora are
left out: the app and the evaluations only need the indexes. Text files get LF
line endings whatever the Windows checkout uses.

Run: venv\\Scripts\\python deploy\\make_server_package.py
On the server:  mkdir eduroute && tar -xzf eduroute_server_package.tar.gz -C eduroute
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "deploy" / "eduroute_server_package.tar.gz"
EXCLUDE = ["data/processed", "data/raw", "colab/*.ipynb", "docs/report"]


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True, text=True).stdout


def main():
    dirty = git("status", "--porcelain", "--untracked-files=no").strip()
    if dirty:
        print("Uncommitted changes are NOT in the package (commit them first):")
        print(dirty)
        if "--allow-dirty" not in sys.argv:
            sys.exit(1)
    commit = git("log", "-1", "--format=%h %s").strip()
    pathspec = ["."] + [f":(exclude){p}" for p in EXCLUDE]
    git("-c", "core.autocrlf=false", "archive", "--format=tar.gz", "-o", str(OUT),
        f"--add-virtual-file=VERSION:{commit}\n", "HEAD", "--", *pathspec)
    print(f"{commit}\n-> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
