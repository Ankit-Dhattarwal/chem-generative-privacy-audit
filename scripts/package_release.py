#!/usr/bin/env python3
"""
scripts/package_release.py

Packages release_candidate/ankit_paper_release.zip adhering to Task T5:
EXCLUDES:
- RESEARCH_AUDIT_SUMMARY_FOR_CLAUDE.md
- ankit_paper_final.zip
- VERIFICATION_REPORT*.md
- .venv, .git, .pt, .parquet, __pycache__, .DS_Store, backups, _archive
"""

import os
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RC_DIR = REPO_ROOT / "release_candidate"
ZIP_PATH = RC_DIR / "ankit_paper_release.zip"

EXCLUDE_EXACT_NAMES = {
    "WORKLOG.md",
    "NEEDS_HUMAN.md",
    "FINAL_REPORT.md",
    "REVIEW_ROUND_1.md",
    "HUMAN_READ_CHECKLIST.md",
    "CODE_PAPER_MATRIX.md",
    "RESEARCH_AUDIT_SUMMARY_FOR_CLAUDE.md",
    "moses.csv",
}

EXCLUDE_PATTERNS = [
    "ankit_paper_final.zip",
    "ankit_paper_release.zip",
    ".venv",
    ".git",
    "__pycache__",
    ".DS_Store",
    "_archive",
    "backup",
    "release_candidate",
    "paper_revised",
    "VERIFICATION_REPORT",
    "REVIEW_ROUND_",
    "scratch",
]

EXCLUDE_EXTENSIONS = {
    ".pt",
    ".parquet",
    ".pyc",
    ".pyo",
    ".partial",
    ".tmp",
    ".log",
    ".aux",
    ".blg",
    ".fls",
    ".fdb_latexmk",
}

EXCLUDE_EXACT_PATHS = {"paper/main.txt"}

def should_exclude(rel_path_str: str) -> bool:
    path = Path(rel_path_str)
    if rel_path_str in EXCLUDE_EXACT_PATHS:
        return True
    if path.name in EXCLUDE_EXACT_NAMES:
        return True
    for p in path.parts:
        if p in EXCLUDE_EXACT_NAMES:
            return True
        for excl in EXCLUDE_PATTERNS:
            if excl in p:
                return True
    for excl in EXCLUDE_PATTERNS:
        if excl in rel_path_str:
            return True
    if any(rel_path_str.endswith(ext) for ext in EXCLUDE_EXTENSIONS):
        return True
    return False

def package():
    RC_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    included_count = 0
    seen_arc_names = set()
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(REPO_ROOT):
            # prune excluded dirs in place
            dirs[:] = [d for d in dirs if not should_exclude(str((Path(root) / d).relative_to(REPO_ROOT)))]
            for file in sorted(files):
                full_path = Path(root) / file
                rel_path = full_path.relative_to(REPO_ROOT)
                rel_path_str = str(rel_path)
                if should_exclude(rel_path_str):
                    continue
                arc_name = rel_path.as_posix()
                if arc_name.count("runs/cuda_arm") > 1:
                    raise ValueError(f"Nested duplicate run root in source path: {arc_name}")
                if arc_name in seen_arc_names:
                    raise ValueError(f"Duplicate archive path: {arc_name}")
                seen_arc_names.add(arc_name)
                zipf.write(full_path, arcname=arc_name)
                included_count += 1

    print(f"Packaged {included_count} files into {ZIP_PATH} ({ZIP_PATH.stat().st_size / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    package()
