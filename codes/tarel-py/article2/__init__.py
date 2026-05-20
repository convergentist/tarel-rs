"""tarel-py.article2 — Stage-1-only results pipeline for article-2.

Consumes a TAReL run directory (artifacts/runs/<exp_seed>/) and emits
2 booktabs LaTeX tables and 4 PNG plots into article-2's section tree.

CLI:
    python -m tarel_py.article2.run_all \
        --run-dir <abs path to run> \
        --out-dir <abs path to sections-v2/04-results-discussion>
"""
