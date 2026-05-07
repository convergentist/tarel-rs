# Article — NeurIPS 2026 submission

Working directory for the NeurIPS 2026 paper.

## Files

- `main.tex` — the paper (working copy of the NeurIPS 2026 shell template). Edit this.
- `checklist.tex` — NeurIPS responsible-research checklist, `\input` from `main.tex`. Required; do not delete.
- `neurips_2026.sty` — official NeurIPS 2026 style file. Do not modify (tweaking it is grounds for desk rejection).
- `Makefile` — build targets.
- `formats/Formatting_Instructions_For_NeurIPS_2026/` — pristine reference copies of the official template. Source of truth; do not edit.

## Build

```sh
make           # full build: pdflatex -> bibtex -> pdflatex -> pdflatex
make quick     # single pdflatex pass (fast, refs may be stale)
make watch     # rebuild on save (requires latexmk)
make clean     # remove .aux/.log/etc., keep main.pdf
make distclean # also remove main.pdf
```

Requires a TeX distribution with `pdflatex` (TeX Live or MacTeX). For `make watch`, install `latexmk`.

## Track / option selection

`main.tex` loads the style as `\usepackage{neurips_2026}` (default = anonymized main-track submission). The other tracks are commented out at the top of the file:

- `[main]`, `[position]`, `[eandd]`, `[creativeai]`, `[sglblindworkshop]`, `[dblblindworkshop]`
- Add `final` (e.g. `[main, final]`) only after acceptance.
- `[preprint]` for arXiv-style preprints.

At submission time, leave both `final` and `preprint` off so line numbers and anonymization are enabled.
