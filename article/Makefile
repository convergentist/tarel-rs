# NeurIPS 2026 article build
#
# Targets:
#   make            -> build main.pdf via pdflatex + bibtex
#   make quick      -> single pdflatex pass (fast, may have stale refs)
#   make watch      -> rebuild on save (requires latexmk)
#   make clean      -> remove LaTeX build artifacts (keeps PDF)
#   make distclean  -> remove build artifacts and PDF

MAIN     := main
TEX      := $(MAIN).tex
PDF      := $(MAIN).pdf
BIB      := $(MAIN).bib
LATEX    := pdflatex -interaction=nonstopmode -halt-on-error
LATEXMK  := latexmk -pdf -interaction=nonstopmode -halt-on-error

AUX_EXT  := aux bbl blg log out toc fls fdb_latexmk synctex.gz nav snm vrb

.PHONY: all quick watch clean distclean

all: $(PDF)

$(PDF): $(TEX) checklist.tex neurips_2026.sty $(wildcard *.bib)
	$(LATEX) $(MAIN)
	@if [ -f $(BIB) ]; then bibtex $(MAIN) || true; fi
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

quick:
	$(LATEX) $(MAIN)

watch:
	$(LATEXMK) -pvc $(MAIN)

clean:
	@for ext in $(AUX_EXT); do rm -f $(MAIN).$$ext; done

distclean: clean
	rm -f $(PDF)
