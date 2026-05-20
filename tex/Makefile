# =============================================================================
# TAReL — selective LaTeX builds
# =============================================================================
# Each driver compiles into its own ./artifacts/<kind>/ folder.  Output
# routing is set in ./latexmkrc; this Makefile is just a thin wrapper that
# exposes one-word targets for the common workflows.
#
#   make article          -> drivers/article.tex           -> artifacts/article/
#   make article-2        -> drivers/article-2.tex         -> artifacts/article-2/
#   make notes-generator  -> drivers/notes-generator.tex   -> artifacts/notes/
#   make notes-critic     -> drivers/notes-critic.tex      -> artifacts/notes/
#   make notes            -> notes-generator + notes-critic
#   make figure-generator -> drivers/figure-generator.tex  -> artifacts/figures/
#   make figure-critic    -> drivers/figure-critic.tex     -> artifacts/figures/
#   make figures          -> figure-generator + figure-critic
#   make all              -> every driver above
#   make watch-<x>        -> continuous build (latexmk -pvc) of <x>
#   make clean            -> remove aux files, keep PDFs
#   make distclean        -> wipe artifacts/<*> entirely
# =============================================================================

LATEXMK  ?= latexmk
ARTDIR    = artifacts

ARTICLE_SRC          = drivers/article.tex
ARTICLE_2_SRC        = drivers/article-2.tex
NOTES_GENERATOR_SRC  = drivers/notes-generator.tex
NOTES_CRITIC_SRC     = drivers/notes-critic.tex
FIGURE_GENERATOR_SRC = drivers/figure-generator.tex
FIGURE_CRITIC_SRC    = drivers/figure-critic.tex

ARTICLE_DIR    = $(ARTDIR)/article
ARTICLE_2_DIR  = $(ARTDIR)/article-2
NOTES_DIR      = $(ARTDIR)/notes
FIGURE_DIR     = $(ARTDIR)/figures

.PHONY: help all \
        article article-2 \
        notes notes-generator notes-critic \
        figures figure-generator figure-critic \
        watch-article watch-article-2 \
        watch-notes-generator watch-notes-critic \
        watch-figure-generator watch-figure-critic \
        clean clean-article clean-article-2 \
        clean-notes-generator clean-notes-critic \
        clean-figure-generator clean-figure-critic \
        distclean

help:
	@echo "TAReL build targets"
	@echo "  make article             compile $(ARTICLE_SRC)"
	@echo "  make article-2           compile $(ARTICLE_2_SRC)"
	@echo "  make notes-generator     compile $(NOTES_GENERATOR_SRC)"
	@echo "  make notes-critic        compile $(NOTES_CRITIC_SRC)"
	@echo "  make notes               both notes-generator and notes-critic"
	@echo "  make figure-generator    compile $(FIGURE_GENERATOR_SRC)"
	@echo "  make figure-critic       compile $(FIGURE_CRITIC_SRC)"
	@echo "  make figures             both figure-generator and figure-critic"
	@echo "  make all                 every driver above"
	@echo "  make watch-<target>      continuous rebuild (latexmk -pvc) of <target>"
	@echo "  make clean               remove aux/log/bbl/... (keeps PDFs)"
	@echo "  make distclean           wipe artifacts/<article|notes|figures>"

all: article article-2 notes-generator notes-critic figure-generator figure-critic

article:
	@mkdir -p $(ARTICLE_DIR)
	$(LATEXMK) $(ARTICLE_SRC)

article-2:
	@mkdir -p $(ARTICLE_2_DIR)
	$(LATEXMK) $(ARTICLE_2_SRC)

notes-generator:
	@mkdir -p $(NOTES_DIR)
	$(LATEXMK) $(NOTES_GENERATOR_SRC)

notes-critic:
	@mkdir -p $(NOTES_DIR)
	$(LATEXMK) $(NOTES_CRITIC_SRC)

notes: notes-generator notes-critic

figure-generator:
	@mkdir -p $(FIGURE_DIR)
	$(LATEXMK) $(FIGURE_GENERATOR_SRC)

figure-critic:
	@mkdir -p $(FIGURE_DIR)
	$(LATEXMK) $(FIGURE_CRITIC_SRC)

figures: figure-generator figure-critic

watch-article:
	@mkdir -p $(ARTICLE_DIR)
	$(LATEXMK) -pvc $(ARTICLE_SRC)

watch-article-2:
	@mkdir -p $(ARTICLE_2_DIR)
	$(LATEXMK) -pvc $(ARTICLE_2_SRC)

watch-notes-generator:
	@mkdir -p $(NOTES_DIR)
	$(LATEXMK) -pvc $(NOTES_GENERATOR_SRC)

watch-notes-critic:
	@mkdir -p $(NOTES_DIR)
	$(LATEXMK) -pvc $(NOTES_CRITIC_SRC)

watch-figure-generator:
	@mkdir -p $(FIGURE_DIR)
	$(LATEXMK) -pvc $(FIGURE_GENERATOR_SRC)

watch-figure-critic:
	@mkdir -p $(FIGURE_DIR)
	$(LATEXMK) -pvc $(FIGURE_CRITIC_SRC)

clean-article:
	$(LATEXMK) -c $(ARTICLE_SRC)

clean-article-2:
	$(LATEXMK) -c $(ARTICLE_2_SRC)

clean-notes-generator:
	$(LATEXMK) -c $(NOTES_GENERATOR_SRC)

clean-notes-critic:
	$(LATEXMK) -c $(NOTES_CRITIC_SRC)

clean-figure-generator:
	$(LATEXMK) -c $(FIGURE_GENERATOR_SRC)

clean-figure-critic:
	$(LATEXMK) -c $(FIGURE_CRITIC_SRC)

clean: clean-article clean-article-2 \
       clean-notes-generator clean-notes-critic \
       clean-figure-generator clean-figure-critic

distclean:
	rm -rf $(ARTICLE_DIR) $(ARTICLE_2_DIR) $(NOTES_DIR) $(FIGURE_DIR)
	@mkdir -p $(ARTICLE_DIR) $(ARTICLE_2_DIR) $(NOTES_DIR) $(FIGURE_DIR)
