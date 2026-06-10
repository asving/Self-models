# Builds both documents:
#   design.tex          -- the working experimental-design document (primary)
#   grant_proposal.tex  -- the grant version (full motivation)
# Both share strand_two.tex, strand_three.tex, and references.bib.

DOCS = design grant_proposal

.PHONY: all clean distclean $(DOCS)

# Prefer latexmk; fall back to plain pdflatex+bibtex if unavailable.
HAS_LATEXMK := $(shell command -v latexmk 2>/dev/null)

all: $(DOCS)

ifdef HAS_LATEXMK
$(DOCS): %: %.tex strand_two.tex strand_three.tex references.bib
	latexmk -pdf -bibtex -interaction=nonstopmode -halt-on-error $<

clean:
	latexmk -c $(addsuffix .tex,$(DOCS))

distclean:
	latexmk -C $(addsuffix .tex,$(DOCS))
else
$(DOCS): %: %.tex strand_two.tex strand_three.tex references.bib
	pdflatex -interaction=nonstopmode -halt-on-error $*.tex
	bibtex $*
	pdflatex -interaction=nonstopmode -halt-on-error $*.tex
	pdflatex -interaction=nonstopmode -halt-on-error $*.tex

clean:
	rm -f $(foreach d,$(DOCS),$d.aux $d.log $d.out $d.bbl $d.blg $d.toc $d.fls $d.fdb_latexmk $d.run.xml $d.synctex.gz)

distclean: clean
	rm -f $(addsuffix .pdf,$(DOCS))
endif
