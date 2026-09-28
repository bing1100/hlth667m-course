# Build the Lab 2 deck. Intermediate files go to build/; the PDF is copied here.
DECK = lab2-slides

all: $(DECK).pdf

$(DECK).pdf: $(DECK).tex hlth667m.sty part0-opening.tex part1-ml.tex part2-genai.tex $(wildcard figures/*.png)
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build $(DECK).tex
	cp build/$(DECK).pdf $(DECK).pdf

# Re-export the "From the notebook" figures after re-executing a notebook.
figures:
	python3 export_notebook_figures.py

# Confirm that speaker-script.md has one entry per slide, in deck order.
script:
	python3 list_slides.py --check

check: $(DECK).pdf
	@grep -c "Overfull" build/$(DECK).log | xargs echo "overfull boxes:"
	@grep "Overfull" build/$(DECK).log | sed -E 's/.*\(([0-9.]+)pt too (high|wide)\).*/\2 \1pt/' | awk '$$2+0>2.5' || true

sheets: $(DECK).pdf
	./render.sh build/sheets

clean:
	rm -rf build

.PHONY: all figures script check sheets clean
