# HLTH 667M — Lab 2 slide deck

One Beamer deck covering both Lab 2 tutorials in
`../../hlth-667-668/Lab-2-tutorial/`:

- **Tutorial 1 — Machine-learning pipelines** (notebooks Part 0, 1A, 1B)
- **Tutorial 2 — From tokens to grounded answers** (notebooks Part 0 to 3, plus a
  slides-only section on decoding)

`lab2-slides.pdf` is the built deck: 106 pages, 16:9.

## How the deck follows the notebooks

Each key notebook section gets an explanation slide, followed by a
**From the notebook** slide that shows the section's executed output:

- `\nbexample` slides embed the figure a student sees when they run the cell.
  The images in `figures/` are exported from the executed notebooks in
  `../Lab-2-tutorial-all-content/` by `export_notebook_figures.py`.
- `\nbtag` marks a slide that redraws a notebook table or number natively. The
  tag names the part and section, for example *Part 3 §7*.

Slides may go further than the notebooks. Those slides carry no tag: the
`√d_k` demonstration, positional encoding ("Beyond the notebook"), and the
whole decoding section (temperature, top-k, top-p), whose numbers come from
the archived decoding notebook in `part-2-rag/archive/`.

Five GraphRAG slides follow the knowledge-graph slide in Part 3. The first two
explain GraphRAG. The last three compare it with RAG using numbers from Han et
al., *RAG vs. GraphRAG: A Systematic Evaluation and Key Insights*
([arXiv:2502.11371](https://arxiv.org/abs/2502.11371), v3, March 2026), and
carry a `\papertag`.

## Build

```bash
make            # builds build/lab2-slides.pdf and copies it here
make figures    # re-exports figures/ from the executed notebooks
make script     # checks that speaker-script.md covers every slide, in order
make check      # lists any overfull boxes larger than 2.5pt
make sheets     # renders contact sheets into build/sheets for a visual check
make clean
```

Needs TeX Live with `beamer`, `pgfplots`, `tcolorbox`, `fontawesome5`,
`FiraSans`, `FiraMono` and `newtxsf`. Built with pdfLaTeX through `latexmk`.
`make sheets` also needs `pdftoppm` and ImageMagick's `montage`.

## Files

| File | Contents |
|---|---|
| `lab2-slides.tex` | Main file and title slide. |
| `hlth667m.sty` | The look: colours, fonts, frame title, footer, and the reusable pieces below. |
| `part0-opening.tex` | Roadmap and ground rules. |
| `part1-ml.tex` | Tutorial 1 slides. |
| `part2-genai.tex` | Tutorial 2 slides. |
| `speaker-script.md` | What to say on every slide, with notebook hand-offs, questions for the room and a time plan. One entry per PDF page. |
| `list_slides.py` | Lists the slides in deck order. `--check` confirms that `speaker-script.md` still matches the deck. |
| `export_notebook_figures.py` | Copies figures out of the executed notebooks into `figures/`. One table row per figure. |
| `figures/` | Exported notebook figures. Generated; re-run `make figures` after re-executing a notebook. |

## Style pieces (`hlth667m.sty`)

| Macro | Use |
|---|---|
| `\tutorialslide{tag}{title}{line}` | Dark divider opening a tutorial. |
| `\notebookslide{tag}{title}{question}` | Light divider opening a notebook, with its guiding question. |
| `\nbexample[height]{Part 2 \S5}{title}{figure}{takeaway}` | A whole "From the notebook" slide built around an exported figure. Lower `height` if a two-line takeaway reaches the footer. |
| `\nbtag{Part 3 \S7}` | Footer tag for a slide that reproduces a notebook output natively. Put it first inside the frame. |
| `\papertag{Han et al., arXiv:2502.11371}` | Footer tag for a slide whose numbers come from a published paper. Put it first inside the frame. |
| `\takeaway{...}` | The one-sentence conclusion under a slide's content. Keep it to two lines. |
| `\keynumber[colour]{value}{caption}` | A large statistic with a caption. |
| `\tok[colour]{text}` | A token chip. Set `\toksize` inside a frame to resize. |
| `\code{...}`, `\eyebrow{...}`, `\soft{...}` | Inline code, small-caps label, muted text. |
| TikZ `stage`, `flow`, `note`, `cell`, `kn` | Boxes, arrows and labels for diagrams. |
| pgfplots `clean`, `cleanx` | Chart styles for vertical and horizontal bars. |

The palette is Okabe–Ito (colour-blind safe), matching the notebook figures.

## Conventions

- Slides carry content and visuals only. What to say, questions for the room
  and notebook hand-offs are in `speaker-script.md`. After adding, removing or
  retitling a slide, update the script and run `make script`.
- Explanation visuals are drawn in TikZ or pgfplots. The only image files are
  the notebook figures in `figures/`, and they appear only on `\nbexample` slides.
- To add an example slide: add a row to `FIGURES` in
  `export_notebook_figures.py`, run `make figures`, then add an `\nbexample`
  after the explanation slide it illustrates.
- Every number on a slide comes from the executed notebooks (see each
  tutorial's `VALIDATION.md`), except on `\papertag` slides, whose numbers
  come from the cited paper. Three visuals are schematic and say so on the
  slide: the BPE merge order, the t-SNE scatter, and the two density curves on
  the prediction-compression slide, which are drawn from the measured standard
  deviations.
- Do not put `#` inside a `frame` (for example in a TikZ `/.style`). Define
  the style in `hlth667m.sty` instead.
- Start a coloured entry in a `p{}` table column with `\leavevmode`, or the
  row gains blank height.
