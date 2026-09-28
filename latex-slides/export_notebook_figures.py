#!/usr/bin/env python3
"""Copy figures out of the executed Lab 2 notebooks into figures/ for the deck.

Every "From the notebook" slide shows the image a student sees when they run
the cell, so the deck cannot drift from the tutorials. Re-run this after
re-executing a notebook:   python export_notebook_figures.py   (or: make figures)

Each entry names the figure, the notebook, a snippet that identifies the code
cell, and which image output of that cell to take.
"""
import base64, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOKS = HERE.parent/'Lab-2-tutorial-all-content'
ML, GENAI = 'part-1-ml', 'part-2-rag'

FIGURES = [
    # name,                 folder, notebook,                                           cell snippet,                          image
    ('ml0-distributions',   ML, 'Part-0_Healthcare_Data_Processing_and_Analytics.ipynb', 'Billing Amount distribution',        0),
    ('ml0-billing-by-type', ML, 'Part-0_Healthcare_Data_Processing_and_Analytics.ipynb', 'Billing Amount by Admission Type',   0),
    ('ml1a-confusion',      ML, 'Part-1A_Classification_ML_Pipeline.ipynb',              'confusion matrix',                   0),
    ('ml1a-roc',            ML, 'Part-1A_Classification_ML_Pipeline.ipynb',              'ROC curve',                          0),
    ('ml1b-target',         ML, 'Part-1B_Regression_ML_Pipeline.ipynb',                  'target distribution',                0),
    ('ml1b-residuals',      ML, 'Part-1B_Regression_ML_Pipeline.ipynb',                  'residuals vs predicted',             0),
    ('tok-cost',            GENAI, 'Part-0_Tokenization_for_Health_Text.ipynb',          'Tokens per word by text type',       0),
    ('w2v-similarity',      GENAI, 'Part-1_Word2Vec_and_tSNE.ipynb',                     'Selected-word cosine similarities',  0),
    ('w2v-tsne',            GENAI, 'Part-1_Word2Vec_and_tSNE.ipynb',                     't-SNE display of varied-corpus',     0),
    ('att-hand-example',    GENAI, 'Part-2_Attention_and_Self_Attention.ipynb',          '1. Scaled match scores',             0),
    ('att-self-vs-causal',  GENAI, 'Part-2_Attention_and_Self_Attention.ipynb',          'No mask: every token reads',         0),
    ('att-mask-gallery',    GENAI, 'Part-2_Attention_and_Self_Attention.ipynb',          'sliding = causal &',                 0),
    ('att-objectives',      GENAI, 'Part-2_Attention_and_Self_Attention.ipynb',          'AR: causal mask,',                   0),
    ('att-training',        GENAI, 'Part-2_Attention_and_Self_Attention.ipynb',          'Same architecture, same corpus',     0),
    ('att-prediction',      GENAI, 'Part-2_Attention_and_Self_Attention.ipynb',          'def prediction_figure',              0),
    ('att-pretrained',      GENAI, 'Part-2_Attention_and_Self_Attention.ipynb',          'AR_PROMPT =',                        0),
    ('att-pretrained-maps', GENAI, 'Part-2_Attention_and_Self_Attention.ipynb',          'lower triangle',                     0),
    ('rag-chunks',          GENAI, 'Part-3_RAG_With_and_Without_Retrieval.ipynb',        'Nine heading-based retrieval chunks', 0),
    ('rag-lexical-scores',  GENAI, 'Part-3_RAG_With_and_Without_Retrieval.ipynb',        'Lexical retrieval scores for every', 0),
]


def main() -> int:
    out = HERE/'figures'; out.mkdir(exist_ok=True)
    cache, missing = {}, []
    for name, folder, notebook, snippet, index in FIGURES:
        path = NOTEBOOKS/folder/notebook
        cells = cache.setdefault(path, json.loads(path.read_text(encoding='utf-8'))['cells'])
        images = [o['data']['image/png'] for c in cells if c['cell_type'] == 'code' and snippet in ''.join(c['source'])
                  for o in c.get('outputs', []) if 'image/png' in o.get('data', {})]
        if len(images) <= index:
            missing.append(f'{name}: no image for {snippet!r} in {notebook}')
            continue
        (out/f'{name}.png').write_bytes(base64.b64decode(images[index]))
        print(f'figures/{name}.png  <-  {notebook}')
    if missing:
        print('\n'.join(['', 'MISSING:'] + missing), file=sys.stderr)
    return 1 if missing else 0


if __name__ == '__main__':
    raise SystemExit(main())
