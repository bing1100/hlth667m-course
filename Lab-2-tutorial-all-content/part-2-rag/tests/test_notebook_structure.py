"""Structural checks on the four Tutorial 2 notebooks (Part 0-3), the chat app and the slide deck."""
from pathlib import Path
import re
import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]

NOTEBOOKS = [
    'Part-0_Tokenization_for_Health_Text.ipynb',
    'Part-1_Word2Vec_and_tSNE.ipynb',
    'Part-2_Attention_and_Self_Attention.ipynb',
    'Part-3_RAG_With_and_Without_Retrieval.ipynb',
]


def text_of(name):
    nb = nbformat.read(ROOT / name, as_version=4)
    return nb, '\n'.join(cell.source for cell in nb.cells)


@pytest.mark.parametrize('name', NOTEBOOKS)
def test_notebook_parses_and_every_code_cell_has_markdown_before_it(name):
    nb = nbformat.read(ROOT / name, as_version=4)
    assert nb.cells
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            assert i > 0 and nb.cells[i - 1].cell_type == 'markdown', f'{name} cell {i}'


@pytest.mark.parametrize('name', NOTEBOOKS)
def test_notebook_is_executed(name):
    nb = nbformat.read(ROOT / name, as_version=4)
    assert all(c.execution_count is not None for c in nb.cells if c.cell_type == 'code'), name


@pytest.mark.parametrize('name', NOTEBOOKS)
def test_notebook_has_no_stored_errors(name):
    nb = nbformat.read(ROOT / name, as_version=4)
    for i, cell in enumerate(nb.cells):
        for output in cell.get('outputs', []):
            assert output.output_type != 'error', f'{name} cell {i}: {output.get("ename")}'


@pytest.mark.parametrize('name', NOTEBOOKS)
def test_no_credential_shaped_value_is_stored(name):
    _, text = text_of(name)
    assert not re.search(r'sk-[A-Za-z0-9_-]{20,}', text), name


@pytest.mark.parametrize('name', NOTEBOOKS)
def test_notebook_states_limits_and_uses_what_do_we_see_scaffolding(name):
    _, text = text_of(name)
    assert 'what do we see' in text.lower() or 'Learner check' in text, name
    assert 'Limits:' in text or 'limitations' in text.lower(), name


def test_tokenization_notebook_builds_bpe_and_connects_to_cost():
    _, text = text_of('Part-0_Tokenization_for_Health_Text.ipynb')
    for phrase in ['learn_bpe_merges', 'apply_one_merge', 'byte pair encoding',
                   'tokens per word', 'cl100k_base', 'subword']:
        assert phrase.lower() in text.lower()
    # The production tokenizer must be optional, not required.
    assert 'PRODUCTION_TOKENIZER is None' in text
    assert 'try:' in text and 'except Exception' in text


def test_word2vec_notebook_compares_repetitive_and_varied_corpora():
    _, text = text_of('Part-1_Word2Vec_and_tSNE.ipynb')
    assert 'baseline_model' in text and 'varied contexts' in text
    assert 'cosine similarities' in text and 'Teaching annotation' in text


def test_attention_notebook_maps_masks_to_training_objectives():
    nb, text = text_of('Part-2_Attention_and_Self_Attention.ipynb')
    for phrase in ['weighted average', 'Read one row slowly', 'gallery of mask patterns', 'Sliding window',
                   'Autoregressive (AR)', 'Masked language modelling (MLM)', 'prediction head',
                   'leak', 'distilgpt2', 'distilbert-base-uncased', 'Try it yourself']:
        assert phrase.lower() in text.lower(), phrase
    assert 'scaled_dot_product_attention_from_scratch' in text
    assert 'torch.testing.assert_close' in text
    # The pretrained section must be optional, like the production tokenizer in Part 0.
    assert 'PRETRAINED is None' in text and 'except Exception' in text
    # Decoding moved to the slides; the notebook has to say so.
    assert 'temperature, top-k and top-p' in text
    assert sum('plt.subplots' in c.source for c in nb.cells if c.cell_type == 'code') >= 6


def test_attention_notebook_stores_the_leak_test_result():
    nb, _ = text_of('Part-2_Attention_and_Self_Attention.ipynb')
    cell = next(c for c in nb.cells if c.cell_type == 'code' and 'first_changed' in c.source)
    printed = ''.join(o.get('text', '') for o in cell.outputs if o.output_type == 'stream')
    assert "'AR (causal mask)': 0.0" in printed


def test_rag_notebook_builds_tests_and_launches_a_local_chat_app():
    _, text = text_of('Part-3_RAG_With_and_Without_Retrieval.ipynb')
    for phrase in ['%%writefile rag_chat_app.py', 'AppTest.from_file', "'--server.address', 'localhost'",
                   'STOP_CHAT_APP', 'st.session_state', 'Never upload patient data']:
        assert phrase in text, phrase
    # The file on disk is the one the notebook writes.
    written = next(c.source for c in nbformat.read(ROOT / 'Part-3_RAG_With_and_Without_Retrieval.ipynb', as_version=4).cells
                   if c.source.startswith('%%writefile rag_chat_app.py'))
    assert written.split('\n', 1)[1].strip() == (ROOT / 'rag_chat_app.py').read_text().strip()


def test_chat_app_answers_offline_in_retrieval_only_mode(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.chdir(ROOT)
    app = AppTest.from_file(str(ROOT / 'rag_chat_app.py'), default_timeout=60).run()
    app.chat_input[0].set_value('Who tells me about my bloodwork if something is wrong?').run()
    assert not app.exception
    reply = app.session_state['history'][-1]
    assert reply['note'].startswith('RETRIEVAL ONLY')
    assert [c['chunk_id'] for c in reply['retrieved']][:3] == ['results_policy::000', 'results_policy::002', 'results_policy::001']


def test_rag_notebook_supports_replay_and_live_without_exposing_key():
    _, text = text_of('Part-3_RAG_With_and_Without_Retrieval.ipynb')
    assert "HLTH667M_RUN_MODE" in text
    assert "'replay'" in text and "'live'" in text
    # Replay must be clearly labelled and must never be called a fresh response.
    assert 'RECORDED' in text
    assert 'does not invent one' in text or 'never invents a response' in text
    assert 'OPENAI_API_KEY' in text


def test_rag_notebook_covers_chunking_terminology_graph_and_audit():
    _, text = text_of('Part-3_RAG_With_and_Without_Retrieval.ipynb')
    for phrase in ['chunk_markdown_fixed_window', 'Compare two chunking strategies',
                   'load_terminology', 'expand_query_with_terminology',
                   'load_knowledge_graph', 'graph_paths_from',
                   'partial evidence', 'Retrieval gate', 'Generation gate',
                   'citation_audit_exercise']:
        assert phrase in text, phrase


def test_rag_notebook_has_four_evaluation_cases():
    _, text = text_of('Part-3_RAG_With_and_Without_Retrieval.ipynb')
    for case in ['Q1_SINGLE', 'Q2_MULTI', 'Q3_UNANSWERABLE', 'Q4_PARTIAL']:
        assert case in text


def test_slide_deck_has_numbered_slides_with_required_fields():
    text = (ROOT / 'slides.md').read_text()
    slides = re.split(r'(?=^## Slide \d+)', text, flags=re.M)[1:]
    assert len(slides) >= 24
    for slide in slides:
        for field in ['**Visible text:**', '**Visual:**', '**Alt text:**', '**Say:**']:
            assert field in slide, slide[:80]


def test_slide_deck_keeps_renderable_math():
    text = (ROOT / 'slides.md').read_text()
    assert text.count('$$') >= 20 and text.count('$$') % 2 == 0
    for formula in [r'\operatorname{cos\_sim}', r'\mathbf{Q}\mathbf{K}^{\mathsf T}',
                    r'\operatorname{softmax}']:
        assert formula in text
