"""Structural and teaching-contract checks for the Part 0-1B notebooks."""
from pathlib import Path
import re

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
LAB_ROOT = ROOT.parent

NOTEBOOKS = [
    'Part-0_Healthcare_Data_Processing_and_Analytics.ipynb',
    'Part-1A_Classification_ML_Pipeline.ipynb',
    'Part-1B_Regression_ML_Pipeline.ipynb',
]


def text_of(name):
    nb = nbformat.read(ROOT / name, as_version=4)
    return nb, '\n'.join(cell.source for cell in nb.cells)


@pytest.mark.parametrize('name', NOTEBOOKS)
def test_every_code_cell_has_markdown_before_it(name):
    nb = nbformat.read(ROOT / name, as_version=4)
    assert nb.cells
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            assert i > 0 and nb.cells[i - 1].cell_type == 'markdown', f'{name} cell {i}'


@pytest.mark.parametrize('name', NOTEBOOKS)
def test_notebook_is_executed_without_errors(name):
    nb = nbformat.read(ROOT / name, as_version=4)
    assert all(c.execution_count is not None for c in nb.cells if c.cell_type == 'code'), name
    for i, cell in enumerate(nb.cells):
        for output in cell.get('outputs', []):
            assert output.output_type != 'error', f'{name} cell {i}: {output.get("ename")}'


@pytest.mark.parametrize('name', NOTEBOOKS)
def test_important_outputs_are_explained(name):
    """The course style guide requires a 'What do we see?' cell after key outputs."""
    _, text = text_of(name)
    assert text.lower().count('what do we see') >= 3, name


@pytest.mark.parametrize('name', NOTEBOOKS)
def test_notebook_has_orientation_and_next_steps(name):
    _, text = text_of(name)
    assert 'Before you begin' in text, name
    assert 'Runtime:' in text, name


@pytest.mark.parametrize('name', NOTEBOOKS)
def test_notebook_states_its_limits(name):
    _, text = text_of(name)
    assert 'Limits' in text or 'limitation' in text.lower(), name


@pytest.mark.parametrize('name', NOTEBOOKS)
def test_no_absolute_home_path_is_required_to_run(name):
    """Students on Colab do not have the instructor's filesystem."""
    _, text = text_of(name)
    assert 'find_data_file' in text or '/home/bhux' not in text, name


@pytest.mark.parametrize('name', NOTEBOOKS)
def test_code_cells_avoid_semicolon_chaining(name):
    """One statement per line: chained statements are unreadable for beginners."""
    nb, _ = text_of(name)
    offenders = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != 'code':
            continue
        for line in cell.source.splitlines():
            stripped = line.strip()
            if stripped.startswith('#') or '"' in stripped or "'" in stripped:
                continue
            if re.search(r';\s*\S', stripped):
                offenders.append(f'{name} cell {i}: {stripped[:60]}')
    assert not offenders, offenders


def test_part0_frames_columns_by_availability_and_asks_the_data_owner():
    _, text = text_of('Part-0_Healthcare_Data_Processing_and_Analytics.ipynb')
    assert 'Question for the data owner' in text
    assert 'feature_policy' in text
    assert 'Future information' in text and 'Identifier' in text
    # Must not quietly delete the awkward values.
    assert 'non-positive count' in text


def test_part1a_teaches_categorical_preprocessing():
    """The slide deck and instructor guide promise one-hot encoding; it must exist."""
    _, text = text_of('Part-1A_Classification_ML_Pipeline.ipynb')
    assert 'ColumnTransformer' in text
    assert 'OneHotEncoder' in text
    assert 'handle_unknown="ignore"' in text
    assert 'get_feature_names_out' in text


def test_part1a_compares_against_a_baseline_and_splits_first():
    _, text = text_of('Part-1A_Classification_ML_Pipeline.ipynb')
    assert 'DummyClassifier' in text
    assert 'stratify=y' in text
    assert 'StratifiedKFold' in text
    assert text.index('train_test_split(') < text.index('cross_validate(')


def test_part1b_demonstrates_preprocessing_leakage():
    _, text = text_of('Part-1B_Regression_ML_Pipeline.ipynb')
    assert 'SelectKBest' in text
    assert 'leaky_scores' in text and 'honest_pipeline' in text
    assert 'pure noise' in text.lower() or 'pure random noise' in text.lower()


def test_part1b_explains_the_negated_scorer_convention():
    _, text = text_of('Part-1B_Regression_ML_Pipeline.ipynb')
    assert 'neg_mean_absolute_error' in text
    assert 'negat' in text.lower()
    assert 'negative R' in text


def test_handouts_referenced_by_notebooks_exist():
    handouts = LAB_ROOT / 'handouts'
    assert handouts.is_dir()
    for name in ['pipeline_map.md', 'data_splitting_map.md',
                 'metric_chooser.md', 'leakage_checklist.md']:
        assert (handouts / name).is_file(), name
