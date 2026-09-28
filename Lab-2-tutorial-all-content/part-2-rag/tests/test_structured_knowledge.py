"""Checks on the second chunking strategy, the terminology, the graph, and replay."""
from pathlib import Path
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tutorial_utils import (  # noqa: E402
    load_manifest, chunk_markdown_by_h2, chunk_markdown_fixed_window, corpus_fingerprint,
    load_terminology, match_concepts, expand_query_with_terminology,
    load_knowledge_graph, graph_paths_from, graph_evidence_ids,
    load_recorded_run, token_overlap_scores, validate_citations,
)

CORPUS_DIR = ROOT / 'data/rag_corpus'
TERMINOLOGY_DIR = ROOT / 'data/terminology'
RECORDED_DIR = ROOT / 'data/recorded_runs'


@pytest.fixture(scope='module')
def manifest():
    return load_manifest(CORPUS_DIR)


@pytest.fixture(scope='module')
def heading_chunks(manifest):
    return [c for r in manifest for c in chunk_markdown_by_h2(CORPUS_DIR / r['filename'], r)]


# ---------------------------------------------------------------- chunking

def test_fixed_window_chunking_is_deterministic_and_respects_overlap(manifest):
    first = [c for r in manifest for c in chunk_markdown_fixed_window(CORPUS_DIR / r['filename'], r)]
    second = [c for r in manifest for c in chunk_markdown_fixed_window(CORPUS_DIR / r['filename'], r)]
    assert [c['chunk_id'] for c in first] == [c['chunk_id'] for c in second]
    assert len({c['chunk_id'] for c in first}) == len(first)
    assert all(len(c['text'].split()) <= 40 for c in first)


def test_window_size_changes_the_number_of_chunks(manifest):
    record = manifest[0]
    small = chunk_markdown_fixed_window(CORPUS_DIR / record['filename'], record, 20, 5)
    large = chunk_markdown_fixed_window(CORPUS_DIR / record['filename'], record, 80, 10)
    assert len(small) > len(large)


def test_overlap_must_be_smaller_than_the_window(manifest):
    record = manifest[0]
    with pytest.raises(ValueError, match='overlap_words'):
        chunk_markdown_fixed_window(CORPUS_DIR / record['filename'], record, 20, 20)


def test_the_two_strategies_produce_disjoint_id_namespaces(manifest, heading_chunks):
    windows = [c for r in manifest for c in chunk_markdown_fixed_window(CORPUS_DIR / r['filename'], r)]
    assert not ({c['chunk_id'] for c in heading_chunks} & {c['chunk_id'] for c in windows})


def test_fingerprint_distinguishes_chunking_algorithms(heading_chunks):
    assert corpus_fingerprint(heading_chunks, 'm', 'h2-v1') != corpus_fingerprint(heading_chunks, 'm', 'window-v1')


# ------------------------------------------------------------- terminology

def test_terminology_is_fictional_and_well_formed():
    terminology = load_terminology(TERMINOLOGY_DIR / 'health_terminology.json')
    assert terminology['code_system'] == 'NORTHSTAR-CT'
    codes = [c['code'] for c in terminology['concepts']]
    assert len(codes) == len(set(codes))
    # Must not imply a real, licensed terminology release.
    assert 'fictional' in terminology['code_system_version']


def test_patient_language_matches_a_concept_and_adds_corpus_vocabulary():
    terminology = load_terminology(TERMINOLOGY_DIR / 'health_terminology.json')
    expansion = expand_query_with_terminology('Who tells me about my bloodwork?', terminology)
    assert [c['code'] for c in expansion['matched_concepts']] == ['NC-00412']
    assert 'laboratory result' in expansion['added_terms']
    assert expansion['expanded_query'] != expansion['original_query']


def test_expansion_adds_no_duplicate_terms():
    terminology = load_terminology(TERMINOLOGY_DIR / 'health_terminology.json')
    added = expand_query_with_terminology('lab result in the online account', terminology)['added_terms']
    assert len(added) == len(set(added))


def test_unmatched_query_is_left_unchanged():
    terminology = load_terminology(TERMINOLOGY_DIR / 'health_terminology.json')
    expansion = expand_query_with_terminology('What is the parking fee?', terminology)
    assert expansion['matched_concepts'] == []
    assert expansion['expanded_query'] == expansion['original_query']


def test_expansion_improves_retrieval_for_patient_language(heading_chunks):
    terminology = load_terminology(TERMINOLOGY_DIR / 'health_terminology.json')
    question = 'Who tells me about my bloodwork if something is wrong?'
    expansion = expand_query_with_terminology(question, terminology)
    before = dict(token_overlap_scores(question, heading_chunks))
    after = dict(token_overlap_scores(expansion['expanded_query'], heading_chunks))
    assert after['results_policy::001'] > before['results_policy::001']


# ---------------------------------------------------------- knowledge graph

def test_every_graph_edge_carries_traceable_evidence(heading_chunks):
    graph = load_knowledge_graph(TERMINOLOGY_DIR / 'care_knowledge_graph.json')
    known_ids = {c['chunk_id'] for c in heading_chunks}
    for edge in graph['edges']:
        assert edge['evidence'] in known_ids, edge


def test_graph_rejects_a_dangling_edge(tmp_path):
    graph = json.loads((TERMINOLOGY_DIR / 'care_knowledge_graph.json').read_text())
    graph['edges'].append({'subject': 'NC-00412', 'predicate': 'x',
                           'object': 'does_not_exist', 'evidence': 'results_policy::000'})
    broken = tmp_path / 'broken.json'
    broken.write_text(json.dumps(graph))
    with pytest.raises(ValueError, match='does not exist'):
        load_knowledge_graph(broken)


def test_one_hop_is_more_precise_than_two_hops():
    graph = load_knowledge_graph(TERMINOLOGY_DIR / 'care_knowledge_graph.json')
    one = graph_evidence_ids(graph_paths_from(graph, ['NC-00412'], hops=1))
    two = graph_evidence_ids(graph_paths_from(graph, ['NC-00412'], hops=2))
    assert set(one) < set(two), 'two hops should reach strictly more evidence'
    assert all(cid.startswith('results_policy') for cid in one)


def test_graph_walk_reaches_the_urgent_result_clinician():
    graph = load_knowledge_graph(TERMINOLOGY_DIR / 'care_knowledge_graph.json')
    edges = graph_paths_from(graph, ['NC-00412'], hops=2)
    assert any(e['object'] == 'designated_clinician' for e in edges)
    assert 'results_policy::001' in graph_evidence_ids(edges)


def test_graph_walk_terminates_and_does_not_repeat_edges():
    graph = load_knowledge_graph(TERMINOLOGY_DIR / 'care_knowledge_graph.json')
    edges = graph_paths_from(graph, ['NC-00412'], hops=10)
    keys = [(e['subject'], e['predicate'], e['object']) for e in edges]
    assert len(keys) == len(set(keys))


# ------------------------------------------------------- recorded-run replay

def test_recorded_run_is_complete_and_matches_real_chunk_ids(heading_chunks):
    recorded = load_recorded_run(RECORDED_DIR / 'rag_recorded_run_2026-09-19.json')
    known_ids = {c['chunk_id'] for c in heading_chunks}
    assert set(recorded['records']) == {'Q1_SINGLE', 'Q2_MULTI', 'Q3_UNANSWERABLE'}
    for record in recorded['records'].values():
        assert set(record['retrieved_ids']) <= known_ids
        assert len(record['retrieved_ids']) == len(record['retrieval_similarities'])
        assert record['direct_response'] and record['rag_response']


def test_recorded_rag_answers_cite_only_retrieved_chunks():
    recorded = load_recorded_run(RECORDED_DIR / 'rag_recorded_run_2026-09-19.json')
    for qid, record in recorded['records'].items():
        check = validate_citations(record['rag_response'], set(record['retrieved_ids']))
        assert check['all_citations_retrieved'], f'{qid} cites {check["unsupported_ids"]}'


def test_recorded_run_stores_no_credential():
    raw = (RECORDED_DIR / 'rag_recorded_run_2026-09-19.json').read_text()
    assert 'sk-' not in raw and 'OPENAI_API_KEY' not in raw


def test_unanswerable_case_abstained_in_the_recorded_run():
    recorded = load_recorded_run(RECORDED_DIR / 'rag_recorded_run_2026-09-19.json')
    assert 'does not contain' in recorded['records']['Q3_UNANSWERABLE']['rag_response'].lower()


# ---------------------------------------------------- citation audit fixture

def test_audit_fixture_is_labelled_as_instructor_authored():
    audit = json.loads((RECORDED_DIR / 'citation_audit_exercise.json').read_text())
    assert 'NOT OpenAI outputs' in audit['_README']
    assert len(audit['cases']) == 5


def test_audit_fixture_verdicts_match_the_automated_checker():
    """The exercise only teaches if the automated check really does miss the subtle faults."""
    audit = json.loads((RECORDED_DIR / 'citation_audit_exercise.json').read_text())
    for case in audit['cases']:
        check = validate_citations(case['answer'], set(case['retrieved_ids']))
        caught = not check['all_citations_retrieved']
        assert caught == case['automated_check_catches_it'], case['case_id']


def test_only_one_audit_fault_is_machine_detectable():
    audit = json.loads((RECORDED_DIR / 'citation_audit_exercise.json').read_text())
    real_faults = [c for c in audit['cases'] if c['fault'] != 'none']
    assert len(real_faults) == 4
    assert sum(c['automated_check_catches_it'] for c in real_faults) == 1
