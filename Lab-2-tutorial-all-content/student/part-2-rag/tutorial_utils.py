"""Small, inspectable helpers for the HLTH 667M representations/attention/RAG tutorial."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
from typing import Any
STOPWORDS={"a","an","and","are","does","how","is","of","the","to","what","when","who"}

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def load_manifest(corpus_dir: Path) -> list[dict[str,Any]]:
    path=corpus_dir/'manifest.jsonl'
    records=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    required={'source_id','filename','title','fictional','sha256'}
    if any(not required <= set(r) for r in records): raise ValueError('Manifest record is missing required fields.')
    if any(not r['fictional'] for r in records): raise ValueError('All tutorial corpus records must be fictional.')
    return sorted(records,key=lambda r:r['filename'])

def chunk_markdown_by_h2(path: Path, source_metadata: dict[str,Any]) -> list[dict[str,Any]]:
    raw=path.read_text(encoding='utf-8')
    if sha256_bytes(raw.encode()) != source_metadata['sha256']: raise ValueError(f'Hash mismatch for {path.name}')
    title=source_metadata['title']; parts=re.split(r'^##\s+(.+?)\s*$',raw,flags=re.M)
    if len(parts)<3: raise ValueError(f'{path.name} has no level-2 sections.')
    chunks=[]
    for ordinal in range((len(parts)-1)//2):
        section=parts[2*ordinal+1].strip(); text=parts[2*ordinal+2].strip()
        if not text: raise ValueError(f'Empty section {section!r}.')
        combined=f'## {section}\n{text}'
        chunks.append({'chunk_id':f"{source_metadata['source_id']}::{ordinal:03d}",'source_id':source_metadata['source_id'],
          'source_filename':path.name,'title':title,'section':section,'ordinal':ordinal,'text':combined,
          'text_sha256':sha256_bytes(combined.encode())})
    return chunks

def corpus_fingerprint(chunks:list[dict[str,Any]],embedding_model:str,chunk_algorithm:str='h2-v1')->str:
    ids=[c['chunk_id'] for c in chunks]
    if len(ids)!=len(set(ids)): raise ValueError('Duplicate chunk IDs.')
    payload={'embedding_model':embedding_model,'chunk_algorithm':chunk_algorithm,'chunks':[{k:c[k] for k in ('chunk_id','text_sha256')} for c in sorted(chunks,key=lambda c:c['chunk_id'])]}
    return sha256_bytes(json.dumps(payload,sort_keys=True,separators=(',',':')).encode())

def _terms(text:str)->set[str]: return set(re.findall(r'[a-z0-9]+',text.lower()))-STOPWORDS

def token_overlap_scores(query:str,chunks:list[dict[str,Any]])->list[tuple[str,float]]:
    q=_terms(query); out=[]
    for c in chunks:
        t=_terms(c['text']); score=len(q&t)/len(q|t) if q|t else 0.0;out.append((c['chunk_id'],score))
    return sorted(out,key=lambda x:(-x[1],x[0]))

def format_context(retrieved:list[dict[str,Any]])->str:
    return '\n\n'.join(f"[{c['chunk_id']}]\n{c['text']}" for c in retrieved)

def validate_citations(answer:str,retrieved_ids:set[str])->dict[str,Any]:
    cited=set(re.findall(r'\[([a-z_]+::\d{3})\]',answer))
    return {'cited_ids':sorted(cited),'unsupported_ids':sorted(cited-retrieved_ids),'all_citations_retrieved':cited<=retrieved_ids}

def resolve_tutorial_root() -> Path:
    """Find the folder holding ``data/rag_corpus``.

    Checks ``TUTORIAL_ROOT`` first, then the working directory and each of its
    parents, then a sibling ``part-2-rag`` folder beside any of them. Walking up
    means the notebooks run from the tutorial folder, from a ``student/`` copy,
    or from a parent directory without any path editing.
    """
    import os
    def looks_right(path: Path) -> bool:
        return (path/'data'/'rag_corpus').is_dir()

    override = os.getenv('TUTORIAL_ROOT')
    if override:
        candidate = Path(override).expanduser()
        if looks_right(candidate):
            return candidate.resolve()
        raise FileNotFoundError(f'TUTORIAL_ROOT={override} has no data/rag_corpus folder.')

    here = Path.cwd().resolve()
    for directory in [here, *here.parents]:
        if looks_right(directory):
            return directory
        sibling = directory/'part-2-rag'
        if looks_right(sibling):
            return sibling.resolve()
    raise FileNotFoundError(
        'Could not find data/rag_corpus above the working directory. '
        'Set TUTORIAL_ROOT to the part-2-rag folder.')


# --------------------------------------------------------------------------
# Second chunking strategy (Part 6 chunk-size comparison)
# --------------------------------------------------------------------------

def chunk_markdown_fixed_window(path: Path, source_metadata: dict[str, Any],
                                words_per_chunk: int = 40, overlap_words: int = 10) -> list[dict[str, Any]]:
    """Split a source into fixed-length word windows instead of at ``##`` headings.

    Heading chunking respects the author's topic boundaries. Fixed windows ignore
    them: a window can start mid-sentence and can mix two topics. Comparing the two
    is the point of the exercise - neither is universally correct.
    """
    if overlap_words >= words_per_chunk:
        raise ValueError('overlap_words must be smaller than words_per_chunk.')
    raw = path.read_text(encoding='utf-8')
    if sha256_bytes(raw.encode()) != source_metadata['sha256']:
        raise ValueError(f'Hash mismatch for {path.name}')
    body = '\n'.join(line for line in raw.splitlines()
                     if not line.startswith('# ') and not line.startswith('> '))
    words = body.split()
    step = words_per_chunk - overlap_words
    chunks: list[dict[str, Any]] = []
    for ordinal, start in enumerate(range(0, max(len(words), 1), step)):
        window = words[start:start + words_per_chunk]
        if len(window) < max(5, overlap_words + 1) and chunks:
            break
        text = ' '.join(window)
        chunks.append({
            'chunk_id': f"{source_metadata['source_id']}#w{ordinal:03d}",
            'source_id': source_metadata['source_id'], 'source_filename': path.name,
            'title': source_metadata['title'], 'section': f'words {start}-{start + len(window)}',
            'ordinal': ordinal, 'text': text, 'text_sha256': sha256_bytes(text.encode()),
        })
    return chunks


# --------------------------------------------------------------------------
# Terminology and knowledge graph (Part 6 structured-knowledge section)
# --------------------------------------------------------------------------

def load_terminology(path: Path) -> dict[str, Any]:
    """Load the fictional NORTHSTAR-CT concept/synonym file."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    for concept in data['concepts']:
        if not {'code', 'preferred_term', 'synonyms'} <= set(concept):
            raise ValueError(f"Concept {concept.get('code')} is missing required fields.")
    return data


def match_concepts(query: str, terminology: dict[str, Any]) -> list[dict[str, Any]]:
    """Return concepts whose preferred term or any synonym appears in the query.

    Matching is plain case-insensitive substring matching so students can read and
    predict it. A production terminology service would do far more (morphology,
    negation, context, post-coordination) and would still make mistakes.
    """
    lowered = query.lower()
    matched = []
    for concept in terminology['concepts']:
        surface_forms = [concept['preferred_term']] + list(concept['synonyms'])
        hits = sorted({form for form in surface_forms if form.lower() in lowered},
                      key=lambda f: (-len(f), f))
        if hits:
            matched.append({'code': concept['code'], 'preferred_term': concept['preferred_term'],
                            'matched_surface_forms': hits, 'corpus_terms': concept.get('corpus_terms', [])})
    return sorted(matched, key=lambda c: c['code'])


def expand_query_with_terminology(query: str, terminology: dict[str, Any]) -> dict[str, Any]:
    """Add each matched concept's preferred term and corpus vocabulary to the query."""
    matched = match_concepts(query, terminology)
    added: list[str] = []
    for concept in matched:
        for term in [concept['preferred_term']] + list(concept['corpus_terms']):
            covered = f"{query} {' '.join(added)}".lower()
            if term.lower() not in covered:
                added.append(term)
    return {'original_query': query, 'matched_concepts': matched, 'added_terms': added,
            'expanded_query': query if not added else f"{query} {' '.join(added)}"}


def load_knowledge_graph(path: Path) -> dict[str, Any]:
    """Load the fictional care knowledge graph and check that no edge dangles."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    node_ids = {node['id'] for node in data['nodes']}
    dangling = [e for e in data['edges'] if e['subject'] not in node_ids or e['object'] not in node_ids]
    if dangling:
        raise ValueError(f'{len(dangling)} edge(s) reference a node that does not exist.')
    if any('evidence' not in e for e in data['edges']):
        raise ValueError('Every edge must carry the chunk it was derived from.')
    return data


def graph_paths_from(graph: dict[str, Any], start_ids: list[str], hops: int = 2) -> list[dict[str, Any]]:
    """Walk outward from the starting nodes and return the edges reached, in order.

    Each returned edge keeps its ``evidence`` chunk id, so a graph answer can always
    be traced back to source text rather than asserted on the graph's authority.
    """
    labels = {node['id']: node['label'] for node in graph['nodes']}
    seen_edges: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    frontier = list(start_ids)
    visited = set(start_ids)
    for hop in range(1, hops + 1):
        next_frontier: list[str] = []
        for edge in graph['edges']:
            if edge['subject'] not in frontier:
                continue
            key = (edge['subject'], edge['predicate'], edge['object'])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            seen_edges.append({**edge, 'hop': hop,
                               'subject_label': labels.get(edge['subject'], edge['subject']),
                               'object_label': labels.get(edge['object'], edge['object'])})
            if edge['object'] not in visited:
                visited.add(edge['object'])
                next_frontier.append(edge['object'])
        frontier = next_frontier
        if not frontier:
            break
    return seen_edges


def graph_evidence_ids(edges: list[dict[str, Any]]) -> list[str]:
    """Collect the distinct corpus chunk ids that justify a set of graph edges."""
    return sorted({edge['evidence'] for edge in edges})


# --------------------------------------------------------------------------
# Recorded-run replay (Part 6 offline mode)
# --------------------------------------------------------------------------

def load_recorded_run(path: Path) -> dict[str, Any]:
    """Load the verbatim transcript of a previous paid run for offline replay."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    for qid, record in data['records'].items():
        if not {'direct_response', 'rag_response', 'retrieved_ids'} <= set(record):
            raise ValueError(f'Recorded record {qid} is incomplete.')
    return data


RECORDED_LABEL = 'RECORDED — replay of a real run on '
