from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from tutorial_utils import load_manifest,chunk_markdown_by_h2,corpus_fingerprint

def chunks():
 manifest=load_manifest(ROOT/'data/rag_corpus')
 return [c for r in manifest for c in chunk_markdown_by_h2(ROOT/'data/rag_corpus'/r['filename'],r)]

def test_manifest_hashes_and_nine_stable_chunks():
 records=load_manifest(ROOT/'data/rag_corpus'); actual=chunks()
 assert len(records)==3 and all(r['fictional'] for r in records)
 assert len(actual)==9
 assert {c['chunk_id'] for c in actual}=={
  'access_guide::000','access_guide::001','access_guide::002',
  'results_policy::000','results_policy::001','results_policy::002',
  'privacy_policy::000','privacy_policy::001','privacy_policy::002'}
 assert all(c['text'].startswith('## ') and c['text_sha256'] for c in actual)

def test_hash_mismatch_is_rejected(tmp_path):
 record=load_manifest(ROOT/'data/rag_corpus')[0]
 copied=tmp_path/record['filename']; copied.write_text('## Altered\ntext')
 with pytest.raises(ValueError,match='Hash mismatch'):
  chunk_markdown_by_h2(copied,record)

def test_fingerprint_changes_with_embedding_model_or_source_hash():
 actual=chunks(); first=corpus_fingerprint(actual,'text-embedding-3-small')
 assert first != corpus_fingerprint(actual,'another-model')
 changed=[dict(c) for c in actual]; changed[0]['text_sha256']='changed'
 assert first != corpus_fingerprint(changed,'text-embedding-3-small')
