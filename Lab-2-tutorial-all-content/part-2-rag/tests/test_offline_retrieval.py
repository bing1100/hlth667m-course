from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tutorial_utils import load_manifest,chunk_markdown_by_h2,token_overlap_scores,format_context,validate_citations

def chunks():
 return [c for r in load_manifest(ROOT/'data/rag_corpus') for c in chunk_markdown_by_h2(ROOT/'data/rag_corpus'/r['filename'],r)]

def test_q1_expected_chunk_is_top_three_and_order_is_stable():
 c=chunks(); ranked=token_overlap_scores('How long does a portal registration code remain valid?',c)
 assert 'access_guide::000' in [x[0] for x in ranked[:3]]
 assert ranked==token_overlap_scores('How long does a portal registration code remain valid?',c)

def test_context_and_citation_validation():
 c=chunks(); selected=[x for x in c if x['chunk_id']=='access_guide::000']
 text=format_context(selected)
 assert '[access_guide::000]' in text and '48 hours' in text
 check=validate_citations('It lasts 48 hours [access_guide::000] [bad_source::999]',{'access_guide::000'})
 assert check['unsupported_ids']==['bad_source::999'] and not check['all_citations_retrieved']
