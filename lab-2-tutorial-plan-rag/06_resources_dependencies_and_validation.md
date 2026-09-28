# 6. Resources, Dependencies, Security, and Validation Plan

## Official resources

### Word2Vec and visualization

- [Gensim Word2Vec documentation](https://radimrehurek.com/gensim/models/word2vec.html)
- [Gensim KeyedVectors / vector access](https://radimrehurek.com/gensim/models/keyedvectors.html)
- [scikit-learn TSNE API](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
- [scikit-learn manifold learning guide](https://scikit-learn.org/stable/modules/manifold.html)

### Attention

- [PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [PyTorch tensor semantics](https://docs.pytorch.org/docs/stable/tensor_attributes.html)
- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) for the original scaled dot-product and Transformer context.

### OpenAI and retrieval

- [OpenAI SDKs and environment-variable authentication](https://platform.openai.com/docs/libraries.md)
- [OpenAI embeddings guide](https://platform.openai.com/docs/guides/embeddings)
- [OpenAI retrieval guide](https://platform.openai.com/docs/guides/retrieval)
- [OpenAI embeddings API reference](https://platform.openai.com/docs/api-reference/embeddings)
- [OpenAI API safety best practices](https://platform.openai.com/docs/guides/safety-best-practices)
- [Chroma introduction](https://docs.trychroma.com/docs/overview/introduction)
- [Chroma persistent client](https://docs.trychroma.com/docs/run-chroma/persistent-client)
- [Chroma collections](https://docs.trychroma.com/docs/collections/manage-collections)

The plan links official documentation but does not copy large passages. Implementation must recheck model names and API syntax before delivery because hosted APIs and library versions change.

## Proposed `requirements.txt`

```text
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
seaborn>=0.12
gensim>=4.3
scikit-learn>=1.4
torch>=2.2
openai>=1.0
python-dotenv>=1.0
chromadb>=0.5
tiktoken>=0.7
jupyter>=1.0
nbformat>=5.9
nbclient>=0.9
pytest>=8.0
```

`tiktoken` is optional and must be moved to an optional requirements group or removed if the implementation does not use it. Pin exact versions after clean-environment implementation testing and record them in `VALIDATION.md`. Do not assume `gensim` or `chromadb` are installed merely because the current environment contains other ML packages. Verify the Gensim/SciPy combination explicitly because incompatible releases can fail at import time.

## Environment bootstrap contract

Target Python 3.11 for the tested classroom environment while retaining Python 3.10+ compatibility where the pinned packages support it. The final README must provide commands equivalent to:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
jupyter lab
```

Also document kernel registration if notebooks are opened from another Jupyter installation. Record operating system, Python version, package freeze, CPU/GPU mode, notebook runtimes, and test command in `VALIDATION.md`.

## Distribution decision

The student package contains source notebooks, `tutorial_utils.py`, fictional corpus and manifest, tests, documentation/slides, and instructor-generated static HTML/PDF notebook backups. It does **not** contain `.env`, API keys, live run logs, cached OpenAI responses, or the generated Chroma directory. Students can run Parts 1–2 and the deterministic lexical retrieval/prompt-construction path offline. Building the OpenAI embedding index and making generation calls are instructor-controlled demonstrations unless a separate approved student API budget is provided.

Do not distribute `embeddings.npz` by default: embeddings are model-derived artifacts that can become stale and obscure the index-build lesson. If an institution later approves a prebuilt cache, distribute it with exact model, dimensions, source hashes, license/provenance, creation date, and checksum, and label it as a snapshot rather than fresh API output.

## API-key and data-security policy

- Read only the variable name and presence status when checking setup; never print the secret.
- Load `/home/bhux/research/proposals/hlth-667-668/.env` only as an instructor-local setup option. Prefer an environment variable in hosted environments.
- Add `.env`, `*.key`, and secret-bearing files to `.gitignore`.
- Do not include real health data in source documents, prompts, embeddings, Chroma, logs, screenshots, or executed notebooks.
- Do not write the API key to notebook outputs, run logs, exception messages, or metadata.
- Set `LIVE_API_CALLS = False` as the safe default; require an explicit instructor change for live calls.
- Limit live calls to a small fixed question set and record model names and approximate usage/cost where available.

## Database and artifact policy

- Source documents are versioned only when fictional/public.
- `chunks.jsonl` is reproducible and contains no secrets.
- Chroma is local and persistent, but its directory is generated state and should normally be ignored by Git.
- Store a corpus SHA-256, chunking parameters, embedding model, distance metric, and creation time in `artifacts/rag/index_metadata.json`.
- Reuse an index only when these values match. Otherwise require a rebuild or explain the mismatch.
- Never commit a Chroma directory containing restricted data.
- Back up or delete local artifacts intentionally; persistence is not access control.

## Notebook execution validation

1. Parse all three `.ipynb` files with `nbformat`.
2. Verify every code cell is immediately preceded by Markdown.
3. Execute Word2Vec notebook from a fresh kernel with no network access.
4. Execute attention notebook from a fresh kernel and assert custom/official outputs agree.
5. Execute RAG notebook in offline mode with no key and verify chunking, fallback retrieval, prompts, and labelled fallback outputs.
6. Execute RAG notebook in live mode with an instructor-controlled key and a tiny fixed question set.
7. Verify the key is absent from notebook source, outputs, logs, and Git status.
8. Verify t-SNE receives a valid perplexity below the selected sample count and fixed seed.
9. Verify attention rows sum to one and causal future entries are zero.
10. Verify Chroma collection count, stable IDs, metadata, corpus fingerprint, and idempotent rerun.
11. Verify RAG answers display retrieved source IDs and distinguish direct from grounded output.
12. Verify unanswerable questions produce an insufficient-evidence path rather than an invented source claim.
13. Inspect plot titles, labels, colorbars, text explanations, and accessibility.
14. Run all notebooks twice in offline mode and confirm deterministic non-API outputs.
15. Record runtime and approximate live API calls before class.
16. Verify the generated corpus has 240 sentences and 44 vocabulary tokens under the documented tokenizer.
17. Verify the three source-file hashes match `manifest.jsonl` and chunking produces exactly nine stable IDs.
18. Verify Q1/Q2 expected-evidence retrieval checks and Q3 insufficient-evidence behavior.
19. Verify direct and RAG requests use the same generation model but condition-specific instructions.
20. Verify `slides.md` contains 24 slides, all required formulas, visible text, alt text, speaker notes, checks, expected answers, and notebook transitions.
21. Scan source, outputs, JSONL logs, generated HTML/images, and Git diff for credential patterns and accidental non-fictional records.

## Acceptance criteria

The implementation is complete only when all three notebooks run independently, every executable cell is explained, visual outputs are interpretable, API-key handling is safe, local storage is documented, offline mode is honest, live mode is tested, and no health claim exceeds the fictional evidence.
