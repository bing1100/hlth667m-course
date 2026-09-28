# Troubleshooting — Tutorial 2, Parts 0 to 3

## Setup

| Symptom | Check and fix |
|---|---|
| `ModuleNotFoundError: gensim`, `chromadb`, `torch`, `tiktoken`, `transformers` | Activate the intended environment and run `python -m pip install -r requirements.txt`. Restart the kernel. |
| `ModuleNotFoundError: streamlit` | `requirements.txt` now lists `streamlit>=1.37`. Run `pip install -r requirements.txt` in the environment the kernel uses, then restart the kernel. |
| A link or note mentions Part 4, Part 5, or Part 6 | Those are the former numbers. Tokenization is Part 0, Word2Vec is Part 1, attention is Part 2, and RAG is Part 3. The former Part 5 decoding notebook is in `archive/`. |
| Gensim/SciPy import error | Use the tested package set in `requirements.txt`. Record `python -m pip freeze` in `VALIDATION.md` rather than patching notebook imports. |
| `FileNotFoundError` for `data/rag_corpus` | The notebooks search upward from the working directory. If Jupyter started somewhere unrelated, set `TUTORIAL_ROOT` to the `part-2-rag` folder. |
| Kernel restarted or disconnected | Reconnect and run every cell from the top. Variables are not retained. |
| Cells were run out of order | Restart the kernel and use **Run All**. Do not create missing variables by hand. |

## Part 0 — Tokenization

| Symptom | Check and fix |
|---|---|
| `Production tokenizer unavailable in this environment` | Expected when `tiktoken` cannot download its vocabulary file. Sections 6–8 skip themselves; the hand-built BPE sections still run and carry the main teaching points. |
| Merge table looks different after editing the corpus | Expected. The merges are learned from your text. Note what changed rather than reverting. |
| A common word still splits into many pieces | Raise `NUM_MERGES`, or check the word's frequency. Frequency, not mere presence, decides what becomes one token. |

## Part 1 — Word2Vec and t-SNE

| Symptom | Check and fix |
|---|---|
| t-SNE perplexity error | Keep `perplexity=5` with the fixed 18 selected words. Perplexity must be below the sample count. |
| Different Word2Vec neighbours between runs | Confirm the fixed corpus, `seed=42`, `workers=1`, package versions, and CPU execution. Do not rerun until you like the result. |
| The t-SNE map looks different from the slide | Expected across platforms. Judge pair similarity from the heat map, not from apparent distance on the map. |

## Part 2 — Attention, masks, and training objectives

| Symptom | Check and fix |
|---|---|
| Shape or dtype error | Print tensor shapes. Q and K must share their last dimension; K and V must share their sequence length. Use the `float64` fixture tensors. |
| Mask produces `NaN` | Check the convention: `True = allowed`. A query row with every key masked has nothing to softmax over, and the helper raises rather than returning `NaN`. |
| `assert_close` fails after editing the function | Compare your intermediate matrices with the printed fixture values step by step. For query `clinic` the scaled scores are `[0.707, 0.000]`, the weights `[0.670, 0.330]`, and the output `[6.70, 6.60]`. |
| The first run of section 10 is slow or appears to hang | Expected once. `transformers` downloads `distilgpt2` and `distilbert-base-uncased`, about 600 MB in total, and caches them. Later runs load from the cache. |
| `Pretrained models unavailable (...)`, then `PRETRAINED is None — nothing to show.` | Expected when the machine is offline, the download is blocked, or `transformers` is missing. Section 10 skips itself and sections 1–9 are complete without it. The toy results in section 8 show the same contrast. To enable the section, install from `requirements.txt` and run it once on a network that allows the download. |
| The AR loss stops near 0.33 and will not reach zero | Correct. After `reports`, four symptoms are equally likely, so the best possible model still spreads its probability across them. |
| The leak test shows a non-zero change for the AR model | The model was trained or evaluated with the wrong mask. Confirm `CAUSAL[:-1, :-1]` is passed inside `train`. A non-zero value means each position has been reading its own answer, and the training loss is meaningless. |
| The MLM loss curve is noisy | Expected. A different random set of positions is masked at every step. |
| Predictions differ slightly from the text | Confirm CPU execution, the fixed seeds in section 0, and the package versions in `VALIDATION.md`. Run from the top. |
| Looking for the softmax zoom, the positional-encoding shuffle test, or the separate PyTorch comparison | Those sections were removed. The PyTorch check is the `assert_close` line in section 3. Order-blindness and positional encoding are on the "beyond the notebook" slide, and the earlier notebook is in `archive/Attention_full_walkthrough_v1.ipynb`. |

## Decoding (slides only)

Decoding has no notebook in the taught sequence. The former notebook is at `archive/Decoding_Temperature_TopK_TopP.ipynb`, and these entries apply to it.

| Symptom | Check and fix |
|---|---|
| Probabilities do not sum to 1 | After top-k or top-p truncation you must renormalise by dividing by the new sum. |
| `top_k=1` output changes with temperature | It should not. Check that truncation happens after temperature and that one surviving token gets probability 1. |
| Generated text stops after a few words | Expected. The bigram model ends when it reaches a word nothing followed in the corpus. |
| Every run gives identical text | Check the temperature. At or near 0 the sampler is not used at all. |

## Part 3 — RAG

| Symptom | Check and fix |
|---|---|
| Notebook says `run_mode: replay` and you wanted live | Set `HLTH667M_RUN_MODE=live` **in your shell**, not in a cell, and confirm `OPENAI_API_KEY` is exported. Restart the kernel. |
| `Live mode needs OPENAI_API_KEY` | Keep replay mode, or export the key outside the notebook. Never print or paste a key into a cell. |
| Authentication or rate-limit failure in live mode | Switch back to `replay` and continue the class. Check account and project configuration after class. |
| Q4 shows no response | Correct behaviour. Q4 was added after the recorded run, and the notebook does not fabricate a response. Run live to see one, or use the section 13 audit. |
| A replayed answer is mistaken for a fresh one | Every replayed card carries a `RECORDED` badge naming the run date. If you do not see that badge, you are in live mode. |
| Chroma index appears stale | Compare the fingerprint, embedding model, chunk algorithm, and expected count (9). Rebuild only the named collection, using `RESET_VECTOR_STORE = True`. |
| Duplicate chunk IDs | Rebuild from the three manifest-verified files. Never append blindly. |
| Changing `words_per_chunk` changes the results | Expected — that is the exercise. Record what changed in retrieval rather than searching for the "best" number. |
| Terminology expansion matched nothing | The matcher uses plain case-insensitive substring matching. Add the exact surface form to `data/terminology/health_terminology.json` and rerun. |
| Graph walk returns unrelated chunks | Expected at two hops. Reduce `hops` to 1 and compare. Noise propagation is the point of the comparison. |
| A fluent RAG answer is unsupported | Inspect the retrieved IDs and text. Mark the unsupported claim; do not edit the output. Section 13 practises exactly this. |

## Part 3 section 14 — Streamlit chat app

| Symptom | Check and fix |
|---|---|
| `ModuleNotFoundError: streamlit` | Run `pip install -r requirements.txt` in the environment the kernel uses, then restart the kernel. |
| Port already in use | The launch cell tries ports 8501 to 8520 and takes the first free one, so read the printed address instead of assuming 8501. `No free port found.` means all twenty are taken, usually by earlier app processes; stop those first (next row). |
| The app is still running after the notebook closes | The app is a separate process. While the notebook is open, set `STOP_CHAT_APP = True` and run the stop cell. If the notebook is already closed, restart or shut down its kernel, which terminates the app. If the address still responds, end the `streamlit` process from your system's task manager, or run `pkill -f "streamlit run rag_chat_app.py"` on macOS or Linux. |
| `The app is already running at ...` | The launch cell found the process it started earlier. Open that address, or run the stop cell and launch again. |
| `Streamlit did not start.` | Run `streamlit run rag_chat_app.py --server.address localhost` in a terminal from `part-2-rag/` to see the error. A missing `rag_chat_app.py` means the `%%writefile` cell has not been run. |
| `generate` mode is greyed out | No key was found. The sidebar shows `No OPENAI_API_KEY found, so the app runs in retrieval-only mode.` Put `OPENAI_API_KEY=...` in a `.env` file beside the notebook, stop the app, and launch it again. The key is read at start-up. Never type a key into a cell or into the app. |
| The reply says `No model was called.` | Correct for `retrieval only` mode. Open the evidence panel and read the passages. |
| The app names a "best-matching passage" for a question the corpus cannot answer | Correct, and it is the point of the section's exercise. Top-k always returns k passages. Change `answer()` so that it declines when the top score is below a threshold. |
| The chat reply says `The request failed: ...` | A paid call failed in `generate` mode. Switch the sidebar back to `retrieval only` and check the account after class. The app never shows the key. |
| Edits to the app do not appear | Edit the `%%writefile` cell and rerun it, then refresh the browser tab. A test compares that cell with `rag_chat_app.py`, so keep the two identical. |
| Launch cell prints `Launch skipped because HLTH667M_SKIP_APP_LAUNCH=1.` | That environment variable is set. Unset it and restart the kernel to launch the app. |
| A warning about running outside a server appears in the `AppTest` cell | Harmless. `AppTest` runs the script in memory, and the cell silences the warning while it runs. |
| Someone wants to open the app from another computer | Keep it on `localhost`. Exposing the app is a deployment decision with privacy, security, and governance consequences, and it is outside this tutorial. |

## First checks, in order

1. Is the correct kernel selected?
2. Did you run from the first cell?
3. For Part 3: does the printed `run_mode` match what you intended, and is the chat app stopped when you are done?
4. Does `python -m pytest -q` pass from the `part-2-rag` folder?
5. Are you about to send anything that is not fictional teaching text? Stop if so.
