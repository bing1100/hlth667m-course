"""Insert (or refresh) the Streamlit chat section in the RAG notebook. Idempotent."""
import sys, nbformat as nbf
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
NB = HERE/'Part-3_RAG_With_and_Without_Retrieval.ipynb'
APP = (HERE/'rag_chat_app.py').read_text()
TAG = 'streamlit-section'
md = lambda s: nbf.v4.new_markdown_cell(s.strip('\n'), metadata={'tags': [TAG]})
code = lambda s: nbf.v4.new_code_cell(s.strip('\n'), metadata={'tags': [TAG]})

new = [
md(r"""
## 14. Build a chat interface with Streamlit

Everything so far ran inside notebook cells. People who use a RAG system never see cells: they see a chat box. [Streamlit](https://docs.streamlit.io) turns a plain Python script into a web page, which makes it a quick way to put a working prototype in front of a colleague.

The app below reuses the functions you have already studied: `chunk_markdown_by_h2`, `expand_query_with_terminology`, `token_overlap_scores`, `format_context` and `validate_citations`. It has two modes.

| Mode | Retrieval | Answer | Needs a key? | Costs money? |
|---|---|---|---|---|
| `retrieval only` (default) | word overlap, with optional terminology expansion | none: the app shows the passages a model would receive | No | No |
| `generate` | embedding similarity | grounded answer with chunk citations | Yes | Yes |

Three Streamlit ideas are enough to read the script:

- **The script reruns from top to bottom on every interaction.** Anything that must survive a rerun, such as the conversation, lives in `st.session_state`.
- **`@st.cache_data` keeps expensive results** (loading the corpus, embedding the chunks) so a rerun does not repeat them.
- **`st.chat_input` and `st.chat_message`** draw the chat box and the message bubbles.

The cell uses the `%%writefile` magic: running it saves the cell body as `rag_chat_app.py` next to this notebook. Edit the cell and rerun it to change the app.

**Look for:** where `retrieve` and `generate` are called, and how little code is specific to the interface.
"""),
code('%%writefile rag_chat_app.py\n' + APP),
md(r"""
### Test the app without opening a browser

Streamlit ships a test harness, `AppTest`, that runs the script in memory, types into the chat box, and returns what the page would show. It is a fast way to check an app, and it is how this notebook can show you the app's reply as ordinary cell output.

**Look for:** the mode note under the reply, and whether the ranking matches section 8, where the same question was expanded with the terminology.
"""),
code(r"""
from streamlit.testing.v1 import AppTest
import logging

logging.disable(logging.WARNING)                      # AppTest runs outside a server and warns about it
app_test = AppTest.from_file('rag_chat_app.py', default_timeout=60).run()
app_test.chat_input[0].set_value(PATIENT_QUESTION).run()
logging.disable(logging.NOTSET)
assert not app_test.exception, app_test.exception

reply = app_test.session_state['history'][-1]
print('QUESTION :', PATIENT_QUESTION)
print('MODE     :', reply['note'])
print('REPLY    :', reply['text'])
print('QUERY    :', reply['query'])
display(pd.DataFrame([{'rank': c['rank'], 'chunk_id': c['chunk_id'], 'section': c['section'], 'score': round(c['score'], 3)}
                      for c in reply['retrieved']]))
"""),
md(r"""
### Launch the app

This cell starts Streamlit as a separate process on your own computer and prints the address. Open it in a browser tab. The notebook stays usable while the app runs.

The server is bound to `localhost`, so only your machine can reach it. That is deliberate. Making an app reachable by other people is a deployment decision with privacy, security and governance consequences, and it is outside this tutorial.

To use `generate` mode, put `OPENAI_API_KEY=...` in a `.env` file beside this notebook **before** launching. Never type a key into a notebook cell or into the app.

**Look for:** the printed address. Then ask the app Q1 to Q4 from section 4 and compare with the recorded responses.
"""),
code(r"""
import subprocess, sys, time, socket, atexit, urllib.request

def first_free_port(start=8501, tries=20):
    for port in range(start, start + tries):
        with socket.socket() as probe:
            if probe.connect_ex(('localhost', port)) != 0:
                return port
    raise RuntimeError('No free port found.')

if os.getenv('HLTH667M_SKIP_APP_LAUNCH') == '1':
    print('Launch skipped because HLTH667M_SKIP_APP_LAUNCH=1.')
elif 'chat_app' in globals() and chat_app.poll() is None:
    print(f'The app is already running at {CHAT_APP_URL}')
else:
    port = first_free_port()
    chat_app = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', 'rag_chat_app.py',
                                 '--server.address', 'localhost', '--server.port', str(port),
                                 '--server.headless', 'true', '--browser.gatherUsageStats', 'false'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    atexit.register(chat_app.terminate)                 # stop the app when the notebook kernel shuts down
    CHAT_APP_URL = f'http://localhost:{port}'
    for _ in range(60):                                 # wait up to 30 seconds for the server to answer
        try:
            if urllib.request.urlopen(f'{CHAT_APP_URL}/_stcore/health', timeout=1).status == 200: break
        except OSError:
            time.sleep(.5)
    else:
        raise RuntimeError('Streamlit did not start. Run "streamlit run rag_chat_app.py" in a terminal to see the error.')
    print(f'Chat app running at {CHAT_APP_URL}  (key present: {bool(os.getenv('OPENAI_API_KEY'))}; the key value is never shown)')
"""),
md(r"""
### Stop the app

Before you close the notebook, set `STOP_CHAT_APP = True` and run this cell. The app is a separate process, and it can keep running after the notebook is closed. If the address still responds later, restart the kernel or end the `streamlit` process from your system's task manager.
"""),
code(r"""
STOP_CHAT_APP = False
if STOP_CHAT_APP and 'chat_app' in globals() and chat_app.poll() is None:
    chat_app.terminate(); chat_app.wait(timeout=10)
    print('Chat app stopped.')
else:
    print('Chat app left as it is.' if 'chat_app' in globals() else 'No chat app was launched from this notebook.')
"""),
md(r"""
### What do we see?

In retrieval-only mode the app returns the same ranking as section 8, because it calls the same functions. The interface added no intelligence. It added a way for someone else to use what you built.

That is also the risk. A chat box looks finished and authoritative, and nothing on the screen tells a user that the corpus has nine chunks, that top-k always returns k passages even for an unanswerable question (section 7), or that a cited answer can still be wrong (section 13). The evidence panel under each reply is there so that a reader can do the check from section 13.

**Try it:** ask the app Q3, the unanswerable question. In retrieval-only mode it still names a "best-matching passage". Decide what the app should say instead, and change `answer()` so that it declines when the top score is below a threshold you choose.
"""),
]

nb = nbf.read(NB, as_version=4)
nb.cells = [c for c in nb.cells if TAG not in c.metadata.get('tags', [])]
at = next(i for i, c in enumerate(nb.cells) if c.cell_type == 'markdown' and 'Learner checks' in c.source.splitlines()[0])
nb.cells[at].source = nb.cells[at].source.replace('## 14. Learner checks', '## 15. Learner checks')
nb.cells[at:at] = new
nbf.write(nb, NB); print('inserted', len(new), 'cells at', at)
