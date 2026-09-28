#!/usr/bin/env python3
"""Print every slide in deck order as 'page<TAB>title'. Used to keep speaker-script.md in step with the deck.

    python list_slides.py            # list the slides
    python list_slides.py --check    # fail if speaker-script.md headings differ from the deck
"""
import re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
PATTERN = re.compile(r'\\begin\{frame\}(?:\[[^\]]*\])?\{(?P<frame>.*)\}\s*$'
                     r'|\\(?:tutorialslide|notebookslide)\{(?P<tag>.*?)\}\{(?P<divider>.*?)\}\{'
                     r'|\\nbexample(?:\[[^\]]*\])?\{.*?\}\{(?P<example>.*?)\}\{', re.M)

def plain(text):
    text = re.sub(r'\\,\\textperiodcentered\\,', '·', text)
    text = text.replace(r'R\textsuperscript{2}', 'R²').replace(r'$\sqrt{d_k}$', '√d_k').replace('---', '—')
    return re.sub(r'\s+', ' ', text).strip()

def slides():
    out = ['Title']
    for name in ['part0-opening.tex', 'part1-ml.tex', 'part2-genai.tex']:
        for m in PATTERN.finditer((HERE/name).read_text(encoding='utf-8')):
            if m.group('frame'): out.append(plain(m.group('frame')))
            elif m.group('divider'): out.append(plain(f"{m.group('tag')} — {m.group('divider')}"))
            else: out.append(plain(m.group('example')))
    return out

if __name__ == '__main__':
    deck = slides()
    if '--check' in sys.argv:
        script = re.findall(r'^### (\d+) · (.+)$', (HERE/'speaker-script.md').read_text(encoding='utf-8'), re.M)
        found = [(int(n), t.strip()) for n, t in script]
        expected = list(enumerate(deck, start=1))
        if found != expected:
            for e, f in zip(expected, found + [(None, None)] * len(expected)):
                if e != f: print(f'first difference: deck has {e}, script has {f}'); break
            print(f'deck: {len(expected)} slides, script: {len(found)} entries'); raise SystemExit(1)
        print(f'speaker-script.md covers all {len(deck)} slides in order')
    else:
        for n, title in enumerate(deck, start=1): print(f'{n}\t{title}')
