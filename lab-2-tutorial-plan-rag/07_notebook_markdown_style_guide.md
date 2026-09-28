# 7. Notebook Markdown Style Guide

Use the existing course plain-language standard at `/home/bhux/research/proposals/hlth667m-course/current_course_materials/plain_language_standard.md`.

## Before every code cell

State:

1. what the cell does;
2. the new term, in plain language;
3. why the step is needed;
4. what students should inspect;
5. the official documentation link when an API is first introduced.

## After important outputs

Add a Markdown cell titled **“What do we see?”**. It must identify the pattern, explain what it means for the next step, state what it cannot establish, and avoid claims that an AI system understands or decides.

## Required notebook navigation

Each notebook begins with title, purpose, date, runtime, objectives, prerequisites, roadmap, safety/data notice, and “run cells from top to bottom.” Each ends with result summary, limitations, completion checklist, links to supporting files, and optional next steps.

## Code rules

- Prefer one conceptual operation per cell.
- Keep most cells under 20 lines; plotting or explicit math cells may be longer.
- Use names such as `attention_weights`, `selected_words`, `retrieved_chunks`, and `direct_response`.
- Keep core attention math visible; do not hide it in a framework.
- Display compact tables rather than full vectors or full document stores.
- Never print keys, prompts containing secrets, or unrestricted environment dictionaries.
- Label fallback output as fallback.

## Required technical wording

Use “embedding vector,” “attention weight,” “retrieved chunk,” and “generated response.” Say “the response is conditioned on” rather than “the model knows.” Say “the heat map displays computed weights” rather than “the model focuses” unless the computation is defined immediately.

## Accessibility

Every plot needs a descriptive title, labelled axes, a colorbar where color encodes a quantity, and a text interpretation. Use colorblind-safe palettes and do not encode the only distinction with color.
