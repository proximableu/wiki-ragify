# Plan: Add "Generate Prompt" to the CONFIG modal

Status: **planned, not yet started** (as of 2026-10-05). The user has been
consulted; no implementation has begun. Write this file so the plan survives
session compaction.

## Goal

Add an OPTIONAL "Generate Prompt" section to the CONFIG modal (`ConfigScreen` in
`wiki_ragify/ui/app.py`):

- two text fields: **Target topic** and **Explicitly exclude**
- a **Generate** button
- On press: run the existing gating-prompt generator, then write the result into
  the **Gating prompt path** field (the field already in the modal), **overwriting**
  it if it exists, and **notify the user** of success/failure.

## Known facts / API (verified via `tests/test_prompt_gen.py` + source)

- Module: `wiki_ragify/gating/prompt_gen.py` (NOT in `__init__.py` `__all__`, import it explicitly).
- Entry point:
  ```python
  def generate_gating_prompt(template_text, topic, exclude="", *, config,
                             evaluator=None, retries=3, num_ctx=None) -> str:
  ```
  Returns the **generated prompt text** (str), or `""` on total failure.
- Template constant: `prompt_gen.DEFAULT_TEMPLATE`, sourced from
  `wiki_ragify/gating/make_a_topic_gate.txt`. Placeholders are exactly
  `{{target_topic | textarea}}` and `{{exclude | textarea}}`.
- `save_prompt(prompt, topic, exclude, *, output_dir) -> Path` writes to
  `<output_dir>/prompts/<slug>.txt` + a `.manifest.json`. **NOT used here** — it
  writes to a slug-named `prompts/` subdir, but the requirement is to write to the
  path already in the Gating prompt field. We write the generated text to that path
  ourselves.
- `ConfigScreen.__init__(start_page="", prompt_path="", output_root="")` currently
  does NOT receive the live `Config`. We must add `config=self.config` so
  generation uses the same Ollama/LLM settings already loaded from `.env`.

## Design decisions to confirm with user (ask before implementing)

1. **Template source:** use `prompt_gen.DEFAULT_TEMPLATE`, or let the user type a
   custom template too? (Recommend: use `DEFAULT_TEMPLATE` for now.)
2. **Generate on a worker thread** (it's network I/O to the remote gate model). ✅
   Recommend yes — `self.app.run_worker(..., thread=True)`.
3. **Save target path** = whatever is already in the Gating prompt path field
   (`Path(#prompt-path).expanduser()`). If that field is empty, save into
   `output-dir/prompts/<topic-slug>.txt` (the natural location) — or require the
   field be set first? (Recommend: if prompt-path field is empty, warn that a
   destination must be given, OR default to `<output-dir>/prompts/<topic>.txt` and
   prefill the field. Ask.)

## Implementation steps

### 1. UI additions in `ConfigScreen.compose()`
Insert, above the Confirm/Cancel `#config-actions` row, a new container:
- Input `#target-topic` — label "Target topic"
- Input `#exclude-topic` — label "Explicitly exclude"
- Button `#generate-prompt` — label "Generate"
Reuse the existing `#output-warning` Static (or a dedicated `#result-warning`) as
the notification area. Add matching CSS rules (margins/spacing) in
`wiki_ragify/ui/styles.tcss`.

### 2. Pass the live config into the modal
In `app.py` `on_button_pressed` `config-button` branch, build the screen with the
live config:
```python
ConfigScreen(
    start_page=self._config_start_page or "",
    prompt_path=self._config_prompt_path or "",
    output_root=str(self.root),
    config=self.config,   # NEW
)
```
Add `config: Config` to `ConfigScreen.__init__`, store `self._config`.

### 3. Generate button handler (`_on_generate_pressed` in ConfigScreen)
- Validate `#target-topic` is non-empty (else inline warning, no-op).
- Read topic / exclude from the two fields.
- Dispatch a **worker thread**:
  ```python
  self.app.run_worker(self._generate_in_worker, thread=True)
  ```
  where the worker calls `prompt_gen.generate_gating_prompt(DEFAULT_TEMPLATE, topic, exclude, config=self._config)`,
  then writes the text to the chosen destination path (overwriting), catching
  exceptions and posting a result message back to the main thread.
- The main-thread result handler updates the notification text:
  - success → `"✓ Prompt saved to <path>"` (green)
  - failure → `"✗ Generation failed: <error>"` (red), leaving prompt-path untouched.

### 4. Keep existing validation intact
`_submit()` still requires the prompt-path field to point at an existing file, so a
fresh Generate → Confirm flow works end to end. The inline notification uses the
same warning Static, cleared on the next Confirm attempt.

### 5. Tests (`tests/test_config_generate_prompt.py`)
One async test that, given a monkeypatched `prompt_gen.generate_gating_prompt`
(returning a fixed string), drives the Generate button, awaits the worker, and
asserts:
- the destination file was created with the generated text (overwriting a prior file),
- the notification Static shows the success text.
Also test the empty-topic guard (no worker fired, warning shown).
Drive `run_test` inside `anyio.run` via a zero-arg coroutine factory (no async
pytest plugin is installed — reuse the pattern from
`tests/test_app_event_fanout.py`).

## Files touched
- `wiki_ragify/ui/app.py` — `ConfigScreen` (fields, handler, config injection) + `on_button_pressed`.
- `wiki_ragify/ui/styles.tcss` — new field/button spacing.
- `tests/test_config_generate_prompt.py` — new test file.

## Not done (deferred / explicitly out of scope)
- No new library code; generation already exists in `prompt_gen`.
- No CLI changes.
- No changes to `save_prompt`/manifest behavior.

## Open question to raise with the user (before implementing)
- Where should the generated prompt be saved when the "Gating prompt path" field is
  empty: require it be set first, or default to `<output-dir>/prompts/<topic-slug>.txt`
  and prefill the field? (This is the one decision that blocks the implementation.)
