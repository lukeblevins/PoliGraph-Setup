# AI Coding Agent Guide for PoliGraph

Concise, project-specific instructions to be productive quickly. Follow these patterns; do not introduce new frameworks without justification.

## Big Picture
PoliGraph builds structured knowledge graphs from privacy policy text. Core pipeline (used by scripts & Gradio app):
1. `html_crawler`: fetch & sanitize a policy webpage; emits `accessibility_tree.json`, `cleaned.html`, `readability.json` into a workdir.
2. `init_document`: parses accessibility tree into hierarchical segments (`PolicyDocument.initialize` + `SegmentExtractor`), runs spaCy NLP, saves `document.pickle` (token graph + serialized docs).
3. `run_annotators`: applies multiple annotators (coref, collection, purpose, list, subject, subsumption) to populate `PolicyDocument.token_relationship` edges.
4. `build_graph`: transforms token-level edges + entity/purpose normalization into final YAML graphs: `graph-original.full.yml` (debug, all SUBSUM edges) and `graph-original.yml` (trimmed). Optional GraphML via `--pretty`.
5. Analyses/evals consume YAML graphs (e.g., under `analyses/`, `evals/`).

Everything centers on `PolicyDocument` (see `poligrapher/document.py`). It stores:
- Segments (hierarchical; heading/list/text) built from accessibility roles.
- `all_docs`: contextual spaCy Docs keyed by (segment_id, context_depth).
- `token_relationship`: MultiDiGraph of inter-token relations added by annotators.

## Environment (Always Activate First)
Always run commands inside the dedicated conda env named `poligrapher` (dependencies: spaCy 3.5.x, transformers 1.2.4, SetFit, etc.).

Activation (macOS/Linux):
```
conda activate poligrapher
```
If the env is missing, create/update it:
```
conda env create -n poligrapher -f environment.yml   # first time
conda env update -n poligrapher -f environment.yml    # thereafter when deps change
```
Never install packages globally; modify `environment.yml` then update the env. Tests and pipeline scripts assume this env (e.g., presence of `en_core_web_md`, spaCy transformers version pin). If a command fails with ModuleNotFoundError, re-check that the env is active.

## Key Modules & Responsibilities
- `poligrapher/scripts/*.py`: Thin CLI wrappers orchestrating steps. Favor importing `main()` for programmatic tests (see `tests/test_pipeline.py`).
- `poligrapher/document.py`: Segment extraction, serialization (`document.pickle`), token source mapping. When extending, maintain `user_data['source']` invariants.
- `poligrapher/utils.py`: NLP pipeline assembly (`setup_nlp_pipeline`) combining custom NER + base spaCy model; adds custom components `align_noun_phrases`, `label_all_phrases`.
- `poligrapher/annotators/*`: Each annotator adds semantic edges (names listed in `run_annotators`). Only modify one annotator per change; test end‑to‑end with a small fixture workdir.
- `poligrapher/scripts/build_graph.py`: Multi-stage graph construction (steps 1‑9 inside `GraphBuilder.build_graph`). When adjusting logic, keep the ordering because downstream reductions (transitive reduction, normalization) assume earlier maps are stable.
- `poligrapher/purpose_classification.py`: Wraps SetFit model; labels purposes for DATA collection edges.

## Data & Artifacts
Workdir produced artifacts (chronological):
```
accessibility_tree.json
cleaned.html
readability.json
(document.pickle)
(graph-original.full.yml)
(graph-original.yml)
(optional graph-original.graphml)
```
Never assume intermediate presence except those prior in sequence; validate before consuming.

## External Dependencies & Constraints
- spaCy 3.5.x + transformers 1.2.4; constrained to older transformers (<4.30). Avoid upgrading transformers casually; it can break custom NER.
- SetFit / sentence-transformers pinned (no 3.x). Purpose model path resolved in `build_graph` (falls back to `extra-data`).
- Playwright (Firefox) used headless; network + JS injection (Readability). For tests, we avoid Playwright and supply a synthetic `accessibility_tree.json`.
- Extra model assets expected in `poligrapher/extra-data` (NER, phrase_map.yml, entity_info.json, purpose model). Guard code paths if assets missing.

## Testing Strategy
- Lightweight pipeline test added in `tests/test_pipeline.py` constructs a minimal workdir fixture and runs `init_document -> run_annotators -> build_graph` without crawling or Playwright.
- When adding annotator logic, extend that test or add new tests under `tests/` reusing the minimal fixture or adding new ones in `tests/fixtures/`.
- Avoid network / Playwright in unit tests; simulate by crafting accessibility tree JSON matching roles used in `SegmentExtractor` (e.g., `document`, `heading`, `paragraph`).

## Common Pitfalls / Conventions
- Do not overwrite spaCy `vocab.vectors` unless you know shapes match; earlier errors (E896) arose from vector table mismatches.
- Always call `PolicyDocument.save()` after mutating `token_relationship` or segments to persist downstream usage.
- Annotator edges: use predefined relationship labels consistently; adding new edge types may require updating `CollectionAnnotator.EDGE_TYPES` and graph reduction rules.
- Graph trimming logic depends on SUBSUM DAG properties; ensure new edges preserve acyclicity (`dag_add_edge`).
- Normalization step may insert synthetic tokens (`UNSPECIFIED_DATA/ACTOR`); maintain those patterns for analyses expecting them.

## Adding Features Safely
1. Add/modify logic in a single module.
2. Update or create small fixture(s) if new roles/edges needed.
3. Run `pytest -q` inside the conda env `poligrapher`.
4. If changing graph schema (node attributes / edge keys), update any analysis scripts reading YAML.

## Useful Examples
- End-to-end generation (no crawling) in tests: see `test_pipeline.py` (programmatic calls to script `main()` functions).
- Graph construction phases: open `GraphBuilder.build_graph` for the ordered transformations (phrase labeling → collect/coref/subsum → normalization → merge → finalize).

## Commands Reference (within activated env)
```
python -m poligrapher.scripts.html_crawler <URL|file> <workdir>
python -m poligrapher.scripts.init_document <workdir...>
python -m poligrapher.scripts.run_annotators <workdir...>
python -m poligrapher.scripts.build_graph [--pretty] <workdir...>
pytest -q
```

## When Unsure
Prefer reading `document.py` + `build_graph.py` to understand token → phrase → graph flows before altering annotators. Keep changes minimal and additive.
