# PoliGraph-er Gradio App

Gradio-based frontend for exploring generated privacy policy knowledge graphs and related analyses. The app wraps the core pipeline (crawl / init / annotators / graph build) and now uses lightweight domain entities instead of passing raw CSV rows between callbacks.

## Quick Start

1. Activate the project environment (required for spaCy + models):
   ```bash
   conda activate poligrapher
   ```
2. Launch the UI from the repository root:
   ```bash
   python -m poligrapher.gradio_app.app
   ```
3. Provide a policy URL (or point to an existing workdir) and generate / view the graph & scores.

## Why Entity Classes (vs. Old CSV Rows)?
Previously the UI, notebooks, and ad‑hoc scripts shared policy metadata via Pandas DataFrames. Each place re‑implemented logic like:
* Rebuilding output directories from provider names.
* Checking for artifact existence (`graph-original.yml`, `knowledge_graph.png`).
* Branching on capture modality (`if row["source"] == 'pdf': ...`).

This produced duplicated string literals, hard‑to‑trace bugs after renames, and unclear separation between a *captured document* and an *analysis result*.

The new model (see `policy_analysis.py`):
* `PolicyDocumentInfo` – encapsulates one captured policy's artifact directory and text extraction (PDF or HTML bundle).
* `PolicyAnalysisResult` – binds a specific graph variant + score to a document.
* `PolicyDocumentProvider` – groups all documents & results for a company/provider.
* Enums (`DocumentCaptureSource`, `GraphKind`) – replace free‑form strings and reduce silent typos.

Benefits:
* Single point of change for artifact naming and path conventions.
* Explicit, typed semantics aid readability & IDE support.
* Easier extension (add attributes / methods without breaking CSV schemas).

## Architecture Overview

Gradio callbacks (in `app.py`) consume these entity objects to:
1. Locate or create a workdir under `./output/<Provider_Slug>/...`.
2. Run pipeline stages (HTML/PDF ingestion → document init → annotators → graph build).
3. Store resulting YAML + rendered `knowledge_graph.png`.
4. Display scores / graphs, referencing helper methods like `PolicyDocumentInfo.has_graph()`.

`policy_analysis.py` isolates file system knowledge so UI logic stays declarative.

## Extending the App

Add a new graph / score variant:
1. Extend `GraphKind` enum with a new member.
2. Produce the artifact (e.g., alternate YAML or metrics) during analysis.
3. Instantiate a `PolicyAnalysisResult(document=..., score=..., kind=GraphKind.NEW_KIND)` and append via `provider.add_result(...)`.

Add new document metadata (e.g., policy version hash):
1. Add a field + constructor param to `PolicyDocumentInfo`.
2. Populate it when constructing the object (keeps downstream stable).

Change artifact naming (e.g., new graph image filename):
1. Update the method in `PolicyDocumentInfo` (`has_graph`, `has_image`, or `get_graph_image_path`).
2. Remove any remaining hard-coded references in UI code (should be minimal).

## Testing / Debugging Tips
* Leverage the existing pipeline test fixtures (see `tests/`) for end‑to‑end checks without launching Gradio.
* If extraction fails, call `PolicyDocumentInfo.get_document_text()` in a REPL to validate PDF/HTML parsing in isolation.
* Keep PDF parsing dependencies inside the conda env (avoid system `poppler` assumptions).

## Structure
* `app.py` – Main Gradio app entry point (UI + callbacks).
* `policy_analysis.py` – Domain entities & helper logic (document/result/provider abstractions).
* `__init__.py` – Package marker.

## Minimal Example (Pseudo-code)
```python
from datetime import date
from poligrapher.gradio_app.policy_analysis import (
	PolicyDocumentInfo, PolicyAnalysisResult, PolicyDocumentProvider,
	DocumentCaptureSource, GraphKind
)

provider = PolicyDocumentProvider(name="Example Corp", industry="Finance")
doc = PolicyDocumentInfo(
	path="./raw/ExampleCorp/policy.pdf",
	output_dir="./output/Example_Corp/policy-2025-09-05",
	source=DocumentCaptureSource.PDF,
	capture_date=date.today(),
	has_results=False,
)
provider.add_document(doc)

# After running pipeline & computing score
result = PolicyAnalysisResult(document=doc, score=0.82, kind=GraphKind.STANDARD)
provider.add_result(result)
```

## Future Enhancements (Ideas)
* Persist entity metadata (JSON alongside workdir) for quick reload without rescanning filesystem.
* Add versioning / provenance fields (pipeline commit hash, model versions) to `PolicyAnalysisResult`.
* Introduce a repository/service layer if multiple storage backends are needed.

---
For deeper pipeline details, consult the root project README and `poligrapher/document.py` & `poligrapher/scripts/*.py`.
