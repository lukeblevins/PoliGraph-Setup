import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import gradio as gr
import logging
import pandas as pd

from poligrapher.gradio_app import functions
from poligrapher.gradio_app.policy_analysis import (
    DocumentCaptureSource,
    GraphKind,
    PolicyAnalysisResult,
    PolicyDocumentInfo,
    PolicyDocumentProvider,
)
# (Legacy direct script imports removed; generation orchestrated through functions.generate_graph)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Global in‑memory provider registry
providers: list[PolicyDocumentProvider] = []
## (Legacy CSV Status augmentation code omitted for clarity)

# def get_analysis_results():
#     try:
#         df = get_company_df()
#         results = []
#         return results
#     except Exception as e:
#         logger.error("Error loading companies from CSV: %s", e)
#         return [PolicyAnalysisResult(company_name="error", privacy_policy_url="", score=None, kind="auto", has_name=False, has_score=False)]

# TODO: Modify to use PolicyAnalysisResult.get_graph_image_path()
# def get_png_for_company(selected_row):
#     if selected_row is None or not isinstance(selected_row, list) or len(selected_row) == 0:
#         return None
#     idx = selected_row[1]
#     df = get_analysis_results()
#     if idx >= len(df):
#         return None
#     domain = df.iloc[idx]["Domain Name"]
#     png_path = f"./output/{domain}/knowledge_graph.png"
#     if os.path.exists(png_path):
#         return png_path
#     return None


def add_provider(name: str, industry: str):
    provider = PolicyDocumentProvider(name=name, industry=industry)
    if any(p.name == name for p in providers):
        return  # Provider already exists

    # create provider folder
    os.makedirs(provider.output_dir, exist_ok=True)
    providers.append(provider)


def add_document_to_provider(
    provider: PolicyDocumentProvider,
    path: str,
    source: DocumentCaptureSource,
    capture_date: str,
    has_results: bool,
    generate_if_missing: bool = True,
) -> PolicyDocumentInfo:
    # output_dir // document capture date + source
    output_dir = f"{provider.output_dir}/{capture_date}_{source.value}"

    document = PolicyDocumentInfo(
        path=path,
        output_dir=output_dir,
        source=source,
        capture_date=capture_date,
        has_results=has_results,
    )

    # If initial load or already present, register and optionally generate.
    provider.add_document(document)

    if not generate_if_missing:
        return document

    if generate_if_missing:
        _ensure_graph_assets(document)

    return document


def add_result_to_provider(
    provider: PolicyDocumentProvider,
    document: PolicyDocumentInfo,
    score: float,
    kind: GraphKind,
):
    provider.add_result(PolicyAnalysisResult(document=document, score=score, kind=kind))

# def analyze_url(policy: PolicyAnalysisResult):
#     try:
#         logger.info("API triggered: analyze_url for company: %s, URL: %s", policy.company_name, policy.privacy_policy_url)
#         if getattr(policy, "has_graph", False):
#             logger.info(
#                 "Existing graph detected for %s; skipping regeneration and only scoring.",
#                 policy.company_name,
#             )
#             output_info = score_existing_policy(policy)
#         else:
#             output_info = process_policy_url(policy)

#         if (output_info is None) or (not output_info.get("success", True)):
#             logger.error("Error processing policy URL: %s", output_info.get('message', 'Unknown error'))
#             return {"error": output_info.get("message", "Unknown error")}

#         # output_info follows shape { success: True, message: ..., result: { ... } }
#         result_payload = output_info.get("result", {})
#         total_score = result_payload.get("total_score")
#         grade = result_payload.get("grade")
#         category_scores = result_payload.get("category_scores")
#         feedback = result_payload.get("feedback")
#         graph_json_path = result_payload.get("graph_json_path")
#         structured = result_payload.get("structured")

#         logger.info(
#             "API analyze_url completed: %s",
#             {
#                 "company": policy.company_name,
#                 "score": total_score,
#                 "grade": grade,
#                 "has_graph": getattr(policy, "has_graph", False),
#             },
#         )

#         return {
#             "total_score": total_score,
#             "grade": grade,
#             "category_scores": category_scores,
#             "feedback": feedback,
#             "graph_json_path": graph_json_path,
#             "structured": structured,
#         }

#     except Exception as e:
#         logger.error("Error in analyze_url: %s", e)
#         return {"error": str(e)}


def get_providers(csv_file: str):
    # Reset existing providers to avoid duplicates on repeated calls
    providers.clear()
    df = pd.read_csv(csv_file)

    # Assume the CSV "Source" values are already normalized to enum values
    # (e.g. "pdf" or "webpage"). Provide a tiny safe helper to construct
    # the enum from that value and fallback to WEBPAGE on error or missing.
    def _safe_enum_from_value(val) -> DocumentCaptureSource:
        try:
            if val is None:
                return DocumentCaptureSource.WEBPAGE
            return DocumentCaptureSource(str(val))
        except Exception:
            return DocumentCaptureSource.WEBPAGE

    for name, group in df.groupby("Provider"):
        provider = PolicyDocumentProvider(name, industry=group.iloc[0]["Industry"])
        for _, row in group.iterrows():
            doc = add_document_to_provider(
                provider=provider,
                path=row.get("Policy URL"),
                source=_safe_enum_from_value(row.get("Source")),
                capture_date=row.get("Date"),
                has_results=row.get("Status", False),
                generate_if_missing=False,  # Avoid heavy generation on initial load
            )
            # Safe GraphKind parsing (CSV may hold value or name, case-insensitive)
            gk_val = row.get("Graph Kind")
            graph_kind = None
            if isinstance(gk_val, str) and gk_val.strip():
                key = gk_val.strip().upper()
                try:
                    graph_kind = GraphKind[key]
                except KeyError:
                    # Try match by enum value
                    for m in GraphKind:
                        if m.value.lower() == gk_val.strip().lower():
                            graph_kind = m
                            break
            # Only add a result if score present or graph_kind present
            if (row.get("Score") is not None) or (graph_kind is not None):
                provider.add_result(
                    PolicyAnalysisResult(
                        document=doc,
                        score=row.get("Score"),
                        kind=graph_kind,
                    )
                )
        providers.append(provider)


with gr.Blocks() as block1:
    gr.Markdown("#### PoliGraph-er Demo")
    company_name_input = gr.Textbox(label="Company Name")
    privacy_policy_input = gr.Textbox(label="Privacy Policy URL")
    kind_input = gr.Radio(choices=["Auto", "Webpage", "PDF"], label="Document Method", value="Auto")
    submit_btn = gr.Button("Generate Graph")
    output_text = gr.Textbox(label="Result", interactive=False)

    def on_submit_click(company_name, privacy_policy_url, kind):
        url = privacy_policy_url
        return None
        # return process_policy(url, kind, company_name)

    submit_btn.click(
        on_submit_click,
        inputs=[company_name_input, privacy_policy_input, kind_input],
        outputs=output_text
    )


with gr.Blocks() as block2:
    gr.Markdown("#### Company Privacy Policy List")
    # Lazy load: summary placeholder (populated on .load())
    status_md = gr.Markdown("")
    # Enable the button for demonstration and add a progress bar
    score_btn = gr.Button("Score All", interactive=True)
    # Show only relevant columns, including Status
    display_cols = [
        "Status",
        "Provider",
        "Policy URL",
        "Industry",
        "Source",
        "Date",
        "Score",
        "Graph Kind",
    ]

    # Prepare a display copy where Status is shown as an emoji, but keep the underlying CSV boolean-only

    def _status_to_emoji(v):
        try:
            return "✅" if bool(v) else "❌"
        except Exception:
            return "❌"

    def _build_display_df():
        """Build dataframe rows from providers (include providers without documents)."""
        rows = []
        for provider in providers:
            if provider.documents:
                for doc in provider.documents:
                    rows.append(
                        {
                            "Status": _status_to_emoji(doc.has_results),
                            "Provider": provider.name,
                            "Industry": provider.industry,
                            "Policy URL": doc.path,
                            "Source": doc.source,
                            "Date": doc.capture_date,
                            "Score": next(
                                (
                                    r.score
                                    for r in provider.results
                                    if r.document == doc
                                ),
                                None,
                            ),
                            "Graph Kind": next(
                                (r.kind for r in provider.results if r.document == doc),
                                None,
                            ),
                        }
                    )
            else:
                # Placeholder row so newly added provider (no documents yet) is visible
                rows.append(
                    {
                        "Status": _status_to_emoji(False),
                        "Provider": provider.name,
                        "Industry": provider.industry,
                        "Policy URL": "",
                        "Source": "",
                        "Date": "",
                        "Score": None,
                        "Graph Kind": None,
                    }
                )
        return pd.DataFrame(rows, columns=display_cols)

    with gr.Row():
        company_df = gr.Dataframe(
            headers=[
                "Status",
                "Provider",
                "Policy URL",
                "Industry",
                "Source",
                "Date",
                "Score",
                "Graph Kind",
            ],
            value=[],
            label="Companies",
            interactive=False,
        )

    # ----- Policies (Documents) View (filtered by selected provider) -----
    def _build_policies_df(provider_filter: str | None = None):
        rows = []
        if not provider_filter:
            # No provider selected => empty table
            return pd.DataFrame(
                columns=[
                    "Provider",
                    "Policy URL",
                    "Source",
                    "Date",
                    "Status",
                    "Score",
                    "Graph Kind",
                ]
            )
        for provider in providers:
            if provider.name != provider_filter:
                continue
            for doc in provider.documents:
                rows.append(
                    {
                        "Provider": provider.name,
                        "Policy URL": doc.path,
                        "Source": (
                            doc.source.value
                            if hasattr(doc.source, "value")
                            else doc.source
                        ),
                        "Date": doc.capture_date,
                        "Status": "✅" if doc.has_results else "❌",
                        "Score": next(
                            (r.score for r in provider.results if r.document == doc),
                            None,
                        ),
                        "Graph Kind": next(
                            (r.kind for r in provider.results if r.document == doc),
                            None,
                        ),
                    }
                )
        return pd.DataFrame(
            rows,
            columns=[
                "Provider",
                "Policy URL",
                "Source",
                "Date",
                "Status",
                "Score",
                "Graph Kind",
            ],
        )

    # Policies UI will be added after company_info & png_image definitions

    # ----- Add Provider Modal UI -----
    add_provider_btn = gr.Button("Add Provider")
    with gr.Group(visible=False, elem_id="add-provider-modal") as add_provider_modal:
        with gr.Column(elem_classes="modal-card"):
            gr.Markdown("### Add Provider")
            new_provider_name = gr.Textbox(
                label="Provider Name", placeholder="e.g. Example Corp"
            )
            new_provider_industry = gr.Textbox(
                label="Industry", placeholder="e.g. Technology"
            )
            with gr.Row(elem_classes="two-btn-row"):
                save_new_provider = gr.Button("Save", variant="primary")
                cancel_new_provider = gr.Button("Cancel")

    def _show_add_provider_modal():
        return gr.update(visible=True)

    def _cancel_add_provider():
        return gr.update(visible=False), "", ""

    def _save_new_provider(name: str, industry: str):
        if name and industry:
            add_provider(name.strip(), industry.strip())
        updated_df = _build_display_df()
        return (
            updated_df,
            gr.update(visible=False),
            "",
            "",
        )

    add_provider_btn.click(
        _show_add_provider_modal, inputs=[], outputs=[add_provider_modal]
    )
    cancel_new_provider.click(
        _cancel_add_provider,
        inputs=[],
        outputs=[add_provider_modal, new_provider_name, new_provider_industry],
    )
    save_new_provider.click(
        _save_new_provider,
        inputs=[new_provider_name, new_provider_industry],
        outputs=[
            company_df,
            add_provider_modal,
            new_provider_name,
            new_provider_industry,
        ],
    )
    with gr.Row():
        company_info = gr.Markdown("", visible=True)
    with gr.Row():
        with gr.Column(scale=1):
            png_image = gr.Image(label="Knowledge Graph", visible=True)
        with gr.Column(scale=1):
            # Policies (documents) sidebar next to image
            selected_provider = gr.State("")
            with gr.Accordion("Provider Policies", open=False) as policies_accordion:
                with gr.Row():
                    policies_df = gr.Dataframe(
                        value=_build_policies_df(), label="Policies", interactive=False
                    )
                with gr.Row():
                    refresh_policies = gr.Button("Refresh Policies")
                    add_policy_btn = gr.Button("Add Policy", variant="secondary")
            # Add Policy Modal (initially hidden)
            with gr.Group(
                visible=False, elem_id="add-policy-modal"
            ) as add_policy_modal:
                with gr.Column(elem_classes="modal-card"):
                    gr.Markdown("### Add Policy to Provider")
                    new_policy_url = gr.Textbox(
                        label="Policy URL", placeholder="https://...", visible=True
                    )
                    # Custom source selector: webpage vs remote PDF (URL) vs local PDF (upload)
                    new_policy_source = gr.Dropdown(
                        choices=[
                            "webpage",  # URL of a webpage
                            "pdf_remote",  # URL pointing directly to a PDF
                            "pdf_local",  # Locally uploaded PDF file
                        ],
                        label="Source Type",
                        value="webpage",
                    )
                    new_policy_file = gr.File(
                        label="Upload File (PDF/HTML)",
                        file_types=[".pdf", ".html", ".htm"],
                        visible=False,
                    )
                    new_policy_date = gr.Textbox(label="Capture Date (YYYY-MM-DD)")
                    with gr.Row():
                        save_new_policy = gr.Button("Save", variant="primary")
                        cancel_new_policy = gr.Button("Cancel")
    scoring_output = gr.Textbox(label="Scoring Results", interactive=False)

    def _show_add_policy_modal(provider_name: str):
        if not provider_name:
            # No provider selected; keep hidden
            return gr.update(visible=False)
        return gr.update(visible=True)

    def _cancel_add_policy():
        return (
            gr.update(visible=False),
            "",
            "webpage",
            "",
            None,
        )

    def _save_new_policy(
        provider_name: str,
        url: str,
        source_val: str,
        date_str: str,
        uploaded_file,
    ):
        prov = next((p for p in providers if p.name == provider_name), None)
        if prov is None:
            # No provider selected; just close modal
            return (
                _build_policies_df(""),
                gr.update(visible=False),
                "",
                "webpage",
                "",
                None,
            )

        # Determine if local PDF expected
        if source_val == "pdf_local" and uploaded_file is None:
            # Keep modal open until a file is provided
            return (
                _build_policies_df(provider_name),
                gr.update(visible=True),
                url,
                source_val,
                date_str,
                uploaded_file,
            )

        file_path = None
        if uploaded_file is not None:
            try:
                src_path = getattr(uploaded_file, "name", None) or uploaded_file
                filename = os.path.basename(src_path)
                out_dir = f"./output/{provider_name.replace(' ', '_')}"
                os.makedirs(out_dir, exist_ok=True)
                dest_path = os.path.join(out_dir, filename)
                import shutil

                shutil.copy(src_path, dest_path)
                file_path = dest_path
                # If a file is uploaded, enforce local PDF selection
                if filename.lower().endswith(".pdf"):
                    source_val = "pdf_local"
            except Exception as e:
                logger.error("File upload failed: %s", e)

        # Compute path_value based on source type
        if source_val in ("webpage", "pdf_remote"):
            path_value = url.strip() if url else ""
        else:  # pdf_local
            path_value = file_path or ""

        if not path_value:
            # Missing required path/URL
            return (
                _build_policies_df(provider_name),
                gr.update(visible=True),
                url,
                source_val,
                date_str,
                uploaded_file,
            )

        # Map UI source choice to enum (remote/local pdf both -> PDF)
        if source_val == "webpage":
            src_enum = DocumentCaptureSource.WEBPAGE
        else:
            src_enum = DocumentCaptureSource.PDF

        # Ensure a capture date string (fallback to placeholder if empty so path is unique)
        capture_date = (date_str or "unknown-date").strip()
        # Delegate creation to helper so output_dir follows pattern capture_date_source
        add_document_to_provider(
            provider=prov,
            path=path_value,
            source=src_enum,
            capture_date=capture_date,
            has_results=False,
        )

        return (
            _build_display_df(),                 # updated company (providers) table including status
            _build_policies_df(provider_name),   # updated policies list
            gr.update(visible=False),
            "",
            "webpage",
            "",
            None,
        )

    add_policy_btn.click(
        _show_add_policy_modal,
        inputs=[selected_provider],
        outputs=[add_policy_modal],
    )

    # Toggle file upload and URL visibility based on custom source type
    def _on_policy_source_change(source_val: str):
        is_local_pdf = source_val == "pdf_local"
        show_url = source_val in ("webpage", "pdf_remote")
        file_update = gr.update(
            visible=is_local_pdf, value=None if not is_local_pdf else None
        )
        url_update = gr.update(visible=show_url, value="" if not show_url else None)
        return file_update, url_update

    new_policy_source.change(
        _on_policy_source_change,
        inputs=[new_policy_source],
        outputs=[new_policy_file, new_policy_url],
    )

    # Auto-adjust source dropdown when a file is uploaded
    def _on_policy_file_change(uploaded_file, current_source):
        if uploaded_file is None:
            return current_source
        fname = getattr(uploaded_file, "name", "") or ""
        if fname.lower().endswith(".pdf"):
            return "pdf_local"
        if fname.lower().endswith((".html", ".htm")):
            return "webpage"
        return current_source

    new_policy_file.change(
        _on_policy_file_change,
        inputs=[new_policy_file, new_policy_source],
        outputs=[new_policy_source],
    )
    cancel_new_policy.click(
        _cancel_add_policy,
        inputs=[],
        outputs=[
            add_policy_modal,
            new_policy_url,
            new_policy_source,
            new_policy_date,
            new_policy_file,
        ],
    )
    save_new_policy.click(
        _save_new_policy,
        inputs=[
            selected_provider,
            new_policy_url,
            new_policy_source,
            new_policy_date,
            new_policy_file,
        ],
        outputs=[
            company_df,        # newly added to refresh status column
            policies_df,
            add_policy_modal,
            new_policy_url,
            new_policy_source,
            new_policy_date,
            new_policy_file,
        ],
    )

    # ---- Shared generation helper ----
    def _ensure_graph_assets(doc: PolicyDocumentInfo, force: bool = False) -> bool:
        """Ensure graph (YAML) and PNG image exist; update has_results accordingly.

        has_results is set True only when BOTH the graph and image exist.
        Returns True on full success, False otherwise.
        """
        success = False
        try:
            # Generate graph if missing or forced
            if force or (not doc.has_graph()):
                os.makedirs(doc.output_dir, exist_ok=True)
                functions.generate_graph(doc)
                logger.info("✅ Graph ready for %s", doc.path)
            # Generate image if missing but graph exists
            if doc.has_graph() and (not doc.has_image()):
                try:
                    functions.visualize_graph(doc)
                    logger.info("🖼️ Image created for %s", doc.path)
                except BaseException as e:
                    logger.warning("⚠️ Image generation failed for %s: %s", doc.path, e)
            # Final status evaluation
            success = doc.has_graph() and doc.has_image()
            doc.has_results = success
            if not success:
                logger.debug("Artifacts incomplete for %s (graph=%s, image=%s)", doc.path, doc.has_graph(), doc.has_image())
        except BaseException as e:
            logger.error("❌ Graph generation failed for %s: %s", doc.path, e)
            doc.has_results = False
            success = False
        return success

    def _refresh_all(curr_provider):
        """Refresh tables and lazily generate any missing artifacts."""
        for provider in providers:
            for doc in provider.documents:
                if not doc.has_graph() or not doc.has_image():
                    _ensure_graph_assets(doc)
        return _build_display_df(), _build_policies_df(curr_provider)

    refresh_policies.click(
        _refresh_all, inputs=[selected_provider], outputs=[company_df, policies_df]
    )

    def on_policy_select(_df: pd.DataFrame, selection: gr.SelectData):
        """Policy selection handler using SelectData.index (Gradio 3.48.0).

        selection.index -> (row, col) or list/tuple; we only need row.
        """
        if selection is None:
            return gr.update(), gr.update()
        try:
            # selection.index may be a tuple (row, col) or an int
            if isinstance(selection.index, (list, tuple)):
                row_idx = selection.index[0]
            else:
                row_idx = selection.index
            if row_idx is None or row_idx == "" or row_idx >= len(_df):
                return gr.update(), gr.update()
            row_series = _df.iloc[row_idx]
        except Exception:
            return gr.update(), gr.update()
        prov = row_series.get("Provider", "Unknown")
        policy_url = row_series.get("Policy URL", "")
        info_md = f"<h1>{prov}</h1><br><b>Policy:</b> {policy_url}"
        png_path = f"./output/{prov.replace(' ', '_')}/knowledge_graph.png"
        return info_md, png_path if os.path.exists(png_path) else None

    policies_df.select(
        fn=on_policy_select, inputs=[policies_df], outputs=[company_info, png_image]
    )

    def on_company_select(_df: pd.DataFrame, selection: gr.SelectData):
        """Provider/company selection handler using SelectData.index."""
        if selection is None:
            return "", None, _build_policies_df(""), "", gr.update(open=False)
        try:
            if isinstance(selection.index, (list, tuple)):
                row_idx = selection.index[0]
            else:
                row_idx = selection.index
            if row_idx is None or row_idx == "" or row_idx >= len(_df):
                return "", None, _build_policies_df(""), "", gr.update(open=False)
            row_series = _df.iloc[row_idx]
        except Exception:
            return "", None, _build_policies_df(""), "", gr.update(open=False)
        company_name = row_series.get("Provider", "")
        policies_df_val = _build_policies_df(company_name)
        # Attempt auto-select of first document (no Dataframe programmatic select API, emulate by populating info + image)
        first_image = None
        info_md: str
        provider_obj = next((p for p in providers if p.name == company_name), None)
        if provider_obj and provider_obj.documents:
            first_doc = provider_obj.documents[0]
            policy_url = first_doc.path
            info_md = f"<h1>{company_name}</h1><br><b>Policy:</b> {policy_url}"
            img_path = os.path.join(first_doc.output_dir, "knowledge_graph.png")
            if os.path.exists(img_path):
                first_image = img_path
        else:
            # Fallback to original provider-only info
            policy_url = row_series.get("Policy URL", "")
            info_md = f"<h1>{company_name}</h1><br><b>Website:</b> {policy_url}"
        return (
            info_md,
            first_image,
            policies_df_val,
            company_name,
            gr.update(open=True),
        )

    def score_all(progress=gr.Progress()):
        progress(0, "Starting...")
        for company in progress.tqdm(providers, desc="Scoring Policies"):
            for policy in company.documents:
                functions.score_policy(policy)

    company_df.select(
        fn=on_company_select,
        inputs=[company_df],
        outputs=[
            company_info,
            png_image,
            policies_df,
            selected_provider,
            policies_accordion,
        ],
    )
    score_btn.click(score_all, inputs=[], outputs=scoring_output, show_progress="full")

    # initial load (after client connects)
    def _initial_load():
        get_providers("./poligrapher/gradio_app/policy_list.csv")
        df = _build_display_df()
        num_success = sum(
            1 for p in providers if p.documents and p.documents[0].has_results
        )
        num_error = len(providers) - num_success
        summary = f"**Status Summary:** {num_success} successful, {num_error} with incomplete YML generation."
        return gr.update(value=df), gr.update(value=summary)

    block2.load(_initial_load, inputs=None, outputs=[company_df, status_md])

if __name__ == "__main__":
    app = gr.TabbedInterface([block1, block2], tab_names=["Demo", "Companies"])
    app.queue(concurrency_count=2)
    app.launch(share=True)
