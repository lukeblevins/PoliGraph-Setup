import glob
import os
import gradio as gr
import logging as logger

from poligrapher.gradio_app import functions
from poligrapher.gradio_app.functions import visualize_graph
from poligrapher.gradio_app.policy_analysis import (
    DocumentCaptureSource,
    GraphKind,
    PolicyAnalysisResult,
    PolicyDocumentInfo,
    PolicyDocumentProvider,
)
from poligrapher.scripts import build_graph, html_crawler, init_document, pdf_parser, run_annotators
import pandas as pd


# Setup logging
logger.basicConfig(level=logger.INFO, format="[%(asctime)s] %(message)s")
logger = logger.getLogger(__name__)

# Global in‑memory provider registry
providers: list[PolicyDocumentProvider] = []
#
#     # Add a column for YML existence (success indicator)
#     def yml_exists(row):
#         company_name = str(row.get("Company Name", "")).replace(" ", "_")
#         yml_path = f"./output/{company_name}/graph-original.full.yml"
#         return os.path.exists(yml_path)
#     df["Status"] = df.apply(yml_exists, axis=1)
#
#     # Move Status to the leftmost column
#     cols = df.columns.tolist()
#     if "Status" in cols:
#         cols.insert(0, cols.pop(cols.index("Status")))
#         df = df[cols]
#     # Persist Status values back to the CSV without adding helper columns
#     try:
#         if "Company Name" in df.columns and "Status" in df.columns:
#             orig_df = pd.read_csv(csv_path)
#             if "Company Name" in orig_df.columns:
#                 status_map = df.set_index("Company Name")["Status"].to_dict()
#                 orig_df["Status"] = orig_df.get("Company Name").map(status_map)
#                 # Write the updated CSV (preserve original column order plus Status if new)
#                 orig_df.to_csv(csv_path, index=False)
#     except Exception as e:
#         logger.error("Failed to persist Status to CSV: %s", e)
#     return df

# def get_analysis_results():
#     try:
#         df = get_company_df()
#         results = []
#         for _, row in df.iterrows():
#             results.append(PolicyAnalysisResult(
#                 company_name=str(row.get("Company Name", "")),
#                 privacy_policy_url=str(row.get("Privacy Policy URL", "")),
#                 score=row.get("Score", None),
#                 kind="auto",  # Default, can be set from another column if present
#                 has_name=bool(row.get("Company Name", "")),
#                 has_score=row.get("Score", None) is not None
#             ))
#         return results
#     except Exception as e:
#         logger.error("Error loading companies from CSV: %s", e)
#         # Return a list with a single PolicyAnalysisResult containing the error
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
    providers.append(provider)


def add_document_to_provider(
    provider: PolicyDocumentProvider,
    path: str,
    output_dir: str,
    source: DocumentCaptureSource,
    capture_date: str,
    has_results: bool,
):
    document = PolicyDocumentInfo(
        path=path,
        output_dir=output_dir,
        source=source,
        capture_date=capture_date,
        has_results=has_results,
    )
    provider.add_document(document)


def add_result_to_provider(
    provider: PolicyDocumentProvider,
    document: PolicyDocumentInfo,
    score: float,
    kind: GraphKind,
):
    provider.add_result(PolicyAnalysisResult(document=document, score=score, kind=kind))


def generate_graph_from_html(html_path, output_folder):
    html_crawler.main(html_path, output_folder)
    init_document.main(workdirs=[output_folder])
    run_annotators.main(workdirs=[output_folder])
    build_graph.main(workdirs=[output_folder])
    build_graph.main(pretty=True, workdirs=[output_folder])


def process_policy(policy_url: str, policy_kind: str, company_name: str):
    output_folder = f"./output/{company_name.replace(' ', '_')}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if policy_kind.lower() == "pdf":
        pdf_parser.main(policy_url, output_folder)
        html_path = os.path.join(output_folder, "output.html")
        generate_graph_from_html(html_path, output_folder)
    else:
        generate_graph_from_html(policy_url, output_folder)
    # Find the .graphml file to return as output
    graphml_files = glob.glob(os.path.join(output_folder, "*.graphml"))
    vis_result = visualize_graph(output_folder)
    if graphml_files:
        return f"Graph generated: {graphml_files[0]}\n{vis_result}"
    else:
        return f"Graph generation completed, but no .graphml file found.\n{vis_result}"


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
            doc = PolicyDocumentInfo(
                path=row.get("Policy URL"),
                output_dir=f"./output/{row.get('Provider','').replace(' ', '_')}",
                source=_safe_enum_from_value(row.get("Source")),
                capture_date=row.get("Date"),
                has_results=row.get("Status", False),
            )
            provider.add_document(doc)
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
        return process_policy(url, kind, company_name)

    submit_btn.click(
        on_submit_click,
        inputs=[company_name_input, privacy_policy_input, kind_input],
        outputs=output_text
    )


with gr.Blocks() as block2:
    gr.Markdown("#### Company Privacy Policy List")

    get_providers("./poligrapher/gradio_app/policy_list.csv")
    # Count successes and errors from Status column
    # Ensure boolean counts (Status is maintained as bool and persisted to CSV as bool)
    num_success = sum(
        1 for p in providers if p.documents and p.documents[0].has_results
    )
    num_error = len(providers) - num_success
    gr.Markdown(
        f"**Status Summary:** {num_success} successful, {num_error} with incomplete YML generation."
    )
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
            value=_build_display_df(), label="Companies", interactive=False
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
        png_image = gr.Image(label="Knowledge Graph", visible=True)
    scoring_output = gr.Textbox(label="Scoring Results", interactive=False)

    # Now add policies accordion (documents for selected provider)
    selected_provider = gr.State("")
    with gr.Accordion("Provider Policies", open=False) as policies_accordion:
        with gr.Row():
            policies_df = gr.Dataframe(
                value=_build_policies_df(), label="Policies", interactive=False
            )
        with gr.Row():
            refresh_policies = gr.Button("Refresh Policies")
            add_policy_btn = gr.Button("Add Policy", variant="secondary")

    # ----- Add Policy Modal -----
    with gr.Group(visible=False, elem_id="add-policy-modal") as add_policy_modal:
        with gr.Column(elem_classes="modal-card"):
            gr.Markdown("### Add Policy to Provider")
            new_policy_url = gr.Textbox(label="Policy URL", placeholder="https://...")
            new_policy_source = gr.Dropdown(
                choices=[s.value for s in DocumentCaptureSource],
                label="Source Type",
                value=DocumentCaptureSource.WEBPAGE.value,
            )
            new_policy_date = gr.Textbox(label="Capture Date (YYYY-MM-DD)")
            with gr.Row():
                save_new_policy = gr.Button("Save", variant="primary")
                cancel_new_policy = gr.Button("Cancel")

    def _show_add_policy_modal(provider_name: str):
        if not provider_name:
            # No provider selected; keep hidden
            return gr.update(visible=False)
        return gr.update(visible=True)

    def _cancel_add_policy():
        return gr.update(visible=False), "", DocumentCaptureSource.WEBPAGE.value, ""

    def _save_new_policy(provider_name: str, url: str, source_val: str, date_str: str):
        # Find provider
        prov = next((p for p in providers if p.name == provider_name), None)
        if not prov or not url:
            return (
                _build_policies_df(provider_name),
                gr.update(visible=False),
                "",
                DocumentCaptureSource.WEBPAGE.value,
                "",
            )
        # Map source
        try:
            src_enum = DocumentCaptureSource(source_val)
        except Exception:
            src_enum = DocumentCaptureSource.WEBPAGE
        # Basic date fallback
        capture_date = date_str if date_str else ""
        # Create document (no results yet)
        doc = PolicyDocumentInfo(
            path=url,
            output_dir=f"./output/{prov.name.replace(' ', '_')}",
            source=src_enum,
            capture_date=capture_date,
            has_results=False,
        )
        prov.add_document(doc)
        return (
            _build_policies_df(provider_name),
            gr.update(visible=False),
            "",
            DocumentCaptureSource.WEBPAGE.value,
            "",
        )

    add_policy_btn.click(
        _show_add_policy_modal,
        inputs=[selected_provider],
        outputs=[add_policy_modal],
    )
    cancel_new_policy.click(
        _cancel_add_policy,
        inputs=[],
        outputs=[add_policy_modal, new_policy_url, new_policy_source, new_policy_date],
    )
    save_new_policy.click(
        _save_new_policy,
        inputs=[selected_provider, new_policy_url, new_policy_source, new_policy_date],
        outputs=[
            policies_df,
            add_policy_modal,
            new_policy_url,
            new_policy_source,
            new_policy_date,
        ],
    )

    def _refresh_all(curr_provider):
        return _build_display_df(), _build_policies_df(curr_provider)

    refresh_policies.click(
        _refresh_all, inputs=[selected_provider], outputs=[company_df, policies_df]
    )

    def on_policy_select(_df: pd.DataFrame, selection: gr.SelectData):
        if selection is None:
            return gr.update(), gr.update()
        row = selection.row_value
        prov = row[0]
        policy_url = row[1]
        info_md = f"<h1>{prov}</h1><br><b>Policy:</b> {policy_url}"
        png_path = f"./output/{prov.replace(' ', '_')}/knowledge_graph.png"
        return info_md, png_path if os.path.exists(png_path) else None

    policies_df.select(
        fn=on_policy_select, inputs=[policies_df], outputs=[company_info, png_image]
    )

    def on_company_select(_df: pd.DataFrame, selection: gr.SelectData):
        """Handle provider/company row selection: update info panel only and clear image.

        PNG preview is now reserved for policy (document) selection in the Policies accordion.
        """
        _ = _df  # avoid unused warning
        if selection is None:
            return "", None, _build_policies_df(""), "", gr.update(open=False)
        row_value = selection.row_value
        company_name = row_value[1] if len(row_value) > 0 else ""
        company_url = row_value[2] if len(row_value) > 1 else ""
        info_md = f"<h1>{company_name}</h1><br><b>Website:</b> {company_url}"
        # Return info, cleared image, filtered policies df, and selected provider state
        # Open the policies accordion when a provider is selected
        return (
            info_md,
            None,
            _build_policies_df(company_name),
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

if __name__ == "__main__":
    app = gr.TabbedInterface([block1, block2], tab_names=["Demo", "Companies"])
    app.launch(share=True)
