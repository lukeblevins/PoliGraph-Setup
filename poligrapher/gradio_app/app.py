import glob
import os
import gradio as gr
import logging as logger
from poligrapher.gradio_app.functions import process_policy_url, get_policy_info, PolicyAnalysisResult
from poligrapher.scripts import build_graph, html_crawler, init_document, pdf_parser, run_annotators
import pandas as pd
import yaml
import networkx as nx
import matplotlib.pyplot as plt


# Setup logging
logger.basicConfig(level=logger.INFO, format="[%(asctime)s] %(message)s")
logger = logger.getLogger(__name__)

def get_company_df():
    csv_path = "./poligrapher/gradio_app/policy_list.csv"
    df = pd.read_csv(csv_path)

    # Add a column for YML existence (success indicator)
    def yml_exists(row):
        company_name = str(row.get("Company Name", "")).replace(" ", "_")
        yml_path = f"./output/{company_name}/graph-original.full.yml"
        return os.path.exists(yml_path)

    df["Status"] = df.apply(yml_exists, axis=1)

    # Move Status to the leftmost column
    cols = df.columns.tolist()
    if "Status" in cols:
        cols.insert(0, cols.pop(cols.index("Status")))
        df = df[cols]
    # Persist Status values back to the CSV without adding helper columns
    try:
        if "Company Name" in df.columns and "Status" in df.columns:
            orig_df = pd.read_csv(csv_path)
            if "Company Name" in orig_df.columns:
                status_map = df.set_index("Company Name")["Status"].to_dict()
                orig_df["Status"] = orig_df.get("Company Name").map(status_map)
                # Write the updated CSV (preserve original column order plus Status if new)
                orig_df.to_csv(csv_path, index=False)
    except Exception as e:
        logger.error("Failed to persist Status to CSV: %s", e)
    return df

def get_analysis_results():
    try:
        df = get_company_df()
        results = []
        for _, row in df.iterrows():
            results.append(PolicyAnalysisResult(
                company_name=str(row.get("Company Name", "")),
                privacy_policy_url=str(row.get("Privacy Policy URL", "")),
                score=row.get("Score", None),
                kind="auto",  # Default, can be set from another column if present
                has_name=bool(row.get("Company Name", "")),
                has_score=row.get("Score", None) is not None
            ))
        return results
    except Exception as e:
        logger.error("Error loading companies from CSV: %s", e)
        # Return a list with a single PolicyAnalysisResult containing the error
        return [PolicyAnalysisResult(company_name="error", privacy_policy_url="", score=None, kind="auto", has_name=False, has_score=False)]


def get_png_for_company(selected_row):
    if selected_row is None or not isinstance(selected_row, list) or len(selected_row) == 0:
        return None
    idx = selected_row[1]
    df = get_analysis_results()
    if idx >= len(df):
        return None
    domain = df.iloc[idx]["Domain Name"]
    png_path = f"./output/{domain}/knowledge_graph.png"
    if os.path.exists(png_path):
        return png_path
    return None

def generate_graph_from_html(html_path, output_folder):
    html_crawler.main(html_path, output_folder)
    init_document.main(workdirs=[output_folder])
    run_annotators.main(workdirs=[output_folder])
    build_graph.main(workdirs=[output_folder])
    build_graph.main(pretty=True, workdirs=[output_folder])

def visualize_graph(output_folder):
    yml_file = os.path.join(output_folder, "graph-original.full.yml")
    output_png = os.path.join(output_folder, "knowledge_graph.png")
    if not os.path.exists(yml_file):
        return "YML file not found for visualization."
    try:
        with open(yml_file, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        G = nx.DiGraph()
        for node in data.get("nodes", []):
            G.add_node(node["id"], type=node["type"])
        for link in data.get("links", []):
            G.add_edge(link["source"], link["target"], label=link["key"])
        plt.figure(figsize=(20, 15), facecolor="white")
        pos = nx.spring_layout(G, k=0.5)
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=3000,
            node_color="lightblue",
            edge_color="gray",
        )
        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.title("Knowledge Graph")
        plt.savefig(output_png, facecolor="white")
        plt.close()
        return f"Visualization saved: {output_png}"
    except Exception as e:
        return f"Error visualizing graph: {e}"


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


def analyze_url(policy: PolicyAnalysisResult):
    try:
        logger.info("API triggered: analyze_url for company: %s, URL: %s", policy.company_name, policy.privacy_policy_url)
        output_info = process_policy_url(policy)
        result = output_info
        if (result is None) or (not output_info.get("success", True)):
            logger.error("Error processing policy URL: %s", output_info.get('message', 'Unknown error'))
            return {"error": output_info.get("message", "Unknown error")}
        else:
            logger.info("Policy URL processed successfully")
            total_score = result["total_score"]
            grade = result["grade"]
            category_scores = result["category_scores"]
            feedback = result["feedback"]
            graph_json_path = result.get("graph_json_path")
            logger.info("API analyze_url completed: %s", result)

            return {
                "total_score": total_score,
                "grade": grade,
                "category_scores": category_scores,
                "feedback": feedback,
                "graph_json_path": graph_json_path,
                "structured": result["structured"],
            }

    except Exception as e:
        logger.error("Error in analyze_url: %s", e)
        return {"error": str(e)}


def fetch_policy_data():
    try:
        top_policies, low_policies, recent_policies = get_policy_info()
        return {
            "top_policies": top_policies,
            "low_policies": low_policies,
            "recent_policies": recent_policies,
        }
    except Exception as e:
        return {"error": str(e)}


with gr.Blocks() as block1:
    gr.Markdown("#### PoliGraph-er Demo")

    company_df = get_company_df()
    domain_names = company_df["Domain Name"].drop_duplicates().tolist() if "Domain Name" in company_df else []

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
    company_df_data = get_company_df()
    # Count successes and errors from Status column
    # Ensure boolean counts (Status is maintained as bool and persisted to CSV as bool)
    num_success = int(company_df_data["Status"].astype(bool).sum())
    num_error = int(len(company_df_data) - num_success)
    gr.Markdown(
        f"**Status Summary:** {num_success} successful, {num_error} with incomplete YML generation."
    )
    # Enable the button for demonstration and add a progress bar
    score_btn = gr.Button("Score All", interactive=True)
    # Show only relevant columns, including Status
    display_cols = [col for col in company_df_data.columns if col not in ["YML Exists"]]

    # Prepare a display copy where Status is shown as an emoji, but keep the underlying CSV boolean-only
    def _status_to_emoji(v):
        try:
            return "✅" if bool(v) else "⚠️"
        except Exception:
            return "⚠️"

    display_df = company_df_data.copy()
    if "Status" in display_df.columns:
        display_df["Status"] = display_df["Status"].apply(_status_to_emoji)

    with gr.Row():
        company_df = gr.Dataframe(
            value=display_df[display_cols], label="Companies", interactive=False
        )
    with gr.Row():
        company_info = gr.Markdown("", visible=True)
    with gr.Row():
        png_image = gr.Image(label="Knowledge Graph", visible=True)
    scoring_output = gr.Textbox(label="Scoring Results", interactive=False)

    def on_company_select(_df: pd.DataFrame, selection: gr.SelectData):
        if selection is None:
            return "", None
        row_value = selection.row_value
        company_name = row_value[1] if len(row_value) > 0 else ""
        company_url = row_value[2] if len(row_value) > 1 else ""
        info_md = f"<h1>{company_name}</h1><br><b>Website:</b> {company_url}"
        png_path = f"./output/{company_name.replace(' ', '_')}/knowledge_graph.png"
        if os.path.exists(png_path):
            return info_md, png_path
        return info_md, None

    def score_all(progress=gr.Progress(track_tqdm=True)):
        results = get_analysis_results()
        n = len(results)
        output_lines = []
        for i, result in enumerate(results):
            company_name = result["Company Name"]
            privacy_url = result["Privacy Policy URL"]
            try:
                score_info = analyze_url(privacy_url)
                result.score = score_info.get("total_score", None)
            except Exception as e:
                logger.error("Error scoring policy for %s: %s", company_name, e)
                result.has_score = False
                result.score = None
            output_lines.append(f"{company_name}: {result.score}")
            progress((i + 1) / n)
        return "\n".join(output_lines)

    company_df.select(
        fn=on_company_select, inputs=[company_df], outputs=[company_info, png_image]
    )
    score_btn.click(score_all, inputs=[], outputs=scoring_output, show_progress=True)

if __name__ == "__main__":
    app = gr.TabbedInterface([block1,block2], tab_names=["Demo", "Saved Results"])
    app.launch(share=True)
