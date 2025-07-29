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
    return pd.read_csv(csv_path)

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
    idx = selected_row[0]
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
        with open(yml_file, "r") as file:
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
    score_btn = gr.Button("Score All")
    company_df = gr.Dataframe(value=get_company_df(), label="Companies", interactive=False)
    company_info = gr.Markdown("", visible=True)
    png_image = gr.Image(label="Knowledge Graph", visible=True)
    scoring_output = gr.Textbox(label="Scoring Results", interactive=False)

    def on_company_select(df: pd.DataFrame, selection: gr.SelectData):
        print("selected:", selection, type(selection))
        if selection is None:
            return "", None
        row_value = selection.row_value
        # row_value: [Company Name, Company Website URL, Domain Name, Privacy Policy URL, Notes, Score]
        company_name = row_value[0] if len(row_value) > 0 else ""
        company_url = row_value[1] if len(row_value) > 1 else ""
        info_md = f"**<h1>{company_name}</h1>**<br>**<p>Website:</p>** {company_url}"
        png_path = f"./output/{company_name}/knowledge_graph.png"
        png_path = png_path.replace(" ", "_")
        if os.path.exists(png_path):
            return info_md, png_path
        return info_md, None

    def score_all():
        results = get_analysis_results()
        for result in results:
            company_name = result["Company Name"]
            privacy_url = result["Privacy Policy URL"]

            try:
                score_info = analyze_url(privacy_url)
                result.score = score_info.get("total_score", None)
            except Exception as e:
                logger.error("Error scoring policy for %s: %s", company_name, e)
                result.has_score = False
                result.score = None
        return "\n".join(results)

    company_df.select(fn=on_company_select, inputs=company_df, outputs=[company_info, png_image])
    score_btn.click(score_all, inputs=[], outputs=scoring_output)

if __name__ == "__main__":
    app = gr.TabbedInterface([block1,block2], tab_names=["Demo", "Saved Results"])
    app.launch(share=True)
