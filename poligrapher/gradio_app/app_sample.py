import os
import gradio as gr
import glob
import yaml
import pandas as pd
from poligrapher.scripts import build_graph, html_crawler, init_document, pdf_parser, run_annotators

# Helper to generate graph from HTML or output.html
def generate_graph_from_html(html_path, output_folder):
    html_crawler.main(html_path, output_folder)
    init_document.main(workdirs=[output_folder])
    run_annotators.main(workdirs=[output_folder])
    build_graph.main(workdirs=[output_folder])
    build_graph.main(pretty=True, workdirs=[output_folder])

def process_policy(policy_url, policy_kind):
    output_folder = "./output/gradio_result"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    try:
        if policy_kind == "pdf":
            pdf_parser.main(policy_url, output_folder)
            html_path = os.path.join(output_folder, "output.html")
            generate_graph_from_html(html_path, output_folder)
        else:
            generate_graph_from_html(policy_url, output_folder)
        # Find the .graphml file to return as output
        graphml_files = glob.glob(os.path.join(output_folder, "*.graphml"))
        if graphml_files:
            return f"Graph generated: {graphml_files[0]}"
        else:
            return "Graph generation completed, but no .graphml file found."
    except Exception as e:
        return f"Error: {e}"

def main():
    iface = gr.Interface(
        fn=process_policy,
        inputs=[
            gr.Textbox(label="Policy URL"),
            gr.Radio(choices=["pdf", "webpage", "auto"], label="Policy Kind"),
        ],
        outputs="text",
        title="PoliGraph-er Demo",
        description="Enter a privacy policy URL and select the type to generate a knowledge graph."
    )
    iface.launch()

if __name__ == "__main__":
    main()
