import os

from bs4 import BeautifulSoup
from poligrapher.gradio_app.database import submit_url_to_db

from poligrapher.analysis.privacy_scorer import PrivacyScorer
from poligrapher.scripts import build_graph, html_crawler, init_document, pdf_parser, run_annotators


# Helper to generate graph from HTML or output.html
def generate_graph(input_file, output_folder):
    html_crawler.main(input_file, output_folder)
    init_document.main(workdirs=[output_folder])
    run_annotators.main(workdirs=[output_folder])
    build_graph.main(workdirs=[output_folder])
    build_graph.main(pretty=True, workdirs=[output_folder])


def extract_policy_text(input_folder: str) -> str:
    """Extract privacy policy text from HTML files in the input folder"""
    text = ""
    for filename in os.listdir(input_folder):
        if filename.endswith(".html"):
            with open(os.path.join(input_folder, filename), "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
                for tag in soup.find_all(["p", "li", "h1", "h2", "h3", "h4", "div"]):
                    text += tag.get_text() + "\n"
    return text


def run_poligrapher(policy_url: str, output_folder: str, policy_kind: str = "auto"):
    try:
        # Send and save URL to database
        submit_url_to_db(policy_url)

        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Process based on policy kind
        if policy_kind == "pdf":
            graphml_path =  run_poligrapher_pdf(policy_url, output_folder)
        else:
            graphml_path = run_poligrapher_html(policy_url, output_folder)

        # Extract policy text
        policy_text = extract_policy_text(policy_url)
        if not policy_text:
            raise ValueError("Could not extract text from the provided URL")

        # Score the policy
        scorer = PrivacyScorer()
        results = scorer.score_policy(policy_text)

        # Ensure results are in the expected format
        return {
            "total_score": float(results["total_score"]),
            "grade": results["grade"],
            "category_scores": results["category_scores"],
            "feedback": results.get("feedback", []),
            "poligraph": graphml_path,  # path to graphml file
        }

    except Exception as e:
        e.add_note("An error occurred while analyzing the privacy policy.")
        raise e


def run_poligrapher_html(policy_url: str, output_folder: str):
    generate_graph(policy_url, output_folder)
    return os.path.join(output_folder, "graph-original.graphml")


def run_poligrapher_pdf(policy_url: str, output_folder: str):
    pdf_parser.main(policy_url, output_folder)
    html_path = os.path.join(output_folder, "output.html")
    generate_graph(html_path, output_folder)
    return os.path.join(output_folder, "graph-original.graphml")
