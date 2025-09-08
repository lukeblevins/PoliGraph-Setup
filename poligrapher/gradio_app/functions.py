import ipaddress
import json
import os
import urllib.parse
import logging as logger
import httpx
from matplotlib import pyplot as plt
import networkx as nx
import yaml

from poligrapher.gradio_app.policy_analysis import (
    DocumentCaptureSource,
    PolicyDocumentInfo,
)
from poligrapher.analysis.privacy_scorer import PrivacyScorer

from poligrapher.scripts import (
    build_graph,
    html_crawler,
    pdf_parser,
    run_annotators,
    init_document,
)

logger = logger.getLogger(__name__)


def is_ip_address(s):
    try:
        return bool(ipaddress.ip_address(s))
    except ValueError:
        return False


def validate_url(url: str) -> dict:
    """Validate URL format and accessibility"""
    if not url or not url.strip():
        return {"valid": False, "message": "No URL provided"}

    url = url.strip()

    # Parse URL
    try:
        result = urllib.parse.urlparse(url)
        if not all([result.scheme, result.netloc]):
            return {"valid": False, "message": "Invalid URL format"}
    except Exception:
        return {"valid": False, "message": "Invalid URL format"}

    # Check for IP addresses
    hostname = result.netloc.split(":")[0]
    if is_ip_address(hostname):
        return {
            "valid": False,
            "message": "IP addresses not allowed. Please use domain names",
        }

    # Check accessibility
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; PrivacyPolicyAnalyzer/1.0)"}
        response = httpx.head(url, headers=headers, follow_redirects=True, timeout=10.0)
        if response.status_code == 405:  # Method not allowed
            response = httpx.get(
                url, headers=headers, follow_redirects=True, timeout=10.0
            )
        if not response.is_success:
            return {
                "valid": False,
                "message": f"URL not accessible (Status code: {response.status_code})",
            }
    except Exception as e:
        return {"valid": False, "message": f"Error accessing URL: {str(e)}"}

    return {"valid": True, "message": "URL is valid"}


def visualize_graph(policy: PolicyDocumentInfo):
    """
    Create and save a knowledge-graph PNG from a YAML export.

    Reads "graph-original.full.yml" in policy.output_dir, builds a graph
    from top-level "nodes" and "links", draws the graph,
    and writes "<output_dir>/knowledge_graph.png".

    Parameters
    ----------
    policy : PolicyDocumentInfo
        Must provide output_dir (path to directory with the YAML file).

    Returns
    -------
    str
        Path to the saved PNG.

    Raises
    ------
    FileNotFoundError
        If the YAML file is missing.
    yaml.YAMLError, KeyError, TypeError, ValueError, OSError
        On parse, data, drawing, or file I/O errors.

    Notes
    -----
    In headless environments set an appropriate matplotlib backend (e.g., "Agg").
    """
    output_folder = policy.output_dir
    yml_file = os.path.join(output_folder, "graph-original.full.yml")
    output_png = os.path.join(output_folder, "knowledge_graph.png")
    if not os.path.exists(yml_file):
        raise FileNotFoundError("YML file not found for visualization.")
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
    return output_png


def score_policy(policy: PolicyDocumentInfo):
    policy.has_results = False
    scorer = PrivacyScorer()
    results = scorer.score_policy(policy.get_document_text())
    if results.get("success") is True:
        policy.score = results.get("total_score")
        policy.has_results = True
    else:
        policy.score = None
        policy.has_results = False
    return results


def test_document_url(url: str) -> bool:
    """Test if the document URL is reachable with a 200 response.

    Notes:
    - Returns False silently for local file paths or values without http/https scheme.
    - Avoids emitting noisy errors for non-URL inputs (e.g., absolute file paths).
    """
    if not url:
        return False
    try:
        parsed = urllib.parse.urlparse(url)
        # Not an http(s) URL → treat as non-URL; return False without logging
        if parsed.scheme not in ("http", "https"):
            return False
    except Exception:
        return False

    try:
        response = httpx.head(url, follow_redirects=True, timeout=10.0)
        if response.status_code == 405:  # Method not allowed
            response = httpx.get(url, follow_redirects=True, timeout=10.0)
        return response.status_code == 200
    except Exception as e:
        logger.error("Error accessing URL %s: %s", url, str(e))
        return False


def generate_graph_from_html(path, output_folder, capture_pdf: bool):
    # Normalize file:// URIs to filesystem paths
    try:
        parsed = urllib.parse.urlparse(path)
        if parsed.scheme == "file":
            path = parsed.path
    except Exception:
        pass
    # Prefer local file check first to avoid URL probes on file paths
    if os.path.isfile(path):
        pass
    else:
        if test_document_url(path) is False:
            raise FileNotFoundError(
                f"Document is not accessible or does not exist: {path}"
            )
    if capture_pdf:
        pdf_parser.main(path, output_folder)
        html_path = os.path.join(output_folder, "output.html")
        html_crawler.main(html_path, output_folder)
    else:
        # 1. Crawl / ingest HTML (produces accessibility_tree.json)
        html_crawler.main(path, output_folder)
    # 2. Initialize document (creates document.pickle expected by later stages)
    init_document.main(workdirs=[output_folder])
    # 3. Run annotators to populate token relationships
    run_annotators.main(workdirs=[output_folder])
    # 4. Build graph (regular + pretty)
    build_graph.main(workdirs=[output_folder])
    build_graph.main(pretty=True, workdirs=[output_folder])


def generate_graph(policy: PolicyDocumentInfo):
    """Run the full PoliGraph pipeline for a single policy."""
    match policy.source:
        case DocumentCaptureSource.WEBPAGE:
            capture_pdf = False
        case DocumentCaptureSource.PDF:
            capture_pdf = True
        case _:
            raise ValueError(f"Unknown document source: {policy.source}")

    generate_graph_from_html(policy.path, policy.output_dir, capture_pdf)
    return True

# def score_existing_policy(policy: PolicyDocumentInfo):
#     """Score an existing policy without regenerating the knowledge graph.

#     Assumes the graph/html artifacts already exist in the standard output folder.
#     """
#     try:
#         # output_folder_variants = [
#         #     os.path.join(
#         #         "../../PoliGraph-Setup/output", policy.company_name.replace(" ", "_")
#         #     ),
#         #     os.path.join("./output", policy.company_name.replace(" ", "_")),
#         # ]
#         output_folder = policy.output_dir
#         # for folder in output_folder_variants:
#         #     if os.path.isdir(folder):
#         #         output_folder = folder
#         #         break
#         # if output_folder is None:
#         #     return {"success": False, "message": "Existing output folder not found"}

#         # Collect text from any html files present
#         policy_text = ""
#         for fname in os.listdir(output_folder):
#             if fname.endswith(".html"):
#                 fpath = os.path.join(output_folder, fname)
#                 try:
#                     with open(fpath, "r", encoding="utf-8") as f:
#                         soup = BeautifulSoup(f.read(), "html.parser")
#                         for tag in soup.find_all(
#                             ["p", "li", "h1", "h2", "h3", "h4", "div"]
#                         ):
#                             policy_text += tag.get_text() + "\n"
#                 except Exception:
#                     continue

#         if not policy_text:
#             # Fallback: attempt using extract_policy_text helper (expects folder)
#             try:
#                 policy_text = extract_policy_text(output_folder)
#             except Exception:
#                 pass

#         if not policy_text:
#             return {"success": False, "message": "No existing HTML text found to score"}

#         scorer = PrivacyScorer()
#         results = scorer.score_policy(policy_text)

#         # Locate existing graphml if present
#         graphml_path = None
#         for candidate in [
#             os.path.join(output_folder, "graph-original.graphml"),
#             os.path.join(output_folder, "graph.graphml"),
#         ]:
#             if os.path.exists(candidate):
#                 graphml_path = candidate
#                 break

#         graph_json_path = graphml_to_json(graphml_path) if graphml_path else None

#         # Update policy flags
#         policy.has_results = policy.score is not None

#         return {
#             "success": True,
#             "message": "Scoring complete (existing graph reused)",
#             "result": {
#                 "total_score": results.get("total_score"),
#                 "grade": results.get("grade"),
#                 "category_scores": results.get("category_scores"),
#                 "feedback": results.get("feedback", []),
#                 "graph_json_path": graph_json_path,
#                 "structured": policy,
#             },
#         }
#     except Exception as e:
#         return {"success": False, "message": f"Error scoring existing policy: {str(e)}"}

def graphml_to_json(graphml_path):
    # Sanitize filename
    base_filename = os.path.splitext(os.path.basename(graphml_path))[0]
    graph_json_filename = f"{base_filename}.json"
    graph_json_path = os.path.join("./static/graphml/json", graph_json_filename)

    # Convert GraphML to JSON
    G = nx.read_graphml(graphml_path)
    nodes = [{"data": {"id": node, "label": node}} for node in G.nodes()]
    edges = [
        {
            "data": {
                "source": source,
                "target": target,
                "label": G.get_edge_data(source, target).get("label", "") or "",
            }
        }
        for source, target in G.edges()
    ]
    elements = nodes + edges

    # Save JSON
    os.makedirs(os.path.dirname(graph_json_path), exist_ok=True)
    with open(graph_json_path, "w", encoding="utf-8") as f:
        json.dump(elements, f)

    # Return web path
    web_path = f"/static/graphml/json/{graph_json_filename}"
    return web_path


# def search_policy_info(query):
#     # Simulated database query
#     policies = [
#         ("Privacy Policy A", 98, "https://example.com/privacy_policy_a"),
#         ("Privacy Policy B", 94, "https://example.com/privacy_policy_b"),
#         ("Privacy Policy M", 82, "https://example.com/privacy_policy_m"),
#         ("Privacy Policy X", 15, "https://example.com/privacy_policy_x"),
#     ]

#     return [policy for policy in policies if query.lower() in policy[0].lower()]


# def fetch_analysis_from_db(url):
#     print(f"Returning hardcoded mock analysis for URL: {url}")

#     total_score = 92.1
#     grade = "A"
#     category_scores = {
#         "data_collection": {
#             "raw_score": 220,
#             "weighted_score": 22.0,
#             "feedback": ["Clear data collection purpose", "Minimal data collected"],
#         },
#         "third_party_sharing": {
#             "raw_score": 250,
#             "weighted_score": 25.0,
#             "feedback": ["Transparent third-party disclosures"],
#         },
#         "user_rights": {
#             "raw_score": 230,
#             "weighted_score": 23.0,
#             "feedback": ["Users can delete data", "Clear opt-out mechanisms"],
#         },
#         "data_security": {
#             "raw_score": 220,
#             "weighted_score": 22.1,
#             "feedback": ["Encryption noted", "Breach policy documented"],
#         },
#     }

#     feedback = [
#         item for section in category_scores.values() for item in section["feedback"]
#     ]

#     return {
#         "total_score": total_score,
#         "grade": grade,
#         "category_scores": category_scores,
#         "feedback": feedback,
#         "graph_json_path": "/static/graphml/json/akili.json",  # Shows the PoliGraph
#     }


# def render_analysis_output(
#     *, url=None, pdf=None, total_score=None, grade=None, category_scores=None
# ):
#     try:
#         normalized_score = (total_score / 100.0) * 10.0

#         summary_html = f"""
#         <div class='summary-section'>
#             <h3>Total Score: {normalized_score:.1f}</h3>
#             <h3>Grade: {grade}</h3>
#             <p><strong>Feedback:</strong> {', '.join([f for cat in category_scores.values() for f in cat['feedback']]) or 'No feedback available'}</p>
#         </div>
#         """

#         accordion_html = ""
#         for category, scores in category_scores.items():
#             feedback = "<br>".join(scores["feedback"]) or "None"
#             accordion_html += f"""
#             <details>
#                 <summary><strong>{category.replace('_', ' ').title()}</strong></summary>
#                 <p>Raw Score: {scores['raw_score']}</p>
#                 <p>Weighted Score: {scores['weighted_score']}</p>
#                 <p>Feedback: {feedback}</p>
#             </details>
#             """

#         return summary_html, accordion_html

#     except Exception as e:
#         print(f"Error in render_analysis_output: {str(e)}")
#         return (
#             "<div>Error generating summary</div>",
#             "<div>Error generating details</div>",
#         )

# def fetch_policy_data():
#     try:
#         top_policies, low_policies, recent_policies = get_policy_info()
#         return {
#             "top_policies": top_policies,
#             "low_policies": low_policies,
#             "recent_policies": recent_policies,
#         }
#     except Exception as e:
#         return {"error": str(e)}
