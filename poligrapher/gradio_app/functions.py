from dataclasses import dataclass

# Structured data for privacy policy analysis
@dataclass
class PolicyAnalysisResult:
    company_name: str
    privacy_policy_url: str
    score: float
    kind: str  # 'pdf', 'webpage', or 'auto'
    has_name: bool = False
    has_score: bool = False

import ipaddress
import json
import os
import urllib.parse

import httpx
import networkx as nx

from poligrapher.gradio_app.poligrapher_functions import run_poligrapher


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


def process_policy_url(policy: PolicyAnalysisResult):
    validation_result = validate_url(policy.privacy_policy_url)
    if not validation_result["valid"]:
        return {"success": False, "message": validation_result["message"]}

    print(f"Processing policy URL: {policy.privacy_policy_url}")

    try:
        # check if folder exists, if not create it
        output_folder = "../../PoliGraph-Setup/output/" + policy.company_name.replace(" ", "_")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        result = run_poligrapher(policy.privacy_policy_url, output_folder=output_folder)
        if not result:
            return {"success": False, "message": "Failed to analyze privacy policy"}

        total_score = result["total_score"]
        grade = result["grade"]
        category_scores = result["category_scores"]
        feedback = result["feedback"]
        graphml_path = result.get("poligraph")

        graph_json_path = graphml_to_json(graphml_path) if graphml_path else None

        # Update the policy object with score and flags
        policy.score = total_score
        policy.has_score = total_score is not None
        policy.has_name = bool(policy.company_name and policy.company_name.strip())

        return {
            "success": True,
            "message": "Analysis complete",
            "result": {
                "total_score": total_score,
                "grade": grade,
                "category_scores": category_scores,
                "feedback": feedback,
                "graph_json_path": graph_json_path,
                "structured": policy,
            },
        }
    except Exception as e:
        return {"success": False, "message": f"Error processing policy: {str(e)}"}


# get the policy info for top and lowest graded polciies and most recently submitted
def get_policy_info():

    ##get data from the database
    ##data NEEDED: date, grade, and policy name, url

    # fetch 10 highest graded policies from the database
    # using fake data for now
    top_policies = [
        ("Privacy Policy A", 98, "https://example.com/privacy_policy_a"),
        ("Privacy Policy B", 94, "https://example.com/privacy_policy_b"),
        ("Privacy Policy C", 91, "https://example.com/privacy_policy_c"),
        ("Privacy Policy D", 91, "https://example.com/privacy_policy_d"),
        ("Privacy Policy E", 90, "https://example.com/privacy_policy_e"),
    ]

    # fetch the 10 lowest graded policies from the database
    # using fake data for now
    low_policies = [
        ("Privacy Policy X", 15, "https://example.com/privacy_policy_x"),
        ("Privacy Policy Y", 17, "https://example.com/privacy_policy_y"),
        ("Privacy Policy Z", 24, "https://example.com/privacy_policy_z"),
        ("Privacy Policy W", 26, "https://example.com/privacy_policy_w"),
        ("Privacy Policy V", 29, "https://example.com/privacy_policy_v"),
    ]

    # fetch the 10 most recently submitted privacy polciies from the database
    # using fake data for now
    recent_policies = [
        ("Privacy Policy M", 82, "https://example.com/privacy_policy_m"),
        ("Privacy Policy N", 80, "https://example.com/privacy_policy_n"),
        ("Privacy Policy O", 79, "https://example.com/privacy_policy_o"),
        ("Privacy Policy P", 85, "https://example.com/privacy_policy_p"),
        ("Privacy Policy Q", 73, "https://example.com/privacy_policy_q"),
    ]

    return top_policies, low_policies, recent_policies


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
    with open(graph_json_path, "w") as f:
        json.dump(elements, f)

    # Return web path
    web_path = f"/static/graphml/json/{graph_json_filename}"
    return web_path


def search_policy_info(query):

    # Simulated database query
    policies = [
        ("Privacy Policy A", 98, "https://example.com/privacy_policy_a"),
        ("Privacy Policy B", 94, "https://example.com/privacy_policy_b"),
        ("Privacy Policy M", 82, "https://example.com/privacy_policy_m"),
        ("Privacy Policy X", 15, "https://example.com/privacy_policy_x"),
    ]

    return [policy for policy in policies if query.lower() in policy[0].lower()]


def fetch_analysis_from_db(url):
    print(f"Returning hardcoded mock analysis for URL: {url}")

    total_score = 92.1
    grade = "A"
    category_scores = {
        "data_collection": {
            "raw_score": 220,
            "weighted_score": 22.0,
            "feedback": ["Clear data collection purpose", "Minimal data collected"],
        },
        "third_party_sharing": {
            "raw_score": 250,
            "weighted_score": 25.0,
            "feedback": ["Transparent third-party disclosures"],
        },
        "user_rights": {
            "raw_score": 230,
            "weighted_score": 23.0,
            "feedback": ["Users can delete data", "Clear opt-out mechanisms"],
        },
        "data_security": {
            "raw_score": 220,
            "weighted_score": 22.1,
            "feedback": ["Encryption noted", "Breach policy documented"],
        },
    }

    feedback = [
        item for section in category_scores.values() for item in section["feedback"]
    ]

    return {
        "total_score": total_score,
        "grade": grade,
        "category_scores": category_scores,
        "feedback": feedback,
        "graph_json_path": "/static/graphml/json/akili.json",  # Shows the PoliGraph
    }


def is_policy_already_analyzed(url):
    # Replace with actual database query
    # For now, simulate some existing policies
    existing_policies = [
        "https://example.com/privacy_policy_a",
        "https://example.com/privacy_policy_b",
        "https://example.com/privacy_policy_m",
    ]
    return url in existing_policies


def save_analysis_to_db(url, total_score, grade, category_scores, feedback):
    # Replace with actual database insert
    print(f"Saving analysis for URL: {url}")
    print(f"Total Score: {total_score}")
    print(f"Grade: {grade}")
    print(f"Category Scores: {category_scores}")
    print(f"Feedback: {feedback}")

    # Actually need to save

    return 1


def render_analysis_output(
    *, url=None, pdf=None, total_score=None, grade=None, category_scores=None
):
    try:
        normalized_score = (total_score / 100.0) * 10.0

        summary_html = f"""
        <div class='summary-section'>
            <h3>Total Score: {normalized_score:.1f}</h3>
            <h3>Grade: {grade}</h3>
            <p><strong>Feedback:</strong> {', '.join([f for cat in category_scores.values() for f in cat['feedback']]) or 'No feedback available'}</p>
        </div>
        """

        accordion_html = ""
        for category, scores in category_scores.items():
            feedback = "<br>".join(scores["feedback"]) or "None"
            accordion_html += f"""
            <details>
                <summary><strong>{category.replace('_', ' ').title()}</strong></summary>
                <p>Raw Score: {scores['raw_score']}</p>
                <p>Weighted Score: {scores['weighted_score']}</p>
                <p>Feedback: {feedback}</p>
            </details>
            """

        return summary_html, accordion_html

    except Exception as e:
        print(f"Error in render_analysis_output: {str(e)}")
        return (
            "<div>Error generating summary</div>",
            "<div>Error generating details</div>",
        )
