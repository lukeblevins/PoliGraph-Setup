import os
import json

LOCAL_STORAGE_PATH = "data/policy_results.json"

##both functions need to be converted to the real database

def save_analysis_to_localstorage(url, total_score, grade, category_scores, feedback):
    os.makedirs("data", exist_ok=True)
    if os.path.exists(LOCAL_STORAGE_PATH):
        with open(LOCAL_STORAGE_PATH, "r") as f:
            db = json.load(f)
    else:
        db = {}

    db[url] = {
        "total_score": total_score,
        "grade": grade,
        "category_scores": category_scores,
        "feedback": feedback
    }

    with open(LOCAL_STORAGE_PATH, "w") as f:
        json.dump(db, f, indent=2)
    print(f"[LocalStorage] Saved: {url}")

def fetch_analysis_from_localstorage(url):
    if not os.path.exists(LOCAL_STORAGE_PATH):
        raise FileNotFoundError("No stored analysis data.")

    with open(LOCAL_STORAGE_PATH, "r") as f:
        db = json.load(f)

    if url not in db:
        raise ValueError("Policy not found in local storage.")

    data = db[url]
    return {
        "total_score": data["total_score"],
        "grade": data["grade"],
        "category_scores": data["category_scores"],
        "feedback": data["feedback"],
        "graph_json_path": "/static/graphml/json/akili.json"  # or dynamic if supported
    }
