from typing import Optional, List, Tuple, Dict
import psycopg2
from psycopg2.extensions import connection
import json
import os

# The contents of this file are currently not in use, as we are temporarily saving data to local storage.
# TODO: This will be re-integrated with a proper database soon

# Database connection parameters
db_params = {
    "dbname": "privacy_analysis",
    "user": "postgres",
    "password": "@Diamond08",
    "host": "localhost",
    "port": "5432",
}


def get_db_connection() -> Optional[connection]:
    try:
        return psycopg2.connect(**db_params)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None


def submit_url_to_db(url):
    try:
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO healthcare_privacy_policies (url) VALUES (%s);", (url,)
        )

        conn.commit()
        cursor.close()
        conn.close()
        print(f"Inserted URL: {url}")

    except Exception as e:
        print("Database error:", e)


DATA_PATH = os.path.join("data", "policies.json")


# temporary, save to local storage
def save_analysis_to_localstorage(url, total_score, grade, category_scores, feedback):
    if not os.path.exists(DATA_PATH):
        with open(DATA_PATH, "w") as f:
            json.dump({}, f)

    with open(DATA_PATH, "r") as f:
        db = json.load(f)

    db[url] = {
        "total_score": total_score,
        "grade": grade,
        "category_scores": category_scores,
        "feedback": feedback,
    }

    with open(DATA_PATH, "w") as f:
        json.dump(db, f, indent=2)

    print(f"Saved analysis for: {url}")


# temporary fetch from local storage
def fetch_analysis_from_localstorage(url):
    if not os.path.exists(DATA_PATH):
        raise ValueError("No saved data found.")

    with open(DATA_PATH, "r") as f:
        db = json.load(f)

    if url not in db:
        raise ValueError("Policy not found.")

    data = db[url]

    return {
        "total_score": data["total_score"],
        "grade": data["grade"],
        "category_scores": data["category_scores"],
        "feedback": data["feedback"],
        "graph_json_path": None,
    }


# def to retrieve top 10 highest graded policy name and grade#def gettopten():

# def to retrieve top 10 lowest graded policy names and grades

# def to retrieve policy name, summary, grade,m and all other inform...
