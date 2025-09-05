from database import get_db_connection

# The following functions are currently not in use
# TODO: Reassess the utility of these functions and remove if unnecessary.

def fetch_all_grades():
    """Return all rows from grade table"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM healthcare_privacy_grade;")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows
    except Exception as e:
        print("Error fetching grades:", e)
        return []


def get_policies_with_scores():  # michael
    """Return URL, score, and provider name"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT p.url, g.grade, p.healthcare_provider
            FROM healthcare_privacy_policies p
            JOIN healthcare_privacy_grade g ON p.id = g.policy_id
            WHERE g.grade IS NOT NULL;
        """
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows
    except Exception as e:
        print("Error getting scores:", e)
        return []


def get_analysis_data():  # michael
    """Return URL, grade, and feedback for analysis page"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT p.url, g.grade, g.feedback
            FROM healthcare_privacy_policies p
            JOIN healthcare_privacy_grade g ON p.id = g.policy_id
            WHERE g.grade IS NOT NULL;
        """
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows
    except Exception as e:
        print("Error getting analysis data:", e)
        return []
