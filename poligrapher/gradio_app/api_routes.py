from poligrapher.gradio_app.localstoragedb import (
    fetch_analysis_from_localstorage,
    save_analysis_to_localstorage,
)
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from poligrapher.gradio_app.functions import (
    fetch_analysis_from_db,
    get_policy_info,
    is_policy_already_analyzed,
    process_policy_url,
    save_analysis_to_db,
    search_policy_info,
)
from poligrapher.gradio_app.validators import validate_url
from pydantic import BaseModel

class AnalyzeRequest(BaseModel):
    url: str

router = APIRouter()

@router.get("/policy-data")
async def get_policy_data():
    try:
        top_policies, low_policies, recent_policies = get_policy_info()
        return JSONResponse(content={
            "top_policies": top_policies,
            "low_policies": low_policies,
            "recent_policies": recent_policies
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@router.get("/search-policy")
async def search_policy(query: str = Query(...)):
    try:
        results = search_policy_info(query)
        return JSONResponse(content={"results": results})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@router.get("/fetch-analysis")
async def fetch_analysis(url: str = Query(...)):
    validation_result = validate_url(url)
    if not validation_result["valid"]:
        return JSONResponse(content={"error": validation_result["message"]}, status_code=400)

    try:
        #need to change back to DB later
        analysis_data = fetch_analysis_from_localstorage(url)
        return JSONResponse(content=analysis_data)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@router.post("/analyze-url")
async def analyze_url(request: AnalyzeRequest):
    url = request.url 

    validation_result = validate_url(url)
    if not validation_result["valid"]:
        return JSONResponse(content={"error": validation_result["message"]}, status_code=400)

    if is_policy_already_analyzed(url):
        print(f"Policy already analyzed for URL: {url}")
        return JSONResponse(content={"already_analyzed": True})

    try:
        analysis = process_policy_url(url)

        if not analysis["success"]:
            return JSONResponse(content={"error": analysis["message"]}, status_code=400)

        result = analysis["result"]
        #need to change back to DB later
        save_analysis_to_localstorage(
            url,
            result["total_score"],
            result["grade"],
            result["category_scores"],
            result["feedback"]
        )

        return JSONResponse(content={"success": {}})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
