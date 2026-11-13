from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
from detector import HallucinationDetector

app = FastAPI(
    title="LLM Hallucination & Factual Consistency Detector",
    description="Production-grade API for detecting unsupported claims and measuring grounding ratios in LLM outputs.",
    version="1.0.0"
)

detector = HallucinationDetector(hallucination_threshold=0.3)

class HallucinationItem(BaseModel):
    context: str = Field(..., example="The company's net revenue for Q3 2024 was $12.5 million.")
    response: str = Field(..., example="Net revenue in Q3 2024 reached $12.5 million, marking significant growth.")

class BatchHallucinationRequest(BaseModel):
    samples: List[HallucinationItem]

@app.post("/v1/detect/hallucination", summary="Detect Hallucinations and Factual Inconsistencies")
def detect_hallucination_endpoint(payload: BatchHallucinationRequest):
    try:
        raw_samples = [{"context": item.context, "response": item.response} for item in payload.items]
        df_results = detector.evaluate_batch(raw_samples)
        
        hallucinated_count = int((df_results["is_hallucinated"] == True).sum())
        total_count = len(raw_samples)
        
        return {
            "status": "success",
            "summary": {
                "total_samples": total_count,
                "hallucinated_samples": hallucinated_count,
                "clean_samples": total_count - hallucinated_count,
                "hallucination_rate_pct": round((hallucinated_count / total_count) * 100, 2) if total_count > 0 else 0.0
            },
            "detection_records": df_results.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
