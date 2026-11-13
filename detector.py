import pandas as pd
from typing import List, Dict, Any

class HallucinationDetector:
    def __init__(self, hallucination_threshold: float = 0.3):
        self.hallucination_threshold = hallucination_threshold

    def detect_hallucination(self, source_context: str, model_response: str) -> Dict[str, Any]:
        """Analyzes factual consistency and detects unsupported claims (hallucinations) in model outputs."""
        context_words = set(source_context.lower().split())
        response_words = set(model_response.lower().split())

        if not response_words:
            return {"status": "error", "message": "Model response is empty."}

        # Calculate ungrounded / unsupported words ratio
        unsupported_words = response_words - context_words
        hallucination_score = round(len(unsupported_words) / max(len(response_words), 1), 2)
        
        is_hallucinated = hallucination_score >= self.hallucination_threshold

        return {
            "is_hallucinated": is_hallucinated,
            "hallucination_score": hallucination_score,
            "threshold": self.hallucination_threshold,
            "metrics": {
                "total_response_words": len(response_words),
                "unsupported_words_count": len(unsupported_words),
                "grounding_ratio": round(1.0 - hallucination_score, 2)
            },
            "analysis_details": "Response contains unsupported statements exceeding threshold." if is_hallucinated else "Response is well-grounded in source context."
        }

    def evaluate_batch(self, samples: List[Dict[str, str]]) -> pd.DataFrame:
        results = []
        for idx, sample in enumerate(samples):
            ctx = sample.get("context", "")
            resp = sample.get("response", "")
            
            res = self.detect_hallucination(ctx, resp)
            
            results.append({
                "sample_id": f"HAL-{idx+1:03d}",
                "is_hallucinated": res.get("is_hallucinated", False),
                "hallucination_score": res.get("hallucination_score", 0.0),
                "grounding_ratio": res.get("metrics", {}).get("grounding_ratio", 0.0)
            })
            
        return pd.DataFrame(results)
