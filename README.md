# LLM Hallucination & Factual Consistency Detector

A high-performance Python framework engineered for detecting unsupported claims, measuring grounding ratios, and evaluating factual consistency in production LLM outputs.

## Architecture & Core Features
- **Semantic Entailment Heuristics:** Automated token-level comparison between source context and model responses to calculate hallucination scores.
- **Batch Evaluation & Metrics:** Computes grounding ratios, hallucination rates, and detailed anomaly logs at scale.
- **RESTful Detection API:** Built with FastAPI and Pydantic for high-throughput asynchronous evaluation runs.

## Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python main.py
