from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.recommendation.ranking.inference import RecommendationPipeline


pipeline = None


class RecommendationRequest(BaseModel):
    observed_aids: List[int] = Field(
        ...,
        min_length=1,
        description="Observed item IDs in session order.",
    )

    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
    )


class RecommendationItem(BaseModel):
    rank: int
    aid: int
    score: float


class RecommendationResponse(BaseModel):
    observed_aids: List[int]
    recommendations: List[RecommendationItem]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline

    print("Loading recommendation pipeline...")

    pipeline = RecommendationPipeline()

    print("Recommendation pipeline loaded.")

    yield

    print("Shutting down recommendation service...")


app = FastAPI(
    title="OTTO Recommendation API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": pipeline is not None,
    }

# src/serving/app.py
@app.post(
    "/recommend",
    response_model=RecommendationResponse,
)
def recommend(
    request: RecommendationRequest,
):
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Recommendation pipeline is not loaded.",
        )

    try:
        recommendations = pipeline.recommend(
            observed_aids=request.observed_aids,
            top_k=request.top_k,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    return RecommendationResponse(
        observed_aids=request.observed_aids,
        recommendations=recommendations,
    )