from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app import config, errors, explain, routing, scorer
from app.db import pool
from app.schemas import CourseRes, ErrorRes, Geometry, RecommendReq, RecommendRes


@asynccontextmanager
async def lifespan(_: FastAPI):
    pool.open()
    yield
    pool.close()


app = FastAPI(title="RunSync AI 코스 추천", lifespan=lifespan)


@app.exception_handler(errors.RecommendError)
async def handle_recommend_error(_: Request, exc: errors.RecommendError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"code": exc.code, "message": exc.message})


@app.get("/health")
def get_health() -> dict:
    return {"status": "ok"}


@app.post(
    "/recommend",
    response_model=RecommendRes,
    summary="순환 코스 추천",
    responses={400: {"model": ErrorRes}, 404: {"model": ErrorRes}},
)
def create_recommendation(request: RecommendReq) -> RecommendRes:
    lat, lng = request.start.lat, request.start.lng
    if not routing.is_in_service_area(lat, lng):
        raise errors.out_of_service_area()

    with pool.connection() as conn:
        start_node, snap_distance_m = routing.nearest_node(conn, lat, lng)
        if snap_distance_m > config.MAX_SNAP_DISTANCE_M:
            raise errors.out_of_service_area()
        candidates = routing.generate_candidates(
            conn, start_node, lat, lng, request.target_distance_km, request.intensity.value)

    selected = scorer.select_top(candidates, request.target_distance_km * 1000, request.intensity.value)
    if not selected:
        raise errors.course_not_found()

    courses = [
        CourseRes(
            rank=rank,
            distance_km=round(candidate.distance_m / 1000, 2),
            elevation_gain_m=round(candidate.ascent_m, 1),
            max_slope_pct=round(candidate.max_slope_pct, 1),
            intensity=request.intensity,
            reason=explain.build_reason(
                candidate, request.target_distance_km, request.intensity.value, request.recent_records),
            geometry=Geometry(coordinates=candidate.coordinates),
        )
        for rank, candidate in enumerate(selected, start=1)
    ]
    return RecommendRes(courses=courses)
