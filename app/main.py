from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app import config, errors, explain, features, model, routing, scorer
from app.db import pool
from app.schemas import CourseRes, ErrorRes, Geometry, RecommendReq, RecommendRes


@asynccontextmanager
async def lifespan(_: FastAPI):
    model.load()
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

    if candidates:
        user_part = features.user_features(request.user, request.recent_records)
        rows = [
            features.build_row(user_part, features.course_features(
                c.distance_m / 1000, c.climb_m_per_km, c.max_slope_pct))
            for c in candidates
        ]
        for candidate, predicted in zip(candidates, model.predict_hr_pct(rows)):
            candidate.predicted_hr_pct = predicted

    selected = scorer.select_top(candidates, request.target_distance_km * 1000, request.intensity.value)
    if not selected:
        raise errors.course_not_found()

    courses = [
        CourseRes(
            rank=rank,
            distance_km=round(candidate.distance_m / 1000, 2),
            # 누적 오르막과 예상 강도는 추천 이유 문장과 같은 정수로 내려 화면의 숫자가 어긋나지 않게 한다
            elevation_gain_m=round(candidate.ascent_m),
            max_slope_pct=round(candidate.max_slope_pct, 1),
            predicted_hr_pct=round(candidate.predicted_hr_pct),
            intensity=request.intensity,
            reason=explain.build_reason(
                candidate, request.target_distance_km, request.intensity.value, request.recent_records,
                age=request.user.age if request.user else None),
            geometry=Geometry(coordinates=candidate.coordinates),
        )
        for rank, candidate in enumerate(selected, start=1)
    ]
    return RecommendRes(courses=courses)
