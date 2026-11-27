from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app import config, detail, elevation, errors, explain, features, model, routing, scorer
from app.db import pool
from app.schemas import (
    Climb, CourseRes, ElevationPoint, ErrorRes, Geometry, RecommendReq, RecommendRes, SlopeSegment,
)


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

    # 고도 프로필과 상세 정보는 추천으로 뽑힌 코스에만 계산한다 (후보 16개 전부에 계산하면 낭비다)
    with pool.connection() as conn:
        profiles = [elevation.full_profile(conn, candidate) for candidate in selected]
        details = [detail.course_detail(conn, candidate) for candidate in selected]

    target_m = request.target_distance_km * 1000
    intensity = request.intensity.value
    age = request.user.age if request.user else None

    courses = []
    used_names = []
    for rank, (candidate, profile, info) in enumerate(zip(selected, profiles, details), start=1):
        name = detail.course_name(info.main_road, candidate.bearing_deg, used_names)
        used_names.append(name)
        climb = elevation.longest_climb(profile, config.CLIMB_DIP_TOLERANCE_M)
        has_climb = climb is not None and climb[2] >= config.MIN_CLIMB_GAIN_M
        courses.append(CourseRes(
            rank=rank,
            name=name,
            score=scorer.display_score(candidate, target_m, intensity),
            distance_km=round(candidate.distance_m / 1000, 2),
            # 누적 오르막과 예상 강도는 추천 이유 문장과 같은 정수로 내려 화면의 숫자가 어긋나지 않게 한다
            elevation_gain_m=round(candidate.ascent_m),
            # 화면용 최대 경사는 100m 이상 구간에서 잰다. 모델 입력(candidate.max_slope_pct)과는 다른 값이다
            max_slope_pct=round(elevation.max_slope_pct(profile, config.MAX_SLOPE_WINDOW_M), 1),
            predicted_hr_pct=round(candidate.predicted_hr_pct),
            intensity=request.intensity,
            reason=explain.build_reason(
                candidate, request.target_distance_km, intensity, request.recent_records,
                age=age, climb=climb),
            geometry=Geometry(coordinates=candidate.coordinates),
            elevation_profile=[
                ElevationPoint(distance_m=round(distance_m), elevation_m=round(elevation_m, 1))
                for distance_m, elevation_m in elevation.downsample(profile, config.ELEVATION_PROFILE_MAX_POINTS)
            ],
            estimated_time_min=explain.estimated_time_min(candidate.distance_m, request.recent_records),
            stairs_count=info.stairs_count,
            hr_zone=scorer.hr_zone(candidate.predicted_hr_pct),
            hr_bpm_range=explain.hr_bpm_range(candidate.predicted_hr_pct, age),
            longest_climb=Climb(
                start_m=round(climb[0]), length_m=round(climb[1]), gain_m=round(climb[2]),
            ) if has_climb else None,
            slope_segments=[
                SlopeSegment(grade=segment.grade, coordinates=segment.coordinates)
                for segment in info.slope_segments
            ],
        ))
    return RecommendRes(courses=courses)
