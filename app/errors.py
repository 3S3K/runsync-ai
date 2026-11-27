"""API가 돌려주는 업무 에러."""


class RecommendError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def out_of_service_area() -> RecommendError:
    return RecommendError(400, "OUT_OF_SERVICE_AREA", "서비스 지역이 아닙니다. 현재는 대전 동구에서만 추천할 수 있어요.")


def course_not_found() -> RecommendError:
    return RecommendError(404, "COURSE_NOT_FOUND", "이 위치에서는 조건에 맞는 코스를 만들지 못했어요.")


def invalid_internal_key() -> RecommendError:
    return RecommendError(401, "INVALID_INTERNAL_KEY", "내부 호출 키가 없거나 올바르지 않습니다.")


def training_disabled() -> RecommendError:
    return RecommendError(503, "TRAINING_DISABLED", "서버에 내부 호출 키가 설정되지 않아 재학습 API를 쓸 수 없습니다.")
