import pytest

from app import config, model, train


@pytest.fixture(scope="session", autouse=True)
def trained_model(tmp_path_factory):
    """테스트용 모델을 임시 폴더에 학습해 둔다. 저장소의 models/ 파일에 의존하지 않는다."""
    path = str(tmp_path_factory.mktemp("models") / "hr_model.joblib")
    train.train("ridge", n_rows=3000, seed=42, out_path=path)
    config.MODEL_PATH = path
    model.load()
    return path
