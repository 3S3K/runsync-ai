FROM python:3.11-slim

WORKDIR /app

# LightGBM 실행에 필요한 시스템 라이브러리 (slim 이미지에는 없음)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 심박 예측 모델을 이미지 안에서 학습 → models/hr_model.joblib 생성
RUN python -m app.train

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
