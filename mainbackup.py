from pathlib import Path
import random

import httpx
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ml.schemas import SalesInput, SalesOutput  # pydantic 모델 import

# -------------------
# 모델 로드
# -------------------
BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "ml" / "ad.pkl")
ad_model = model["model"]

print("✅모델 로드 완료!")
print(f"저장 당시 sklearn 버전: {model['sklearn_version']}")

# -------------------
# FastAPI 앱 생성
# -------------------
app = FastAPI()

# -------------------
# CORS 설정
# -------------------
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------
# API 엔드포인트
# -------------------
@app.get("/")
def home():
    return {"message": "이 메세지를 변경하면 바뀜"}


@app.get("/animal")
async def random_animal():
    characteristics = ["귀여운", "용감한", "느긋한", "쏘 쿨한"]
    animals = ["고양이", "강아지", "여우"]

    selected_animal = random.choice(animals)
    selected_characteristic = random.choice(characteristics)

    image_url = None

    async with httpx.AsyncClient(timeout=8.0) as client:
        try:
            if selected_animal == "강아지":
                response = await client.get("https://dog.ceo/api/breeds/image/random")
                response.raise_for_status()
                image_url = response.json().get("message")

            elif selected_animal == "고양이":
                response = await client.get("https://api.thecatapi.com/v1/images/search?limit=1")
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, list) and payload:
                    image_url = payload[0].get("url")

            elif selected_animal == "여우":
                response = await client.get("https://randomfox.ca/floof/")
                response.raise_for_status()
                image_url = response.json().get("image")
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"동물 이미지 API 호출 실패: {e}") from e

    if not image_url:
        raise HTTPException(status_code=502, detail="동물 이미지 URL을 가져오지 못했습니다.")

    return {
        "message": f"{selected_characteristic} {selected_animal} 입니다!",
        "image_url": image_url,
    }


@app.post("/sales_predict", response_model=SalesOutput)
def sales_predict(data: SalesInput):
    features = [[data.tv, data.radio, data.newspaper]]
    X = pd.DataFrame(features, columns=["TV", "Radio", "Newspaper"])

    prediction = ad_model.predict(X)[0]
    return SalesOutput(predicted_sales=round(float(prediction), 2))
