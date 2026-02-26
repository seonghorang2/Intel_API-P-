from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import random 
import httpx
import os                         # 환경변수 접근을 위한 모듈
from dotenv import load_dotenv    # .env 파일 로딩 라이브러리
import joblib
from schemas import SalesInput, SalesOutput  # pydantic 모델 import
import pandas as pd

# -------------------
# 모델 로드
# -------------------
model = joblib.load('ml/ad.pkl')
ad_model = model['model']
print('✅모델 로드 완료!')
print(f"저장 당시 sklearn 버전: {model['sklearn_version']}")


# .env 파일의 환경변수를 메모리에 로드
# → 이후 os.getenv()로 값을 읽을 수 있게 됩니다.
load_dotenv()   

# ──────────────────────────────────────────────
# 1) FastAPI 앱 인스턴스 생성
# ──────────────────────────────────────────────
# FastAPI() 를 호출하면 웹 애플리케이션 객체가 만들어집니다.
# 이 app 객체에 API 경로(라우트)를 등록하고, 서버를 실행합니다
app = FastAPI()


# ──────────────────────────────────────────────
# 2) CORS 설정
# ──────────────────────────────────────────────
# CORS(Cross-Origin Resource Sharing)란?
# → 브라우저의 보안 정책으로, 다른 출처(도메인, 포트)에서 오는 요청을 기본적으로 차단합니다.
# → React(포트 5173)에서 FastAPI(포트 8000)로 요청하면 "출처가 다르다"고 판단하여 차단됩니다.
# → 이 설정을 통해 특정 프론트엔드 주소에서 오는 요청을 허용합니다.


# 허용할 프론트엔드 주소 목록
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,           # 위에서 지정한 주소만 허용
    allow_credentials=True,
    allow_methods=["*"],             # 모든 HTTP 메서드(GET, POST 등) 허용
    allow_headers=["*"],             # 모든 헤더 허용
)

# ──────────────────────────────────────────────
# 3) API 엔드포인트(라우트) 정의
# ──────────────────────────────────────────────
# "/" 경로로 GET 요청이 들어오면 아래 함수를 실행
@app.get("/")
def home():
    return {"message":"여기는 home입니다"}

# "/about" 경로로 GET 요청이 들어오면 실행
@app.get("/about")
def about():
    return {
        "name": "김철수",
        "phone": "010-123-4567",
        "address": "서울시 종로구"
    }   

# 랜덤 명언
@app.get("/quote")
def random_quote():
    quotes = [
        "성공은 준비된 자에게 찾아온다.",
        "노력은 배신하지 않는다.",
        "오늘 걷지 않으면 내일은 뛰어야 한다."
    ]
    return {"quote":random.choice(quotes)}

# 랜덤 고양이 사진 API
@app.get("/random_cat")
def get_random_cat():

    # 외부 API 주소
    url = "https://api.thecatapi.com/v1/images/search?limit=6"
    
    # httpx.get()으로 외부 API에 GET 요청을 보냄
    response = httpx.get(url)
    
    # 응답 상태 코드가 200이 아니면 에러 반환
    if response.status_code != 200:
        return {"error": "고양이 API 요청 실패"}
    
    # JSON 형식의 응답을 파이썬 리스트로 변환
    cats = response.json()

    # 랜덤 고양이 선정
    random_cat = random.choice(cats)

    # 랜덤 고양이 데이터 리턴
    return random_cat


@app.get("/festivals")
def get_festivals():      
    # 환경변수에서 공공데이터 API 서비스 키 가져오기
    service_key = os.getenv("API_SERVICE_KEY")
    
    # 공공데이터 문화축제 API URL
    url = 'http://api.data.go.kr/openapi/tn_pubr_public_cltur_fstvl_api'
    
    # API 요청 파라미터 설정
    params = {
        'serviceKey': service_key,  # 인증 키
        'pageNo': '1',              # 페이지 번호
        'numOfRows': '100',         # 한 페이지 결과 수
        'type': 'json'              # 응답 형식
    }
    # API 호출
    response = httpx.get(url, params=params)

    # 응답 상태 코드가 200이 아니면 에러 반환
    if response.status_code != 200:
        return {"error": "축제 API 요청 실패"}

    # JSON 형식의 데이터를 파이썬 리스트로 변환
    data = response.json()    

    # 축제 목록 데이터 반환
    return data["response"]["body"]["items"]
    
@app.post("/sales_predict" , response_model=SalesOutput)
def sales_predict(data: SalesInput): 
    
    # 모델에 입력할 데이터 준비    
    features = [[data.tv, data.radio, data.newspaper ]]        
    X = pd.DataFrame(features, columns=['TV', 'Radio', 'Newspaper'])
    
    # 예측 수행
    prediction = ad_model.predict(X)[0]

    # 결과 반환
    return SalesOutput(predicted_sales=round(float(prediction), 2))