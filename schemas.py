from pydantic import BaseModel

# 요청 데이터 모델 (클라이언트 → 서버)
class SalesInput(BaseModel):
    tv: float        # TV 광고비
    radio: float     # Radio 광고비
    newspaper: float # Newspaper 광고비

# 응답 데이터 모델 (서버 → 클라이언트)
class SalesOutput(BaseModel):
    predicted_sales: float  # 예측 판매량