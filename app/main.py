from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from typing import List
import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

from .models.address import (
    AddressRequest,
    Address,
    AddressResponse,
    AddressValidationRequest,
    AddressValidationResponse,
    TrainingData,
    TrainingResponse,
)
from .services.ml_model import address_model
from .services.db_service import db_service
from .services.train_service import ModelTrainer
from .utils.preprocess import validate_address_pattern

# FastAPI 앱 생성
app = FastAPI(
    title="주소 추출 API",
    description="텍스트에서 주소를 추출하는 NER 모델 API",
    version="1.0.0",
)

# 모델 학습 스케줄러
scheduler = None
trainer = ModelTrainer()


@app.on_event("startup")
async def startup_event():
    """앱 시작 시 실행되는 이벤트"""
    # 데이터베이스 연결 풀 초기화
    await db_service.init_pool()

    # 모델 로드
    await address_model.load_model()

    # 스케줄러 설정
    global scheduler
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        schedule_train_model, "cron", hour=2
    )  # 매일 새벽 2시에 재학습
    scheduler.start()

    print(
        "애플리케이션 시작: 데이터베이스 연결, 모델 로드 및 스케줄러 설정 완료"
    )


@app.on_event("shutdown")
async def shutdown_event():
    """앱 종료 시 실행되는 이벤트"""
    # 데이터베이스 연결 풀 종료
    await db_service.close_pool()

    # 스케줄러 종료
    if scheduler:
        scheduler.shutdown()
    print("애플리케이션 종료: 데이터베이스 연결 종료 및 리소스 정리 완료")


async def schedule_train_model():
    """주기적 모델 재학습 작업"""
    # 학습 데이터 가져오기
    training_data = await db_service.get_training_data()

    if len(training_data) < 10:  # 최소 학습 데이터 수 확인
        print("충분한 학습 데이터가 없습니다. 재학습을 건너뜁니다.")
        return

    # 모델 학습 실행
    try:
        result = await trainer.train_model(training_data)
        # 모델 버전 정보 저장
        await db_service.add_model_version(result["version"], result["metrics"])
        # 모델 다시 로드
        await address_model.load_model()
        print(f"모델 재학습 완료. 버전: {result['version']}")
    except Exception as e:
        print(f"모델 재학습 중 오류 발생: {str(e)}")


@app.post("/extract-address", response_model=AddressResponse)
async def extract_address(request: AddressRequest):
    """텍스트에서 주소 추출"""
    text = request.text

    if not text or len(text.strip()) == 0:
        raise HTTPException(status_code=400, detail="텍스트가 비어 있습니다.")

    try:
        extracted_addresses = await address_model.extract_addresses(text)

        # Pydantic 모델로 변환
        addresses = [
            Address(
                text=addr["text"],
                start=addr["start"],
                end=addr["end"],
                confidence=addr["confidence"],
            )
            for addr in extracted_addresses
        ]

        return AddressResponse(addresses=addresses, original_text=text)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"주소 추출 중 오류 발생: {str(e)}"
        )


@app.post("/validate-address", response_model=AddressValidationResponse)
async def validate_address(request: AddressValidationRequest):
    """주소의 유효성 검증"""
    address = request.address

    if not address or len(address.strip()) == 0:
        raise HTTPException(status_code=400, detail="주소가 비어 있습니다.")

    try:
        is_valid, confidence = validate_address_pattern(address)

        # 학습 데이터 저장 (선택적)
        await db_service.add_training_data(address, address, is_valid)

        return AddressValidationResponse(
            address=address, is_valid=is_valid, confidence=confidence
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"주소 검증 중 오류 발생: {str(e)}"
        )


@app.post("/train-model", response_model=TrainingResponse)
async def train_model_endpoint(
    data: List[TrainingData], background_tasks: BackgroundTasks
):
    """새로운 학습 데이터로 모델 재학습 (관리자용)"""
    try:
        # 학습 데이터 저장
        for item in data:
            await db_service.add_training_data(
                text=item.text, address=item.address, is_valid=item.is_valid
            )

        # 백그라운드에서 모델 재학습
        background_tasks.add_task(train_model_background, list(data))

        return TrainingResponse(
            status="success",
            message=f"{len(data)}개의 학습 데이터가 수신되었습니다. 모델 재학습이 백그라운드에서 시작됩니다.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"학습 요청 처리 중 오류 발생: {str(e)}"
        )


async def train_model_background(data: List[TrainingData]):
    """백그라운드에서 모델 재학습"""
    try:
        result = await trainer.train_model(data)
        # 모델 버전 정보 저장
        await db_service.add_model_version(result["version"], result["metrics"])
        # 모델 다시 로드
        await address_model.load_model()
        print(f"백그라운드 모델 재학습 완료. 버전: {result['version']}")
    except Exception as e:
        print(f"백그라운드 모델 재학습 중 오류 발생: {str(e)}")


@app.get("/model-info")
async def get_model_info():
    """현재 모델 정보 조회"""
    model_version = await db_service.get_latest_model_version()
    return {
        "model_loaded": address_model.model is not None,
        "latest_version": model_version,
    }


# 메인 실행 코드
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
