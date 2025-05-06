import asyncio
import torch
import os
from datetime import datetime
import pandas as pd
from app.utils.preprocess import (
    load_and_preprocess_data,
    create_bio_tags,
    split_data,
)
from app.services.train_service import ModelTrainer


async def train_model_from_csv(
    csv_path="data/korean_address_dataset.csv", epochs=5, batch_size=16
):
    """CSV 파일에서 데이터를 로드하여 RoBERTa + BiLSTM + CRF 모델 학습"""
    print("CSV 파일에서 RoBERTa + BiLSTM + CRF 모델 학습 시작...")
    print(f"설정: 에폭 {epochs}회, 배치 크기 {batch_size}")
    print(f"CSV 경로: {csv_path}")

    # 데이터 로드 및 전처리
    try:
        # Kaggle 환경인지 확인
        is_kaggle = "/kaggle/input" in csv_path

        # 모델 저장 경로 설정
        if is_kaggle:
            models_dir = "/kaggle/working/models"
            model_dir = "/kaggle/working/models/address_ner_model"
        else:
            models_dir = "./models"
            model_dir = "./models/address_ner_model"

        # 필요한 디렉토리 생성
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        # CSV 파일 존재 확인
        if not os.path.exists(csv_path):
            print(f"경고: {csv_path} 파일을 찾을 수 없습니다.")

            # Kaggle 환경에서의 대체 경로 시도
            if "/kaggle/input" in csv_path:
                alt_paths = [
                    "/kaggle/input/korean-address-dataset/korean_address_dataset.csv",
                    "/kaggle/input/extract-address-ner/data/korean_address_dataset.csv",
                ]

                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        print(f"대체 CSV 파일 발견: {alt_path}")
                        csv_path = alt_path
                        break

        # 파일 존재 여부 최종 확인
        if not os.path.exists(csv_path):
            print(f"경고: CSV 파일을 찾을 수 없어 샘플 데이터로 학습합니다.")

        # 데이터 준비
        df = load_and_preprocess_data(csv_path)
        print(f"데이터 로드 완료. 총 {len(df)} 개의 데이터.")

        # 샘플 데이터로 학습하고 있는지 확인 (5개 이하면 경고)
        if len(df) <= 5:
            print(
                "경고: 매우 적은 수의 데이터로 학습합니다. 이는 샘플 데이터일 가능성이 높습니다."
            )
            print(
                "경로가 올바른지 확인하고, 실제 데이터셋을 사용하는지 확인하세요."
            )

        # 모델 훈련기 초기화
        trainer = ModelTrainer()

        # 모델 학습 - 에폭과 배치 크기 전달
        result = await trainer.train_model(epochs=epochs, batch_size=batch_size)

        # 결과 출력
        print(f"모델 학습 완료. 버전: {result['version']}")
        print(f"성능 지표:")
        print(f"  - 정확도: {result['metrics']['accuracy']:.4f}")
        print(f"  - F1 점수: {result['metrics']['f1']:.4f}")
        print(f"  - 정밀도: {result['metrics']['precision']:.4f}")
        print(f"  - 재현율: {result['metrics']['recall']:.4f}")
        print(f"모델 저장 경로: {model_dir}")

        return result

    except Exception as e:
        print(f"모델 학습 중 오류 발생: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


def run_training(
    csv_path="data/korean_address_dataset.csv", epochs=5, batch_size=16
):
    """Jupyter/Kaggle 환경에서도 작동하는 학습 함수"""
    # 필수 패키지 확인
    try:
        import torch
        from torchcrf import CRF
        from transformers import AutoModel, AutoTokenizer

        print(f"PyTorch 버전: {torch.__version__}")
        print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA 장치: {torch.cuda.get_device_name(0)}")
        print("모델 학습에 필요한 모든 패키지가 설치되어 있습니다.")
    except ImportError as e:
        print(f"패키지 불러오기 오류: {e}")
        print(
            "필요한 패키지가 설치되어 있지 않습니다. 'pip install -r requirements.txt'를 실행하세요."
        )
        return False

    # 현재 환경 확인 및 적절한 방법으로 비동기 함수 실행
    try:
        # 이미 실행 중인 이벤트 루프가 있는지 확인 (Jupyter/Kaggle 환경)
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Jupyter/Kaggle 환경인 경우
            try:
                # nest_asyncio 사용 시도
                import nest_asyncio

                nest_asyncio.apply()
                return asyncio.run(
                    train_model_from_csv(csv_path, epochs, batch_size)
                )
            except ImportError:
                # nest_asyncio가 없는 경우 현재 루프 사용
                return loop.run_until_complete(
                    train_model_from_csv(csv_path, epochs, batch_size)
                )
        else:
            # 일반 환경인 경우
            return asyncio.run(
                train_model_from_csv(csv_path, epochs, batch_size)
            )
    except RuntimeError as e:
        if "already running" in str(e):
            print(
                "이벤트 루프가 이미 실행 중입니다. Jupyter/Kaggle 환경으로 판단됩니다."
            )
            print("이 문제를 해결하려면 다음 명령을 먼저 실행하세요:")
            print("!pip install nest_asyncio")
            print("import nest_asyncio")
            print("nest_asyncio.apply()")
            return False
        else:
            raise
    except Exception as e:
        print(f"실행 중 오류 발생: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    # 비동기 함수 실행 - 모든 환경에서 호환되는 함수 사용
    run_training()
