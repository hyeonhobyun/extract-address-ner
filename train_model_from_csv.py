import asyncio
import torch
import os
import gc
from datetime import datetime
import pandas as pd
from app.utils.preprocess import (
    load_and_preprocess_data,
    create_bio_tags,
    split_data,
)
from app.services.train_service import ModelTrainer


async def train_model_from_csv(
    csv_path="data/korean_address_dataset.csv",
    epochs=5,
    batch_size=32,
    gradient_accumulation_steps=2,
    num_workers=4,
    use_mixed_precision=True,
):
    """CSV 파일에서 데이터를 로드하여 RoBERTa + BiLSTM + CRF 모델 학습 - 최적화 버전"""
    print("CSV 파일에서 RoBERTa + BiLSTM + CRF 모델 학습 시작 (최적화 버전)...")
    print(
        f"설정: 에폭 {epochs}회, 배치 크기 {batch_size}, 그래디언트 누적 스텝 {gradient_accumulation_steps}"
    )
    print(
        f"워커 수: {num_workers}, Mixed Precision 사용: {use_mixed_precision}"
    )
    print(f"CSV 경로: {csv_path}")

    # 학습 시작 전 메모리 정리
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"CUDA 캐시 정리 완료")

    # 데이터 로드 및 전처리
    try:
        # Kaggle 환경인지 확인
        is_kaggle = os.path.exists("/kaggle")
        print(f"Kaggle 환경: {is_kaggle}")

        # Kaggle 환경에서 CSV 파일 경로 확인
        if is_kaggle:
            kaggle_paths = [
                "/kaggle/input/korean-address-dataset/korean_address_dataset.csv",
                "/kaggle/input/extract-address-ner/data/korean_address_dataset.csv",
                "/kaggle/working/data/korean_address_dataset.csv",
                "/kaggle/input/rootpath/data/korean_address_dataset.csv",
            ]

            for alt_path in kaggle_paths:
                if os.path.exists(alt_path):
                    print(f"Kaggle 환경에서 CSV 파일 발견: {alt_path}")
                    csv_path = alt_path
                    break

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

        # 데이터 준비
        df = load_and_preprocess_data(csv_path)
        print(f"데이터 로드 완료. 총 {len(df)} 개의 데이터.")

        # 샘플 데이터로 학습하고 있는지 확인 (10개 이하면 경고)
        if len(df) <= 10:
            print(
                "경고: 매우 적은 수의 데이터로 학습합니다. 이는 샘플 데이터일 가능성이 높습니다."
            )
            print(
                "경로가 올바른지 확인하고, 실제 데이터셋을 사용하는지 확인하세요."
            )
            # 매우 작은 데이터셋에 대한 배치 크기 및 그래디언트 누적 스텝 조정
            batch_size = 2
            gradient_accumulation_steps = 1
            num_workers = 0
            print(
                f"작은 데이터셋으로 인해 배치 크기({batch_size}), 그래디언트 누적 스텝({gradient_accumulation_steps}), 워커 수({num_workers})를 조정했습니다."
            )

        # 시스템 환경에 따른 워커 수 조정
        if os.name == "nt":  # Windows 환경
            num_workers = 0
            print(f"Windows 환경에서 실행 중이므로 워커 수를 0으로 설정합니다.")

        # 사용 가능한 GPU 확인 및 설정
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_names = [
                torch.cuda.get_device_name(i) for i in range(gpu_count)
            ]
            gpu_mem = [
                torch.cuda.get_device_properties(i).total_memory / (1024**3)
                for i in range(gpu_count)
            ]

            print(f"사용 가능한 GPU: {gpu_count}개")
            for i in range(gpu_count):
                print(f"  - GPU {i}: {gpu_names[i]} ({gpu_mem[i]:.1f} GB)")

            # 메모리가 충분한 경우 배치 크기 자동 조정
            if gpu_count > 0 and gpu_mem[0] > 16:  # 16GB 이상인 경우
                batch_size = max(batch_size, 64)  # 최소 64로 설정
                print(
                    f"고성능 GPU 감지: 배치 크기를 {batch_size}로 자동 조정했습니다."
                )
        else:
            print("CUDA를 사용할 수 없습니다. CPU로 학습합니다.")
            # CPU 학습 시 효율성을 위해 배치 크기와 워커 수 조정
            batch_size = min(batch_size, 16)
            gradient_accumulation_steps = max(gradient_accumulation_steps, 4)
            num_workers = min(num_workers, 2)
            print(
                f"CPU 학습을 위해 배치 크기({batch_size}), 그래디언트 누적 스텝({gradient_accumulation_steps}), 워커 수({num_workers})를 조정했습니다."
            )

        # 모델 훈련기 초기화
        trainer = ModelTrainer()

        # 모델 학습 - 최적화된 매개변수 전달
        result = await trainer.train_model(
            epochs=epochs,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_workers=num_workers,
        )

        # 결과 출력
        print("\n" + "=" * 50)
        print("모델 학습 완료 요약:")
        print(f"  - 버전: {result['version']}")
        print(
            f"  - 소요 시간: {result['total_time']:.1f}초 ({result['total_time']/60:.1f}분)"
        )
        print(f"  - 최고 F1 점수: {result['best_f1']:.4f}")
        print("성능 지표:")
        print(f"  - 정확도: {result['metrics']['accuracy']:.4f}")
        print(f"  - F1 점수: {result['metrics']['f1']:.4f}")
        print(f"  - 정밀도: {result['metrics']['precision']:.4f}")
        print(f"  - 재현율: {result['metrics']['recall']:.4f}")
        print(f"모델 저장 경로: {model_dir}")
        print("=" * 50)

        return result

    except Exception as e:
        print(f"모델 학습 중 오류 발생: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


def run_training(
    csv_path="data/korean_address_dataset.csv",
    epochs=5,
    batch_size=32,
    gradient_accumulation_steps=2,
    num_workers=4,
    use_mixed_precision=True,
):
    """Jupyter/Kaggle 환경에서도 작동하는 학습 함수 - 최적화 버전"""
    # 필수 패키지 확인
    try:
        import torch
        from torchcrf import CRF
        from transformers import AutoModel, AutoTokenizer

        print(f"PyTorch 버전: {torch.__version__}")
        print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA 장치: {torch.cuda.get_device_name(0)}")

            # CUDNN 자동 튜너 설정으로 속도 향상
            torch.backends.cudnn.benchmark = True
            print("cuDNN 벤치마크 모드 활성화")

            # Mixed Precision 지원 확인
            if hasattr(torch.cuda, "amp") and use_mixed_precision:
                print("Mixed Precision 학습 지원")
            else:
                use_mixed_precision = False
                print("Mixed Precision 학습이 지원되지 않습니다")
        else:
            use_mixed_precision = False

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
                    train_model_from_csv(
                        csv_path,
                        epochs,
                        batch_size,
                        gradient_accumulation_steps,
                        num_workers,
                        use_mixed_precision,
                    )
                )
            except ImportError:
                # nest_asyncio가 없는 경우 현재 루프 사용
                return loop.run_until_complete(
                    train_model_from_csv(
                        csv_path,
                        epochs,
                        batch_size,
                        gradient_accumulation_steps,
                        num_workers,
                        use_mixed_precision,
                    )
                )
        else:
            # 일반 환경인 경우
            return asyncio.run(
                train_model_from_csv(
                    csv_path,
                    epochs,
                    batch_size,
                    gradient_accumulation_steps,
                    num_workers,
                    use_mixed_precision,
                )
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
    # 명령줄 인수로 하이퍼파라미터를 받을 수 있도록 수정
    import argparse

    parser = argparse.ArgumentParser(
        description="한국어 주소 추출 NER 모델 학습"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/korean_address_dataset.csv",
        help="학습 데이터 CSV 파일 경로",
    )
    parser.add_argument("--epochs", type=int, default=5, help="학습 에폭 수")
    parser.add_argument("--batch_size", type=int, default=32, help="배치 크기")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="그래디언트 누적 스텝 수",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="데이터 로딩 워커 수"
    )
    parser.add_argument(
        "--no_mixed_precision",
        action="store_true",
        help="Mixed precision 학습 비활성화",
    )

    args = parser.parse_args()

    # 비동기 함수 실행 - 모든 환경에서 호환되는 함수 사용
    run_training(
        csv_path=args.csv_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_workers=args.num_workers,
        use_mixed_precision=not args.no_mixed_precision,
    )
