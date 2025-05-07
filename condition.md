# 한국어 주소 추출 NER 프로젝트 개요

## 1. 프로젝트 소개

본 프로젝트는 텍스트에서 한국어 주소를 인식하고 추출하는 개체명 인식(Named Entity Recognition, NER) 모델을 구현한 것입니다. RoBERTa(klue/roberta-base)와 BiLSTM, CRF(Conditional Random Field)를 결합한 아키텍처를 활용하여 한국어 주소를 높은 정확도로 식별합니다.

### 주요 기능
- 텍스트에서 한국어 주소 추출
- 추출된 주소의 유효성 검증
- 모델 재학습 및 버전 관리
- REST API를 통한 서비스 제공

## 2. 개발 환경 및 요구사항

### 기술 스택
- **언어**: Python 3.8 이상
- **웹 프레임워크**: FastAPI
- **모델**: PyTorch, Transformers(Hugging Face), pytorch-crf
- **데이터베이스**: PostgreSQL
- **기타 라이브러리**: pandas, numpy, scikit-learn, seqeval

### 주요 패키지 버전
```
fastapi==0.95.0
uvicorn==0.21.1
pydantic==1.10.7
torch==2.0.0
transformers==4.27.4
pandas==2.0.0
numpy==1.24.2
scikit-learn==1.2.2
seqeval==1.2.2
pytorch-crf==0.7.2
asyncpg==0.30.0
python-dotenv==1.1.0
```

## 3. 프로젝트 구조

```
extract-address-ner/
├── app/                        # 애플리케이션 코드
│   ├── services/               # 서비스 로직
│   │   ├── ml_model.py         # 기계 학습 모델 서비스
│   │   ├── train_service.py    # 모델 학습 서비스
│   │   └── db_service.py       # 데이터베이스 서비스
│   ├── utils/                  # 유틸리티 함수
│   │   └── preprocess.py       # 데이터 전처리 함수
│   ├── models/                 # 데이터 모델 정의
│   │   └── address.py          # 주소 관련 Pydantic 모델
│   └── main.py                 # FastAPI 메인 애플리케이션
├── data/                       # 학습 데이터
│   └── korean_address_dataset.csv  # 한국어 주소 데이터셋
├── train_model_from_csv.py     # CSV 파일에서 모델 학습 스크립트
├── run.py                      # 애플리케이션 실행 스크립트
├── requirements.txt            # 필요 패키지 목록
├── README.md                   # 프로젝트 설명
└── Mermaid.md                  # 프로젝트 구조 다이어그램
```

## 4. 모델 아키텍처 및 학습 과정

### 모델 아키텍처
- **RoBERTa**: 한국어 사전학습 모델 (klue/roberta-base)
- **BiLSTM**: 양방향 LSTM 계층
- **CRF**: 조건부 랜덤 필드 계층

이 세 가지 계층을 결합하여 시퀀스 라벨링 태스크에 최적화된 모델을 구현했습니다. RoBERTa에서 추출한 토큰 표현(token representation)을 BiLSTM을 통해 문맥 정보를 강화하고, CRF 계층을 통해 출력 태그 간의 의존성을 고려합니다.

### 학습 과정
1. CSV 파일에서 주소 데이터 로드
2. 데이터 전처리 및 BIO 태깅 (Beginning, Inside, Outside)
3. 학습/검증/테스트 세트로 분할
4. RoBERTa + BiLSTM + CRF 모델 초기화 및 학습
5. 모델 평가 및 저장
6. 학습 결과 및 성능 지표 기록

## 5. API 기능 설명

### 엔드포인트

#### 1. 주소 추출 (`/extract-address`)
- **HTTP 메서드**: POST
- **설명**: 입력 텍스트에서 한국어 주소를 추출
- **요청 형식**: 
  ```json
  {
    "text": "내일 서울특별시 강남구 테헤란로 123번길 45에서 회의가 있습니다."
  }
  ```
- **응답 형식**:
  ```json
  {
    "addresses": [
      {
        "text": "서울특별시 강남구 테헤란로 123번길 45",
        "start": 3,
        "end": 28,
        "confidence": 0.95
      }
    ],
    "original_text": "내일 서울특별시 강남구 테헤란로 123번길 45에서 회의가 있습니다."
  }
  ```

#### 2. 주소 유효성 검증 (`/validate-address`)
- **HTTP 메서드**: POST
- **설명**: 주소의 유효성 검증
- **요청 형식**:
  ```json
  {
    "address": "서울특별시 강남구 테헤란로 123번길 45"
  }
  ```
- **응답 형식**:
  ```json
  {
    "address": "서울특별시 강남구 테헤란로 123번길 45",
    "is_valid": true,
    "confidence": 0.98
  }
  ```

#### 3. 모델 재학습 (`/train-model`)
- **HTTP 메서드**: POST
- **설명**: 새로운 학습 데이터로 모델 재학습 (관리자용)
- **요청 형식**:
  ```json
  [
    {
      "text": "내일 서울특별시 강남구 테헤란로 123번길 45에서 회의가 있습니다.",
      "address": "서울특별시 강남구 테헤란로 123번길 45",
      "is_valid": true
    }
  ]
  ```
- **응답 형식**:
  ```json
  {
    "status": "success",
    "message": "1개의 학습 데이터가 수신되었습니다. 모델 재학습이 백그라운드에서 시작됩니다."
  }
  ```

#### 4. 모델 정보 조회 (`/model-info`)
- **HTTP 메서드**: GET
- **설명**: 현재 모델 정보 조회
- **응답 형식**:
  ```json
  {
    "model_loaded": true,
    "latest_version": {
      "version": "v1.2.3",
      "metrics": {
        "accuracy": 0.95,
        "f1": 0.92,
        "precision": 0.93,
        "recall": 0.91
      }
    }
  }
  ```

## 6. 데이터베이스 구조

PostgreSQL 데이터베이스를 사용하여 다음 두 가지 주요 테이블을 관리합니다:

### 학습 데이터 테이블 (training_data)
- id: 기본 키
- text: 전체 텍스트
- address: 주소 텍스트
- is_valid: 유효한 주소 여부
- created_at: 생성 시간

### 모델 버전 테이블 (model_versions)
- id: 기본 키
- version: 모델 버전
- metrics: 성능 지표 (JSON 형식)
- created_at: 생성 시간

## 7. 배포 및 실행 방법

### 설치
```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 필요 패키지 설치
pip install -r requirements.txt
```

### 모델 학습
```bash
python train_model_from_csv.py
```

### 웹 서비스 실행
```bash
python run.py
```

서비스는 기본적으로 `http://0.0.0.0:8000`에서 실행되며, Swagger 문서는 `http://0.0.0.0:8000/docs`에서 확인할 수 있습니다.

## 8. 성능 평가 지표

모델의 성능은 다음 지표를 통해 평가됩니다:

- **정확도(Accuracy)**: 전체 토큰 중 올바르게 예측된 토큰의 비율
- **F1 점수(F1 Score)**: 정밀도와 재현율의 조화 평균
- **정밀도(Precision)**: 주소로 예측한 토큰 중 실제 주소인 토큰의 비율
- **재현율(Recall)**: 실제 주소 토큰 중 주소로 예측된 토큰의 비율

## 9. 기타 참고사항

### 자동 재학습
시스템은 매일 새벽 2시에 데이터베이스에 저장된 학습 데이터를 기반으로 모델을 자동으로 재학습합니다. 이를 통해 모델이 새로운 패턴을 학습하고 성능을 개선할 수 있습니다.

### Kaggle 환경 지원
본 프로젝트는 Kaggle 환경에서도 실행할 수 있도록 설계되었습니다. 다양한 CSV 파일 경로를 검색하여 적합한 데이터셋을 찾아 학습합니다.

### 확장성
새로운 유형의 한국어 주소를 지원하기 위해 데이터셋을 확장하고 모델을 재학습할 수 있습니다. API 엔드포인트를 통해 새로운 학습 데이터를 추가하고 모델을 개선할 수 있습니다. 