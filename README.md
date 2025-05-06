# 한국어 주소 추출 NER 프로젝트

한국어 주소를 인식하고 추출하는 개체명 인식(NER) 모델입니다. RoBERTa+BiLSTM+CRF 아키텍처를 활용하여 한국어 주소를 높은 정확도로 식별합니다.

## 설치 방법

```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 필요 패키지 설치
pip install -r requirements.txt
```

## 사용 방법

### 모델 학습
```bash
python train_model_from_csv.py
```

### 웹 서비스 실행
```bash
python run.py
```

## 프로젝트 구조
- `app/`: 애플리케이션 코드
  - `services/`: 모델 및 서비스 관련 로직
  - `utils/`: 데이터 전처리 및 유틸리티 함수
  - `models/`: 데이터 모델 정의
- `data/`: 학습 데이터
- `models/`: 학습된 모델 저장 위치

## 모델 아키텍처
- RoBERTa(klue/roberta-base) + BiLSTM + CRF
- 한국어 주소 추출에 최적화된 구조 