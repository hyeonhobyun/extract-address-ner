import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


def load_and_preprocess_data(csv_path="data/korean_address_dataset.csv"):
    """CSV 파일에서 데이터 로드 및 전처리"""
    try:
        # CSV 파일 로드
        if not os.path.exists(csv_path):
            print(f"경고: CSV 파일을 찾을 수 없습니다: {csv_path}")

            # Kaggle 환경에서의 대체 경로 시도
            kaggle_paths = [
                "/kaggle/input/korean-address-dataset/korean_address_dataset.csv",
                "/kaggle/input/extract-address-ner/data/korean_address_dataset.csv",
                "/kaggle/working/data/korean_address_dataset.csv",
                "/kaggle/input/rootpath/data/korean_address_dataset.csv",
            ]

            for alt_path in kaggle_paths:
                if os.path.exists(alt_path):
                    print(f"대체 CSV 파일 발견: {alt_path}")
                    csv_path = alt_path
                    break

            # 파일이 여전히 존재하지 않으면 기본 예제 데이터 사용
            if not os.path.exists(csv_path):
                return create_sample_data()

        print(f"CSV 파일 로드 중: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"CSV 파일 로드 완료: {len(df)}개 데이터 발견")

        # 필요한 형식으로 변환
        processed_data = []

        for _, row in df.iterrows():
            text = row["text"]
            is_valid = 1 if row["label"] == "정상" else 0

            # 주소 텍스트가 전체 텍스트와 동일
            if is_valid:
                processed_data.append(
                    {
                        "text": text,
                        "is_address": is_valid,
                        "start": 0,  # 텍스트 전체가 주소이므로 시작은 0
                        "end": len(
                            text
                        ),  # 텍스트 전체가 주소이므로 끝은 텍스트 길이
                    }
                )
            else:
                processed_data.append(
                    {"text": text, "is_address": is_valid, "start": 0, "end": 0}
                )

        result_df = pd.DataFrame(processed_data)
        print(
            f"데이터 전처리 완료: 정상 주소 {result_df['is_address'].sum()}개, 비정상 주소 {len(result_df) - result_df['is_address'].sum()}개"
        )
        return result_df

    except Exception as e:
        print(f"CSV 파일 로드 오류: {e}")
        return create_sample_data()


def create_sample_data():
    """샘플 데이터 생성"""
    print("경고: 기본 예제 데이터를 사용합니다 (10개의 샘플 데이터)")
    # 샘플 데이터 크기를 10개로 늘림
    data = [
        {
            "text": "내일 서울특별시 강남구 테헤란로 123번길 45에서 회의가 있습니다.",
            "is_address": 1,
            "start": 3,
            "end": 28,
        },
        {
            "text": "경기도 성남시 분당구 판교역로 235 에서 만나자",
            "is_address": 1,
            "start": 0,
            "end": 23,
        },
        {
            "text": "우리 집은 제주특별자치도 서귀포시 123-45입니다",
            "is_address": 1,
            "start": 6,
            "end": 25,
        },
        {
            "text": "서울시 강남구 123길은 존재하지 않는 주소입니다",
            "is_address": 0,
            "start": 0,
            "end": 13,
        },
        {
            "text": "경기도 신도시에서 저녁을 먹었어요",
            "is_address": 0,
            "start": 0,
            "end": 8,
        },
        # 추가 샘플 데이터 (5개 더 추가)
        {
            "text": "부산광역시 해운대구 해운대해변로 264",
            "is_address": 1,
            "start": 0,
            "end": 22,
        },
        {
            "text": "경상북도 경주시 보문로 507",
            "is_address": 1,
            "start": 0,
            "end": 17,
        },
        {
            "text": "인천광역시 중구 월미로 329",
            "is_address": 1,
            "start": 0,
            "end": 16,
        },
        {
            "text": "도로명주소가 아니라 그냥 동네 이름입니다",
            "is_address": 0,
            "start": 0,
            "end": 10,
        },
        {
            "text": "서울시 강동구 어딘가에 살고 있어요",
            "is_address": 0,
            "start": 0,
            "end": 9,
        },
    ]
    return pd.DataFrame(data)


def create_bio_tags(df):
    """BIO 태깅 방식으로 레이블 생성"""
    print("BIO 태깅 시작...")
    # 특수 토큰도 포함해서 적절히 처리하기 위해 tokenizer 준비
    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")

    b_tags_count = 0
    i_tags_count = 0
    o_tags_count = 0

    dataset = []
    for idx, row in enumerate(df.iterrows()):
        _, row_data = row
        text = row_data["text"]

        # RoBERTa 토큰화 과정에는 특수 토큰이 추가됨
        # 여기서 특수 토큰을 포함한 정확한 토큰 목록 가져오기
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        # 특수 토큰을 제외한 원래 토큰 위치
        offset_mapping = encoding.offset_mapping[0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(encoding.input_ids[0])

        # 모든 토큰에 'O' 태그 할당
        labels = ["O"] * len(tokens)

        if row_data["is_address"] == 1:
            start_char = row_data["start"]
            end_char = row_data["end"]

            # 첫 번째 주소 토큰 찾았는지 여부
            found_first = False

            # 오프셋 매핑을 사용하여 각 토큰의 위치와 주소 위치 비교
            for i, (token_start, token_end) in enumerate(offset_mapping):
                # 특수 토큰([CLS], [SEP] 등)은 건너뜀 (start == end == 0)
                if token_start == token_end:
                    continue

                # 토큰이 주소 범위 내에 있는지 확인
                if token_start >= start_char and token_end <= end_char:
                    if not found_first:
                        labels[i] = "B-ADDRESS"
                        found_first = True
                        b_tags_count += 1
                    else:
                        labels[i] = "I-ADDRESS"
                        i_tags_count += 1
                else:
                    labels[i] = "O"
                    o_tags_count += 1

        # 현재 레이블 추가 디버깅
        if idx < 3:  # 처음 3개 예시만 출력
            print(f"예시 {idx+1}:")
            print(f"텍스트: {text}")
            print(f"토큰: {tokens}")
            print(f"레이블: {labels}")
            print()

        dataset.append(
            {
                "text": text,
                "tokens": tokens,
                "token_ids": encoding.input_ids[0].tolist(),
                "labels": labels,
                "is_address": row_data["is_address"],
            }
        )

    print(
        f"BIO 태깅 완료: B-ADDRESS {b_tags_count}개, I-ADDRESS {i_tags_count}개, O {o_tags_count}개"
    )
    print(f"총 {len(dataset)}개 데이터셋 생성")
    return dataset


def split_data(dataset, test_size=0.2, random_state=42):
    """데이터 학습/테스트 세트 분할"""
    # 데이터 크기에 따라 stratify 사용 여부 결정
    is_address_list = [item["is_address"] for item in dataset]
    indices = np.arange(len(dataset))

    # 데이터 크기와 클래스 분포 확인
    dataset_size = len(dataset)
    class_counts = {}
    for is_address in is_address_list:
        if is_address not in class_counts:
            class_counts[is_address] = 0
        class_counts[is_address] += 1

    n_classes = len(class_counts)
    min_test_size = n_classes + 1  # 최소한 클래스 수 + 1개의 샘플이 필요

    # 계산된 테스트 크기(전체의 20%)가 최소 테스트 크기보다 작은지 확인
    calculated_test_size = max(int(dataset_size * test_size), min_test_size)

    # 데이터셋이 너무 작으면 stratify 사용하지 않음
    if dataset_size < 20 or calculated_test_size <= n_classes:
        print(
            f"경고: 데이터셋이 작아서 stratify 없이 분할합니다 (크기: {dataset_size})"
        )
        train_indices, test_indices = train_test_split(
            indices,
            test_size=min(test_size, 0.5),  # 테스트 크기를 최대 50%로 제한
            random_state=random_state,
        )
    else:
        # 일반적인 경우: 층화 샘플링 사용
        train_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=is_address_list,
        )

    # 인덱스를 사용하여 원본 데이터 분할
    train_data = [dataset[i] for i in train_indices]
    test_data = [dataset[i] for i in test_indices]

    # 분포 확인
    train_address = sum(1 for item in train_data if item["is_address"] == 1)
    test_address = sum(1 for item in test_data if item["is_address"] == 1)

    print(
        f"학습 데이터: {len(train_data)}개 (정상 주소: {train_address}개, 비정상 주소: {len(train_data) - train_address}개)"
    )
    print(
        f"테스트 데이터: {len(test_data)}개 (정상 주소: {test_address}개, 비정상 주소: {len(test_data) - test_address}개)"
    )

    return train_data, test_data


def validate_address_pattern(address):
    """주소 패턴 검증 (간단한 문법 체크)"""
    valid_patterns = [
        r"(.+[시군구])(.+[동읍면])(.+[길로])?(.+[번지번길])?",
        r"(.+도)(.+[시군])(.+[동읍면])(.+[길로])?(.+[번지번길])?",
        r"(.+[특별시광역시])(.+[구])(.+[동])(.+[길로])?(.+[번지번길])?",
    ]

    is_valid = any(re.search(pattern, address) for pattern in valid_patterns)
    confidence = (
        0.85 if is_valid else 0.15
    )  # 실제로는 모델 기반 신뢰도 계산 필요

    return is_valid, confidence
