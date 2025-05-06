import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


def load_and_preprocess_data(csv_path="data/korean_address_dataset.csv"):
    """CSV 파일에서 데이터 로드 및 전처리"""
    try:
        # CSV 파일 로드
        df = pd.read_csv(csv_path)

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

        return pd.DataFrame(processed_data)

    except Exception as e:
        print(f"CSV 파일 로드 오류: {e}")
        # 오류 발생 시 기본 예제 데이터 사용
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
        ]
        return pd.DataFrame(data)


def create_bio_tags(df):
    """BIO 태깅 방식으로 레이블 생성"""
    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")

    dataset = []
    for _, row in df.iterrows():
        text = row["text"]
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        # 토큰 경계와 문자 단위 경계가 일치하지 않을 수 있으므로 조정이 필요함
        labels = ["O"] * len(tokens)

        if row["is_address"] == 1:
            start_char = row["start"]
            end_char = row["end"]

            # 각 토큰의 시작 위치와 끝 위치를 찾아 BIO 태그 할당
            char_idx = 0
            in_entity = False

            for i, token in enumerate(tokens):
                token_len = len(token.replace("##", ""))
                token_start = char_idx
                token_end = char_idx + token_len

                if token_start >= start_char and token_end <= end_char:
                    if not in_entity:
                        labels[i] = "B-ADDRESS"
                        in_entity = True
                    else:
                        labels[i] = "I-ADDRESS"
                elif (
                    token_start < start_char < token_end
                    or token_start < end_char < token_end
                ):
                    # 부분적으로 겹치는 경우
                    if not in_entity:
                        labels[i] = "B-ADDRESS"
                        in_entity = True
                    else:
                        labels[i] = "I-ADDRESS"
                else:
                    labels[i] = "O"
                    in_entity = False

                char_idx += token_len

        dataset.append(
            {
                "text": text,
                "tokens": tokens,
                "token_ids": token_ids,
                "labels": labels,
                "is_address": row["is_address"],
            }
        )

    return dataset


def split_data(dataset):
    """데이터 학습/테스트 세트 분할"""
    train_data, test_data = train_test_split(
        dataset, test_size=0.2, random_state=42
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
