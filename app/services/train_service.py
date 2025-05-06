import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from transformers import AdamW
import os
import pandas as pd
import time
from datetime import datetime
from torchcrf import CRF
from seqeval.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

from ..utils.preprocess import (
    load_and_preprocess_data,
    create_bio_tags,
    split_data,
)
from .ml_model import AddressDataset, RoBERTaBiLSTMCRF


class ModelTrainer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_name = "klue/roberta-base"  # 한국어에 적합한 모델
        print(f"사용 디바이스: {self.device}")

        # BiLSTM 및 CRF 하이퍼파라미터
        self.lstm_hidden_size = 256
        self.lstm_layers = 1
        self.dropout = 0.1
        self.num_labels = 3  # O, B-ADDRESS, I-ADDRESS

    async def train_model(self, custom_data=None, epochs=5, batch_size=16):
        """주소 추출 모델 학습"""
        print("모델 학습 시작...")
        start_time = time.time()

        # 데이터 준비
        if custom_data:
            # 커스텀 데이터로 학습
            print("커스텀 데이터 사용")
            df = self._prepare_custom_data(custom_data)
        else:
            # 기본 예제 데이터로 학습
            print("CSV 파일 데이터 사용")
            df = load_and_preprocess_data()

        print(f"데이터 로드 완료: {len(df)}개 데이터")

        print("BIO 태깅 생성 시작...")
        dataset = create_bio_tags(df)
        print(f"BIO 태깅 생성 완료: {len(dataset)}개 데이터")

        # 데이터셋 크기에 따라 배치 크기 조정
        if len(dataset) < batch_size * 2:
            adj_batch_size = max(1, len(dataset) // 4)
            print(
                f"경고: 데이터셋이 작아서 배치 크기를 {batch_size}에서 {adj_batch_size}로 조정합니다."
            )
            batch_size = adj_batch_size

        train_data, test_data = split_data(dataset)
        print(
            f"데이터 분할 완료: 학습 데이터 {len(train_data)}개, 테스트 데이터 {len(test_data)}개"
        )

        # 토크나이저 로드
        print(f"토크나이저 로드 중: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print("토크나이저 로드 완료")

        # RoBERTa + BiLSTM + CRF 모델 생성
        print(f"RoBERTa + BiLSTM + CRF 모델 초기화 중...")
        self.model = RoBERTaBiLSTMCRF(
            self.model_name,
            num_labels=self.num_labels,
            lstm_hidden_size=self.lstm_hidden_size,
            lstm_layers=self.lstm_layers,
            dropout=self.dropout,
        )
        print("모델 초기화 완료")

        # 데이터셋 및 데이터로더 생성
        print("데이터셋 준비 중...")
        train_dataset = AddressDataset(train_data, self.tokenizer)
        test_dataset = AddressDataset(test_data, self.tokenizer)

        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
        print(
            f"데이터로더 준비 완료: 학습 배치 {len(train_dataloader)}개, 테스트 배치 {len(test_dataloader)}개"
        )

        # 옵티마이저 및 스케줄러 설정
        print("옵티마이저 및 스케줄러 설정 중...")

        # 가중치 감쇠 설정
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps * 0.1,
            num_training_steps=total_steps,
        )
        print(
            f"학습 설정: 에폭 {epochs}개, 배치 크기 {batch_size}, 총 스텝 {total_steps}개"
        )

        # 학습 루프
        print(f"모델을 {self.device}로 이동 중...")
        self.model.to(self.device)
        print("학습 시작")

        global_step = 0
        best_f1 = 0.0

        for epoch in range(epochs):
            epoch_start_time = time.time()
            print(f"\n{'='*50}")
            print(f"에폭 {epoch+1}/{epochs} 시작")
            self.model.train()
            total_loss = 0

            for batch_idx, batch in enumerate(train_dataloader):
                # 배치 로그 (10개 배치마다 한 번씩 출력)
                if batch_idx % 10 == 0:
                    print(
                        f"  배치 {batch_idx+1}/{len(train_dataloader)} 처리 중... (글로벌 스텝: {global_step+1})"
                    )

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                self.model.zero_grad()

                # 순방향 전파
                logits, loss = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                batch_loss = loss.item()
                total_loss += batch_loss

                # 더 자세한 손실 로그 (50개 배치마다 한 번씩 출력)
                if batch_idx % 50 == 0:
                    print(f"    배치 손실: {batch_loss:.4f}")

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                global_step += 1

            avg_loss = total_loss / len(train_dataloader)
            epoch_time = time.time() - epoch_start_time
            print(f"에폭 {epoch+1} 완료 (소요 시간: {epoch_time:.2f}초)")
            print(f"평균 손실: {avg_loss:.4f}")

            # 평가
            print("에폭 평가 중...")
            eval_start_time = time.time()
            metrics = self._evaluate_model(test_dataloader)
            eval_time = time.time() - eval_start_time
            print(f"평가 완료 (소요 시간: {eval_time:.2f}초)")
            print(
                f"평가 결과: 정확도 = {metrics['accuracy']:.4f}, F1 = {metrics['f1']:.4f}"
            )

            # 최고 성능 모델 저장
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                print(
                    f"새로운 최고 F1 점수 ({best_f1:.4f}) 달성! 모델 저장 중..."
                )
                best_model_path = f"./models/address_ner_model_best"
                os.makedirs(best_model_path, exist_ok=True)
                self.model.save_pretrained(best_model_path)
                self.tokenizer.save_pretrained(best_model_path)
                torch.save(
                    self.model.state_dict(),
                    os.path.join(best_model_path, "pytorch_model.bin"),
                )
                print(f"최고 성능 모델 저장 완료: {best_model_path}")

            # 중간 저장 (옵션)
            if (epoch + 1) % 2 == 0:  # 2 에폭마다 저장
                print(f"중간 모델 저장 중... (에폭 {epoch+1})")
                intermediate_path = (
                    f"./models/address_ner_model_epoch_{epoch+1}"
                )
                os.makedirs(intermediate_path, exist_ok=True)
                torch.save(
                    self.model.state_dict(),
                    os.path.join(intermediate_path, "pytorch_model.bin"),
                )
                self.tokenizer.save_pretrained(intermediate_path)
                print(f"중간 모델 저장 완료: {intermediate_path}")

        # 최종 모델 저장
        print("\n최종 모델 저장 중...")
        model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"./models/address_ner_model_{model_version}"
        os.makedirs(model_path, exist_ok=True)

        torch.save(
            self.model.state_dict(),
            os.path.join(model_path, "pytorch_model.bin"),
        )
        self.tokenizer.save_pretrained(model_path)
        print(f"버전별 모델 저장 완료: {model_path}")

        # 기본 모델 경로에도 저장
        default_path = "./models/address_ner_model"
        os.makedirs(default_path, exist_ok=True)
        torch.save(
            self.model.state_dict(),
            os.path.join(default_path, "pytorch_model.bin"),
        )
        self.tokenizer.save_pretrained(default_path)
        print(f"기본 모델 저장 완료: {default_path}")

        total_time = time.time() - start_time
        print(f"\n모델 학습 완료 (총 소요 시간: {total_time:.2f}초)")
        print(f"모델 저장 경로: {model_path}")

        return {"version": model_version, "metrics": metrics}

    def _prepare_custom_data(self, custom_data):
        """커스텀 데이터를 학습 데이터 형식으로 변환"""
        print("커스텀 데이터 변환 중...")
        data = []
        for item in custom_data:
            # 주소 위치 찾기
            text = item.text
            if item.is_valid and item.address in text:
                start = text.find(item.address)
                end = start + len(item.address)
                data.append(
                    {"text": text, "is_address": 1, "start": start, "end": end}
                )
            else:
                data.append(
                    {"text": text, "is_address": 0, "start": 0, "end": 0}
                )

        result_df = pd.DataFrame(data)
        print(f"커스텀 데이터 변환 완료: {len(result_df)}개 데이터")
        return result_df

    def _evaluate_model(self, test_dataloader):
        """모델 평가"""
        self.model.eval()

        # 평가에 필요한 변수 초기화
        all_predictions = []
        all_labels = []

        # 예외 사항: 테스트 데이터셋이 작은 경우 처리
        if len(test_dataloader) == 0:
            print("경고: 테스트 데이터가 없어 평가를 건너뜁니다.")
            return {
                "accuracy": 0.0,
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
            }

        # 클래스별 ID-레이블 매핑
        id_to_label = {0: "O", 1: "B-ADDRESS", 2: "I-ADDRESS"}

        with torch.no_grad():
            for batch in test_dataloader:
                # 배치 데이터 가져오기
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # 모델 추론
                logits, _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                # 마스크 생성 (패딩 토큰 제외)
                mask = attention_mask.bool()

                # 예측 결과 디코딩
                try:
                    predictions = self.model.decode(logits, attention_mask=mask)
                except Exception as e:
                    print(f"디코딩 오류: {e}")
                    predictions = logits.argmax(dim=2)

                # 원래 레이블과 예측을 평평하게 만들기
                for i in range(len(input_ids)):
                    sample_mask = mask[i].tolist()
                    valid_indices = [
                        idx for idx, m in enumerate(sample_mask) if m
                    ]

                    if not valid_indices:  # 유효한 토큰이 없는 경우 건너뜀
                        continue

                    sample_preds = [
                        id_to_label[predictions[i][idx].item()]
                        for idx in valid_indices
                    ]
                    sample_labels = [
                        id_to_label[labels[i][idx].item()]
                        for idx in valid_indices
                    ]

                    all_predictions.append(sample_preds)
                    all_labels.append(sample_labels)

        # 예외 처리: 평가 데이터가 없는 경우
        if not all_predictions or not all_labels:
            print("경고: 평가를 위한 유효한 예측 또는 레이블이 없습니다.")
            return {
                "accuracy": 0.0,
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
            }

        try:
            # seqeval 라이브러리로 평가
            accuracy = sum(
                1 for p, l in zip(all_predictions, all_labels) if p == l
            ) / len(all_predictions)

            # 혹시 오류가 발생할 경우를 대비해 try-except 추가
            try:
                f1 = f1_score(all_labels, all_predictions)
                precision = precision_score(all_labels, all_predictions)
                recall = recall_score(all_labels, all_predictions)
            except Exception as e:
                print(f"평가 지표 계산 오류: {e}, 기본값 반환")
                f1 = 0.0
                precision = 0.0
                recall = 0.0

            # 분류 보고서 출력
            try:
                report = classification_report(all_labels, all_predictions)
                print("\n분류 보고서:")
                print(report)
            except Exception as e:
                print(f"분류 보고서 생성 오류: {e}")

        except Exception as e:
            print(f"평가 중 오류 발생: {e}")
            accuracy = 0.0
            f1 = 0.0
            precision = 0.0
            recall = 0.0

        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }
