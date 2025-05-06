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
        self.model.training_mode = False  # 디버깅 정보 끄기

        eval_predictions = []
        eval_true_labels = []

        # 레이블 매핑
        id_to_label = {0: "O", 1: "B-ADDRESS", 2: "I-ADDRESS"}

        # 혼동 행렬 초기화
        confusion_matrix = {
            "TP": 0,  # True Positive
            "FP": 0,  # False Positive
            "FN": 0,  # False Negative
            "TN": 0,  # True Negative
        }

        print(f"총 {len(test_dataloader)} 배치 평가 중...")
        for batch_idx, batch in enumerate(test_dataloader):
            # 평가 진행상황 (20개 배치마다 한 번씩 출력)
            if batch_idx % 20 == 0:
                print(
                    f"  평가 배치 {batch_idx+1}/{len(test_dataloader)} 처리 중..."
                )

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            with torch.no_grad():
                logits, _ = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                )

                # CRF 디코딩으로 최적 태그 시퀀스 얻기
                predictions = self.model.decode(logits, attention_mask)

            # 배치의 각 시퀀스에 대해 처리
            for i, (pred_seq, true_seq, mask) in enumerate(
                zip(predictions, labels, attention_mask)
            ):
                pred_tags = []
                true_tags = []

                # 각 토큰에 대해 레이블 처리
                valid_tokens = 0
                correct_tokens = 0

                # 유효한 토큰에 대해서만 처리 (패딩 제외)
                for j, is_valid in enumerate(mask):
                    if is_valid and j < len(pred_seq):
                        valid_tokens += 1

                        pred_id = pred_seq[j]
                        true_id = true_seq[j].item()

                        # 정수 ID를 문자열 태그로 변환
                        pred_tag = id_to_label.get(pred_id, "O")
                        true_tag = id_to_label.get(true_id, "O")

                        # 정확도 계산용
                        if pred_id == true_id:
                            correct_tokens += 1

                        # 혼동 행렬 업데이트 (ADDRESS 태그 식별 기준)
                        if true_tag != "O":  # 실제 주소인 경우
                            if pred_tag != "O":  # 주소로 예측한 경우
                                confusion_matrix["TP"] += 1
                            else:  # 주소가 아닌 것으로 예측한 경우
                                confusion_matrix["FN"] += 1
                        else:  # 실제 주소가 아닌 경우
                            if pred_tag != "O":  # 주소로 예측한 경우
                                confusion_matrix["FP"] += 1
                            else:  # 주소가 아닌 것으로 예측한 경우
                                confusion_matrix["TN"] += 1

                        pred_tags.append(pred_tag)
                        true_tags.append(true_tag)

                # 평가를 위해 저장
                if pred_tags:  # 빈 시퀀스가 아닌 경우에만
                    eval_predictions.append(pred_tags)
                    eval_true_labels.append(true_tags)

        # 토큰 단위 정확도 계산
        correct = 0
        total = 0
        for pred_seq, true_seq in zip(eval_predictions, eval_true_labels):
            for pred_tag, true_tag in zip(pred_seq, true_seq):
                if pred_tag == true_tag:
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0

        # 혼동 행렬에서 지표 계산
        tp = confusion_matrix["TP"]
        fp = confusion_matrix["FP"]
        fn = confusion_matrix["FN"]
        tn = confusion_matrix["TN"]

        print(f"혼동 행렬: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

        # 직접 F1 점수 계산
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        print(
            f"직접 계산 - 정확도: {accuracy:.4f}, F1: {f1:.4f}, 정밀도: {precision:.4f}, 재현율: {recall:.4f}"
        )

        # seqeval로 F1 점수 계산
        try:
            seqeval_f1 = f1_score(eval_true_labels, eval_predictions)
            seqeval_precision = precision_score(
                eval_true_labels, eval_predictions
            )
            seqeval_recall = recall_score(eval_true_labels, eval_predictions)

            # 상세 분류 보고서
            report = classification_report(
                eval_true_labels, eval_predictions, output_dict=True
            )

            # ADDRESS 엔티티에 대한 성능
            address_metrics = {
                "B-ADDRESS": report.get("B-ADDRESS", {"f1-score": 0}),
                "I-ADDRESS": report.get("I-ADDRESS", {"f1-score": 0}),
            }

            print(
                f"seqeval - 정확도: {accuracy:.4f}, F1: {seqeval_f1:.4f}, 정밀도: {seqeval_precision:.4f}, 재현율: {seqeval_recall:.4f}"
            )
            print(
                f"주소 엔티티 성능: B-ADDRESS F1={address_metrics['B-ADDRESS'].get('f1-score', 0):.4f}, "
                + f"I-ADDRESS F1={address_metrics['I-ADDRESS'].get('f1-score', 0):.4f}"
            )

            # 최종 평가 지표로 seqeval 결과 사용 (더 정확함)
            f1 = seqeval_f1
            precision = seqeval_precision
            recall = seqeval_recall

        except Exception as e:
            print(f"seqeval F1 점수 계산 오류: {e}")
            # 이미 직접 계산한 값 사용

        print(f"정확도 계산 완료: {correct}/{total} = {accuracy:.4f}")

        return {
            "accuracy": float(accuracy),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
        }
