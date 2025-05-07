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
import gc  # 가비지 컬렉션을 위한 모듈 추가

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

        # 멀티 GPU 지원 확인
        self.use_multi_gpu = torch.cuda.device_count() > 1
        self.device = torch.device("cuda")

        # Mixed precision 학습을 위한 scaler 추가
        self.scaler = torch.cuda.amp.GradScaler()

        self.model_name = "klue/roberta-base"  # 한국어에 적합한 모델
        print(f"사용 디바이스: {self.device}")
        if self.use_multi_gpu:
            print(
                f"멀티 GPU 사용: {torch.cuda.device_count()}개의 GPU 사용 가능"
            )

        # BiLSTM 및 CRF 하이퍼파라미터
        self.lstm_hidden_size = 256
        self.lstm_layers = 1
        self.dropout = 0.1
        self.num_labels = 3  # O, B-ADDRESS, I-ADDRESS

    async def train_model(
        self,
        custom_data=None,
        epochs=5,
        batch_size=32,
        gradient_accumulation_steps=2,
        num_workers=4,
    ):
        """주소 추출 모델 학습 - 최적화 버전"""
        print("모델 학습 시작 (최적화 버전)...")
        start_time = time.time()

        # 메모리 캐시 정리
        gc.collect()
        torch.cuda.empty_cache()

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

        # 데이터셋 크기에 따라 배치 크기 및 GPU 메모리 최적화
        if len(dataset) < batch_size * 2:
            adj_batch_size = max(1, len(dataset) // 4)
            print(
                f"경고: 데이터셋이 작아서 배치 크기를 {batch_size}에서 {adj_batch_size}로 조정합니다."
            )
            batch_size = adj_batch_size
            gradient_accumulation_steps = 1
        elif self.use_multi_gpu:
            # 멀티 GPU 사용 시 더 큰 배치 사용
            batch_size = min(batch_size * 2, 64)
            print(f"멀티 GPU 환경에서 배치 크기 {batch_size}로 조정")

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

        # 데이터셋 및 데이터로더 생성 (멀티 프로세싱 데이터 로딩 최적화)
        print("데이터셋 준비 중...")
        train_dataset = AddressDataset(train_data, self.tokenizer)
        test_dataset = AddressDataset(test_data, self.tokenizer)

        # 데이터 로딩 최적화: num_workers, pin_memory 설정
        # Windows 환경에서는 num_workers가 0이면 에러 방지
        if os.name == "nt":
            num_workers = 0
            print("Windows 환경에서 num_workers=0으로 설정")

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size * 2,  # 평가 시 더 큰 배치 사용 가능
            num_workers=num_workers,
            pin_memory=True,
        )
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

        # AdamW 옵티마이저 사용 (더 높은 학습률 설정)
        optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5)

        # 그래디언트 누적 스텝을 고려한 총 스텝 계산
        total_steps = (
            len(train_dataloader) // gradient_accumulation_steps * epochs
        )

        # 학습률 스케줄러 설정
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps * 0.1,
            num_training_steps=total_steps,
        )
        print(
            f"학습 설정: 에폭 {epochs}개, 배치 크기 {batch_size}, 그래디언트 누적 스텝 {gradient_accumulation_steps}, 총 스텝 {total_steps}개"
        )

        # 학습 루프
        print(f"모델을 {self.device}로 이동 중...")
        self.model.to(self.device)

        # 멀티 GPU 지원
        if self.use_multi_gpu:
            print(
                f"{torch.cuda.device_count()}개의 GPU로 데이터 병렬 처리 설정"
            )
            self.model = torch.nn.DataParallel(self.model)

        print("학습 시작")

        global_step = 0
        best_f1 = 0.0

        # 디버깅 모드 비활성화 (불필요한 출력 줄이기)
        if hasattr(self.model, "module"):
            self.model.module.training_mode = False
        else:
            self.model.training_mode = False

        for epoch in range(epochs):
            epoch_start_time = time.time()
            print(f"\n{'='*50}")
            print(f"에폭 {epoch+1}/{epochs} 시작")
            self.model.train()
            total_loss = 0

            # 학습 진행상황 표시 간격 계산
            log_interval = max(1, len(train_dataloader) // 10)

            for batch_idx, batch in enumerate(train_dataloader):
                # 로깅 최적화: 10% 간격으로만 출력
                if batch_idx % log_interval == 0:
                    print(
                        f"  배치 진행률: {batch_idx}/{len(train_dataloader)} ({batch_idx/len(train_dataloader)*100:.1f}%)"
                    )

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # 그래디언트 누적 처리
                # 그래디언트 누적의 첫 번째 스텝에서만 그래디언트 초기화
                if batch_idx % gradient_accumulation_steps == 0:
                    self.model.zero_grad()

                # Mixed Precision 학습 사용 (GPU 사용 시)
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        logits, loss = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                        # loss가 스칼라가 아닌 경우 mean() 적용
                        if loss.dim() > 0:
                            loss = loss.mean()
                        # 그래디언트 누적을 위한 손실 스케일링
                        loss = loss / gradient_accumulation_steps

                    # 그래디언트 역전파 및 스케일링
                    self.scaler.scale(loss).backward()

                    # 그래디언트 누적 완료 시에만 옵티마이저 스텝
                    if (batch_idx + 1) % gradient_accumulation_steps == 0 or (
                        batch_idx + 1
                    ) == len(train_dataloader):
                        # 그래디언트 클리핑
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=1.0
                        )

                        # 옵티마이저 스텝 및 스케일러 업데이트
                        self.scaler.step(optimizer)
                        self.scaler.update()
                        scheduler.step()
                        global_step += 1
                else:
                    # GPU가 없거나 Mixed Precision을 지원하지 않는 경우 일반 학습
                    logits, loss = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    # loss가 스칼라가 아닌 경우 mean() 적용
                    if loss.dim() > 0:
                        loss = loss.mean()
                    # 그래디언트 누적을 위한 손실 스케일링
                    loss = loss / gradient_accumulation_steps
                    loss.backward()

                    # 그래디언트 누적 완료 시에만 옵티마이저 스텝
                    if (batch_idx + 1) % gradient_accumulation_steps == 0 or (
                        batch_idx + 1
                    ) == len(train_dataloader):
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=1.0
                        )
                        optimizer.step()
                        scheduler.step()
                        self.model.zero_grad()
                        global_step += 1

                # 원래 스케일의 손실값 기록
                batch_loss = loss.item() * gradient_accumulation_steps
                total_loss += batch_loss

            avg_loss = total_loss / len(train_dataloader)
            epoch_time = time.time() - epoch_start_time
            print(f"에폭 {epoch+1} 완료 (소요 시간: {epoch_time:.2f}초)")
            print(f"평균 손실: {avg_loss:.4f}")

            # 메모리 정리
            gc.collect()
            torch.cuda.empty_cache()

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

                # 멀티 GPU 사용 시 모델 모듈 추출
                if self.use_multi_gpu:
                    model_to_save = self.model.module
                else:
                    model_to_save = self.model

                model_to_save.save_pretrained(best_model_path)
                self.tokenizer.save_pretrained(best_model_path)
                print(f"최고 성능 모델 저장 완료: {best_model_path}")

            # 중간 저장은 에폭 수가 많을 때만 수행 (에폭이 적으면 불필요)
            if epochs > 3 and (epoch + 1) % 3 == 0:
                print(f"중간 모델 저장 중... (에폭 {epoch+1})")
                intermediate_path = (
                    f"./models/address_ner_model_epoch_{epoch+1}"
                )
                os.makedirs(intermediate_path, exist_ok=True)

                # 멀티 GPU 사용 시 모델 모듈 추출
                if self.use_multi_gpu:
                    torch.save(
                        self.model.module.state_dict(),
                        os.path.join(intermediate_path, "pytorch_model.bin"),
                    )
                else:
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

        # 멀티 GPU 사용 시 모델 모듈 추출
        if self.use_multi_gpu:
            torch.save(
                self.model.module.state_dict(),
                os.path.join(model_path, "pytorch_model.bin"),
            )
        else:
            torch.save(
                self.model.state_dict(),
                os.path.join(model_path, "pytorch_model.bin"),
            )

        self.tokenizer.save_pretrained(model_path)

        # 최종 결과 출력
        total_time = time.time() - start_time
        print(
            f"\n학습 완료! 총 소요 시간: {total_time:.2f}초 ({total_time/60:.2f}분)"
        )
        print(f"최고 F1 점수: {best_f1:.4f}")
        print(f"최종 모델 저장 경로: {model_path}")

        # 최종 결과 반환
        result = {
            "version": model_version,
            "metrics": metrics,
            "best_f1": best_f1,
            "total_time": total_time,
        }

        return result

    def _prepare_custom_data(self, custom_data):
        """커스텀 데이터 준비"""
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
        """모델 평가 함수"""
        # 평가 모드로 변경
        self.model.eval()

        all_predictions = []
        all_labels = []

        # 불필요한 그래디언트 계산 방지
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Mixed Precision 사용 (GPU 사용 시에만)
                with torch.cuda.amp.autocast():
                    logits, _ = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    predictions = (
                        self.model.module.decode(logits, attention_mask)
                        if self.use_multi_gpu
                        else self.model.decode(logits, attention_mask)
                    )

                # CPU로 이동하여 처리
                predictions = predictions.detach().cpu().numpy()
                labels_np = labels.detach().cpu().numpy()
                attention_np = attention_mask.detach().cpu().numpy()

                # 각 배치에서 유효한 부분만 추출
                for i in range(len(predictions)):
                    pred_list = []
                    label_list = []

                    for j in range(len(predictions[i])):
                        if attention_np[i][j] == 0:  # 패딩 부분은 건너뜀
                            continue

                        pred_list.append(self._id_to_label(predictions[i][j]))
                        label_list.append(self._id_to_label(labels_np[i][j]))

                    all_predictions.append(pred_list)
                    all_labels.append(label_list)

        # 평가 지표 계산
        accuracy = precision_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)

        # 평가 결과 반환
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        return metrics

    def _id_to_label(self, label_id):
        """ID를 레이블로 변환"""
        label_map = {0: "O", 1: "B-ADDRESS", 2: "I-ADDRESS"}
        return label_map.get(label_id, "O")
