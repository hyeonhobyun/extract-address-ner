import torch
import torch.nn as nn
import os
from transformers import AutoModel, AutoTokenizer, AutoConfig
from torch.utils.data import Dataset, DataLoader
from torchcrf import CRF
from ..utils.preprocess import (
    load_and_preprocess_data,
    create_bio_tags,
    split_data,
)


class AddressDataset(Dataset):
    def __init__(
        self, data, tokenizer, max_len=256
    ):  # 최대 시퀀스 길이 증가 (128 -> 256)
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_map = {"O": 0, "B-ADDRESS": 1, "I-ADDRESS": 2}

        # 라벨 인코딩 사전 계산 - 재사용을 위해 캐싱
        self.encoded_data = []
        for i, item in enumerate(data):
            self.encoded_data.append(self._preprocess_item(item))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.encoded_data[idx]

    def _preprocess_item(self, item):
        """각 항목을 미리 전처리하여 캐싱"""
        text = item["text"]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # 레이블 매핑
        labels = [self.label_map[label] for label in item["labels"]]
        labels = labels[: self.max_len]
        labels = labels + [self.label_map["O"]] * (self.max_len - len(labels))

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class RoBERTaBiLSTMCRF(nn.Module):
    """RoBERTa + BiLSTM + CRF 모델"""

    def __init__(
        self,
        model_name,
        num_labels,
        lstm_hidden_size=256,
        lstm_layers=1,
        dropout=0.1,
    ):
        super(RoBERTaBiLSTMCRF, self).__init__()
        self.num_labels = num_labels

        # 모델 이름이 문자열이면 사전훈련된 모델 로드, 아니면 config만 사용
        if isinstance(model_name, str):
            # RoBERTa 모델 로드
            self.config = AutoConfig.from_pretrained(model_name)
            self.roberta = AutoModel.from_pretrained(
                model_name, config=self.config
            )
        else:
            self.config = model_name
            self.roberta = AutoModel.from_pretrained(
                "klue/roberta-base", config=self.config
            )

        # BiLSTM 레이어
        self.lstm = nn.LSTM(
            input_size=self.config.hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
        )

        # 드롭아웃
        self.dropout = nn.Dropout(dropout)

        # 출력 레이어
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_labels)

        # CRF 레이어
        self.crf = CRF(num_labels, batch_first=True)

        # 훈련 모드 플래그 (디버깅 출력 제어용)
        self.training_mode = False

    def save_pretrained(self, save_directory):
        """사용자 정의 save_pretrained 메소드"""
        # 디렉토리가 존재하지 않으면 생성
        os.makedirs(save_directory, exist_ok=True)

        # 모델 설정 저장
        self.config.save_pretrained(save_directory)

        # 모델 가중치 저장
        torch.save(
            self.state_dict(), os.path.join(save_directory, "pytorch_model.bin")
        )

        print(f"모델 저장 완료: {save_directory}")

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        token_type_ids=None,
    ):
        # RoBERTa 인코딩
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs.last_hidden_state

        # BiLSTM 레이어
        lstm_output, _ = self.lstm(sequence_output)
        lstm_output = self.dropout(lstm_output)

        # 로짓 계산
        logits = self.classifier(lstm_output)

        # 손실 계산
        loss = None
        if labels is not None:
            # 디버깅 정보 출력 (훈련 모드에서만, 첫번째 배치만)
            if self.training_mode:
                with torch.no_grad():
                    # 첫 번째 배치의 첫 번째 시퀀스 디버깅
                    sample_logits = logits[0].detach().cpu().numpy()
                    sample_labels = labels[0].detach().cpu().numpy()
                    sample_mask = (
                        attention_mask[0].bool().detach().cpu().numpy()
                    )

                    # logits의 클래스별 분포 확인
                    class_probs = (
                        torch.nn.functional.softmax(logits[0], dim=1)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    class_0_mean = class_probs[:, 0].mean()
                    class_1_mean = class_probs[:, 1].mean()
                    class_2_mean = class_probs[:, 2].mean()

                    print(
                        f"\n로짓 분포 - O: {class_0_mean:.4f}, B: {class_1_mean:.4f}, I: {class_2_mean:.4f}"
                    )

                    # 레이블 분포 확인
                    valid_labels = sample_labels[sample_mask]
                    label_counts = {0: 0, 1: 0, 2: 0}
                    for lbl in valid_labels:
                        if lbl in label_counts:
                            label_counts[lbl] += 1

                    print(
                        f"레이블 분포 - O: {label_counts[0]}, B: {label_counts[1]}, I: {label_counts[2]}"
                    )

            # CRF 손실 계산
            try:
                # 마스크 변환
                mask = attention_mask.bool()

                # CRF 손실 계산 (효율성 향상을 위해 예외 처리 간소화)
                log_likelihood = self.crf(
                    logits, labels, mask=mask, reduction="mean"
                )
                # CRF는 로그 가능도를 최대화하므로 음수를 취함
                # log_likelihood가 이미 스칼라인지 확인
                if (
                    isinstance(log_likelihood, torch.Tensor)
                    and log_likelihood.dim() > 0
                ):
                    log_likelihood = log_likelihood.mean()
                loss = -log_likelihood

            except Exception as e:
                # 대체 손실 함수로 CrossEntropyLoss 사용
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                active_loss = mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)

        return logits, loss

    def decode(self, logits, attention_mask=None):
        """CRF를 사용한 최적 태그 시퀀스 디코딩"""
        if attention_mask is None:
            # 기본적으로 모든 토큰이 유효하다고 가정
            mask = torch.ones(
                logits.shape[0],
                logits.shape[1],
                dtype=torch.bool,
                device=logits.device,
            )
        else:
            mask = attention_mask.bool()

        # CRF 디코딩으로 최적의 태그 시퀀스 찾기 (간소화된 버전)
        try:
            best_tags = self.crf.decode(logits, mask=mask)

            # 배치 내 모든 시퀀스에 대한 결과를 텐서로 변환
            max_len = logits.shape[1]
            result = torch.zeros(
                (logits.shape[0], max_len),
                dtype=torch.long,
                device=logits.device,
            )

            for i, tags in enumerate(best_tags):
                # 패딩 처리 최적화
                length = min(len(tags), max_len)
                result[i, :length] = torch.tensor(
                    tags[:length], dtype=torch.long, device=logits.device
                )

            return result

        except Exception as e:
            # 디코딩 실패 시 argmax 사용 (오류 메시지 없이 조용히 처리)
            return logits.argmax(dim=2)


class AddressModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda")
        self.label_map = {0: "O", 1: "B-ADDRESS", 2: "I-ADDRESS"}
        self.model_name = "klue/roberta-base"  # 한국어에 적합한 모델

    async def load_model(self, model_path="./models/address_ner_model"):
        """모델 로드 함수"""
        try:
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

            # 모델 파라미터 로드를 위한 설정
            config = AutoConfig.from_pretrained(model_path)

            # RoBERTa + BiLSTM + CRF 모델 생성
            self.model = RoBERTaBiLSTMCRF(
                model_path, num_labels=3, lstm_hidden_size=256, lstm_layers=1
            )

            # 모델 가중치 로드
            model_state_dict = torch.load(
                os.path.join(model_path, "pytorch_model.bin"),
                map_location=self.device,
            )
            self.model.load_state_dict(model_state_dict)

            self.model.to(self.device)
            self.model.eval()
            print("RoBERTa + BiLSTM + CRF 모델 로드 완료!")
            return True
        except Exception as e:
            print(f"모델 로드 오류: {e}")
            # 백업 옵션으로 사전 학습된 모델 사용
            try:
                print("사전 학습된 모델로 초기화 시도...")
                # 토크나이저 로드
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

                # 모델 생성
                self.model = RoBERTaBiLSTMCRF(
                    self.model_name,
                    num_labels=3,
                    lstm_hidden_size=256,
                    lstm_layers=1,
                )

                self.model.to(self.device)
                self.model.eval()
                print("사전 학습된 RoBERTa 모델 기반 초기화 완료!")
                return True
            except Exception as e2:
                print(f"사전 학습된 모델 로드 오류: {e2}")
                return False

    async def extract_addresses(self, text):
        """텍스트에서 주소 추출"""
        if not self.model or not self.tokenizer:
            raise ValueError(
                "모델이 로드되지 않았습니다. load_model을 먼저 호출하세요."
            )

        self.model.eval()

        # 토큰화 - 더 긴 텍스트를 처리할 수 있도록 최대 길이 증가
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            max_length=256,  # 기존 128에서 256으로 증가
            return_tensors="pt",
        )

        # 오프셋 정보 저장
        offset_mapping = encoding.pop("offset_mapping")[0]

        # 입력을 장치로 이동
        for key, value in encoding.items():
            encoding[key] = value.to(self.device)

        # 예측 - Mixed Precision 사용 (GPU 사용 시)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                logits, _ = self.model(
                    input_ids=encoding["input_ids"],
                    attention_mask=encoding["attention_mask"],
                )

            # CRF 디코딩으로 최적 태그 시퀀스 얻기
            predictions = self.model.decode(logits, encoding["attention_mask"])
            predictions = predictions[0]  # 배치의 첫 번째 항목

            # 소프트맥스로 신뢰도 계산
            probabilities = torch.nn.functional.softmax(logits, dim=2)[0]
            confidence_scores = [
                probabilities[i, pred].item() if i < len(predictions) else 0.0
                for i, pred in enumerate(predictions)
            ]

        # 결과 추출
        addresses = []
        current_address = []
        current_indices = []
        current_confidences = []

        for i, (pred, offset, conf) in enumerate(
            zip(predictions, offset_mapping, confidence_scores)
        ):
            if offset[0] == offset[1]:  # 특수 토큰이나 패딩은 건너뜀
                continue

            # 0: O, 1: B-ADDRESS, 2: I-ADDRESS
            if pred == 1:  # B-ADDRESS
                if current_address:
                    start_idx = current_indices[0][0]
                    end_idx = current_indices[-1][1]
                    address_text = text[start_idx:end_idx]
                    avg_confidence = sum(current_confidences) / len(
                        current_confidences
                    )

                    addresses.append(
                        {
                            "text": address_text,
                            "start": start_idx,
                            "end": end_idx,
                            "confidence": avg_confidence,
                        }
                    )

                    current_address = []
                    current_indices = []
                    current_confidences = []

                current_address.append(
                    self.tokenizer.decode([encoding["input_ids"][0][i].item()])
                )
                current_indices.append((offset[0].item(), offset[1].item()))
                current_confidences.append(conf)

            elif pred == 2:  # I-ADDRESS
                current_address.append(
                    self.tokenizer.decode([encoding["input_ids"][0][i].item()])
                )
                current_indices.append((offset[0].item(), offset[1].item()))
                current_confidences.append(conf)

            elif pred == 0 and current_address:  # O
                start_idx = current_indices[0][0]
                end_idx = current_indices[-1][1]
                address_text = text[start_idx:end_idx]
                avg_confidence = sum(current_confidences) / len(
                    current_confidences
                )

                addresses.append(
                    {
                        "text": address_text,
                        "start": start_idx,
                        "end": end_idx,
                        "confidence": avg_confidence,
                    }
                )

                current_address = []
                current_indices = []
                current_confidences = []

        # 마지막 주소 처리
        if current_address:
            start_idx = current_indices[0][0]
            end_idx = current_indices[-1][1]
            address_text = text[start_idx:end_idx]
            avg_confidence = sum(current_confidences) / len(current_confidences)

            addresses.append(
                {
                    "text": address_text,
                    "start": start_idx,
                    "end": end_idx,
                    "confidence": avg_confidence,
                }
            )

        return addresses

    async def predict(self, input_ids, attention_mask):
        """주어진 입력에 대한 예측을 수행합니다."""
        # 모델이 로드되지 않은 경우 로드
        if self.model is None:
            await self.load_model()

        # CPU -> GPU 이동
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Inference 모드로 전환
        self.model.eval()

        # 예측 수행
        with torch.no_grad():
            # CUDA 메모리 최적화
            torch.cuda.empty_cache()

            # Forward pass
            logits, _ = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )

            # CRF 디코딩으로 최적의 태그 시퀀스 찾기
            predictions = self.model.decode(
                logits, attention_mask=attention_mask
            )

        return predictions


# 전역 모델 인스턴스
address_model = AddressModel()
