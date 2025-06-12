import pandas as pd
from datasets import Dataset
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer

# 1. 학습 데이터 로드
try:
    df = pd.read_csv("data/tarot_dataset.csv")
    if df.empty:
        raise ValueError("tarot_dataset.csv is empty. Please ensure it contains data.")
    required_columns = ['input_cards', 'prediction_text', 'directions']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing columns in tarot_dataset.csv. Expected: {required_columns}")
    print(f"Loaded {len(df)}")  # 로드 파일 체크
except FileNotFoundError:
    print("Error: 'data/tarot_dataset.csv' not found.")
    exit()
except ValueError as e:
    print(f"Data loading error: {e}")
    exit()

# Pandas DataFrame을 Hugging Face Dataset으로 변환
dataset = Dataset.from_pandas(df)

# 2. 토크나이저와 모델 로드 (`KETI-AIR/ke-t5-small` 모델)
model_name = "KETI-AIR/ke-t5-small" 
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name) 

print(f"Tokenizer and model '{model_name}' loaded successfully.")


# 3. 데이터 전처리 함수 정의
def preprocess(example):
    cards = example["input_cards"].split(', ')
    directions = example["directions"].split(', ')

    # 선택한 카드 데이터의 수와 방향 데이터 수가 다를 경우를 대비해 작은 데이터에 맞춤
    min_len = min(len(cards), len(directions))  
    card_with_directions = [f"{card.strip()}({direction.strip()})" for card, direction in zip(cards[:min_len], directions[:min_len])]

    # 모델의 입력 프롬프트 형식
    input_text = "카드 조합: " + ", ".join(card_with_directions)
    target_text = example["prediction_text"]

    # 입력 텍스트 토큰화
    model_inputs = tokenizer(input_text, padding="max_length", max_length=256, truncation=True) # 입력 최대 길이 256
    # 정답 텍스트 토큰화
    labels = tokenizer(target_text, padding="max_length", max_length=512, truncation=True) # 정답 최대 길이 512

    # 토큰화된 정답 ID를 'labels' 키에 할당
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 데이터셋을 토큰화 함수에 적용
tokenized_dataset = dataset.map(preprocess, batched=False)
print(f"Tokenized dataset contains {len(tokenized_dataset)} examples.")

# 학습에 필요 없는 원본 컬럼 제거 
tokenized_dataset = tokenized_dataset.remove_columns(["input_cards", "prediction_text", "directions"])
print(f"Removed original columns. Final tokenized dataset columns: {tokenized_dataset.column_names}")

# 4. TrainingArguments 설정
training_args = TrainingArguments(
    output_dir="model/tarot-koT5",       # 학습된 모델 및 체크포인트 저장 디렉토리
    eval_strategy="no",                  # 평가를 수행하지 않음 (훈련 데이터셋이 작을 때 유용)
    per_device_train_batch_size=4,       # GPU당 훈련 배치 크기
    num_train_epochs=50,                 # 전체 훈련 에포크 수 
    save_strategy="epoch",               # 각 에포크 종료 시 모델 저장
    logging_steps=10,                    # 10 스텝마다 로그 출력
    save_total_limit=1,                  # 저장할 체크포인트 수 제한. 마지막 모델만 남김.
    learning_rate=3e-5,                  # 학습률
    fp16=True,                           # Mixed precision 학습 (GPU 사용) 활성화 (GPU 사용 시 속도 및 메모리 효율 향상)
    push_to_hub=False,                   # Hugging Face Hub에 모델 푸시 여부
)

# 5. Trainer 초기화 및 학습 시작
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer, # 토크나이저도 Trainer에 전달하여 저장 시 함께 저장
)

print("Starting model training ----------")
trainer.train()
print("Model training finished ----------")

# 6. 학습된 모델 저장
trainer.save_model("model/tarot-koT5")