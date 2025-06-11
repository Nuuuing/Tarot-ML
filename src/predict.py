# src/predict.py
import json
import random
from transformers import T5Tokenizer, T5ForConditionalGeneration

# 모델과 토크나이저 로드
model = T5ForConditionalGeneration.from_pretrained("model/tarot-t5")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# --- 전체 카드 이름과 방향 조합 로드 ---
# JSON 파일에서 카드 이름만 로드하고, 각 카드에 대해 정/역방향 조합을 만듭니다.
def load_all_card_names_with_directions(json_path="data/tarot.json"):
    names_with_directions = []
    directions_options = ["forward", "reversed"] # 가능한 방향 목록
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for card in data.get('tarot', []):
                name = card.get('name')
                if name:
                    # 각 카드 이름에 대해 모든 방향을 조합하여 리스트에 추가합니다.
                    for direction in directions_options:
                        names_with_directions.append(f"{name}({direction})")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading card names from JSON: {e}")
    return names_with_directions

all_tarot_card_names_with_directions = load_all_card_names_with_directions()

# --- 랜덤으로 카드 N개 (방향 포함) 뽑는 함수 ---
def get_random_card_combination_with_directions(num_cards=3):
    if not all_tarot_card_names_with_directions or len(all_tarot_card_names_with_directions) < num_cards:
        print("Not enough cards (including directions) in the list to select from.")
        return ""
    # 중복 없이 무작위로 카드를 선택합니다. (ex: The Fool(forward), The Lovers(reversed) 등)
    selected_cards = random.sample(all_tarot_card_names_with_directions, num_cards)
    return ", ".join(selected_cards)

# --- 예측 함수 수정: input_cards 문자열에 이미 방향 정보가 포함되어야 합니다. ---
def predict(card_input_str_with_directions):
    """
    주어진 카드 조합 문자열 (방향 포함)으로 점괘를 생성합니다.
    예: "The Magician(forward), Judgment(forward), Strength(forward)"
    """
    input_text = "카드 조합: " + card_input_str_with_directions
    inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True) # max_length 조정
    
    output = model.generate(
        **inputs, 
        max_length=256,        # 점괘의 최대 길이 (학습 시의 max_length와 비슷하게 유지하거나 더 길게)
        num_beams=5,           # 빔 서치 사용
        early_stopping=True,   # 모델이 EOS 토큰 생성 시 멈춤
        no_repeat_ngram_size=2,# 반복 방지
        do_sample=True,        # 샘플링 방식 사용
        top_k=50,              # 상위 K개 토큰 내에서 샘플링
        temperature=0.7        # 생성 다양성 조절
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# --- 실행 예시 ---
if __name__ == "__main__":
    # 무작위로 3장의 카드 (방향 포함)를 뽑아 예측
    if all_tarot_card_names_with_directions:
        cards_for_prediction = get_random_card_combination_with_directions(num_cards=3) 
        if cards_for_prediction:
            print(f"🔮 무작위 카드 조합 (방향 포함): {cards_for_prediction}")
            model_result = predict(cards_for_prediction)
            print("📜 점괘:", model_result)
        else:
            print("무작위 카드 조합을 생성할 수 없습니다.")
    else:
        print("타로 카드 이름 목록을 로드할 수 없어 예측을 실행할 수 없습니다. 'data/tarot.json' 파일을 확인해 주세요.")

    print("\n" + "="*50 + "\n")

    # 특정 카드 조합 (정방향)으로 테스트
    specific_cards_forward = "The Magician(forward), Judgment(forward), Strength(forward)"
    print(f"🔮 특정 카드 조합 (정방향): {specific_cards_forward}")
    model_result_forward = predict(specific_cards_forward)
    print("📜 점괘:", model_result_forward)

    # 특정 카드 조합 (역방향 섞음)으로 테스트
    specific_cards_mixed = "The Magician(reversed), Judgment(forward), Strength(reversed)"
    print(f"\n🔮 특정 카드 조합 (역방향 섞음): {specific_cards_mixed}")
    model_result_mixed = predict(specific_cards_mixed)
    print("📜 점괘:", model_result_mixed)