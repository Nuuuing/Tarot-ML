import streamlit as st
from src.predict import predict

st.title("🔮 타로 점괘 생성기")
cards = st.text_input("카드 3장 입력 (예: The Fool, The Magician, The Lovers)")
if st.button("점괘 보기"):
    result = predict(cards)
    st.write("✨ 점괘:", result)