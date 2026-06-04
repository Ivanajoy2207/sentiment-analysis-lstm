import os
import pickle
import re
import string

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_PATH = "sentiment_lstm_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
CONFIG_PATH = "config.pkl"

st.set_page_config(page_title="Text Sentiment Analysis LSTM", page_icon="💬", layout="centered")

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+|#\w+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@st.cache_resource
def load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH) and os.path.exists(CONFIG_PATH)):
        return None, None, None

    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    with open(CONFIG_PATH, "rb") as f:
        config = pickle.load(f)
    return model, tokenizer, config

def predict_sentiment(text: str, model, tokenizer, config):
    cleaned = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=config["max_len"], padding="post", truncating="post")
    probability = float(model.predict(padded, verbose=0)[0][0])

    label = "Positif" if probability >= 0.5 else "Negatif"
    confidence = probability if probability >= 0.5 else 1 - probability
    return label, probability, confidence, cleaned

st.title("💬 Text Sentiment Analysis dengan LSTM")
st.write("Aplikasi ini memprediksi sentimen teks Bahasa Indonesia menjadi **Positif** atau **Negatif** menggunakan model LSTM.")

model, tokenizer, config = load_artifacts()

if model is None:
    st.error("Model belum ditemukan. Jalankan `python train_model.py` terlebih dahulu, lalu deploy ulang ke Streamlit.")
    st.stop()

with st.sidebar:
    st.header("Info Model")
    st.write(f"Maximum sequence length: `{config['max_len']}`")
    st.write(f"Vocabulary size: `{config['num_words']}`")
    st.write(f"Label positif: `{config['positive_label']}`")
    st.write(f"Label negatif: `{config['negative_label']}`")

example_text = "makanannya enak porsinya banyak dan pelayanannya ramah"
user_text = st.text_area(
    "Masukkan teks yang ingin dianalisis:",
    value=example_text,
    height=140,
    placeholder="Contoh: produknya bagus dan pengiriman cepat"
)

if st.button("Prediksi Sentimen", type="primary"):
    if not user_text.strip():
        st.warning("Masukkan teks terlebih dahulu.")
    else:
        label, probability, confidence, cleaned = predict_sentiment(user_text, model, tokenizer, config)

        if label == "Positif":
            st.success(f"Sentimen: {label}")
        else:
            st.error(f"Sentimen: {label}")

        st.metric("Confidence", f"{confidence * 100:.2f}%")
        st.progress(confidence)
        st.write("Probabilitas kelas positif:", f"{probability:.4f}")

        with st.expander("Teks setelah preprocessing"):
            st.write(cleaned)

st.divider()
st.caption("Dibuat dengan Streamlit + TensorFlow/Keras LSTM.")
