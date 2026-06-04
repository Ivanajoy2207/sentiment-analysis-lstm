import re
import string
import joblib
import streamlit as st

MODEL_PATH = "sentiment_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

model, vectorizer = load_model()

st.title("💬 Sentiment Analysis App")
st.write("Prediksi sentimen teks menggunakan Machine Learning.")

text_input = st.text_area(
    "Masukkan teks:",
    "makanannya enak dan pelayanannya cepat"
)

if st.button("Prediksi"):

    cleaned = clean_text(text_input)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    if prediction == 1:
        st.success("Sentimen Positif 😊")
    else:
        st.error("Sentimen Negatif 😡")

    st.write("Teks setelah preprocessing:")
    st.code(cleaned)
