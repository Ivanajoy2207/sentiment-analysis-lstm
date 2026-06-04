import os
import pickle
import random
import re
import string

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

DATA_PATH = "data.csv"
MODEL_PATH = "sentiment_lstm_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
CONFIG_PATH = "config.pkl"

SEED = 42
NUM_WORDS = 10000
MAX_LEN = 80
EMBEDDING_DIM = 128
BATCH_SIZE = 32
EPOCHS = 10

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+|#\w+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset harus memiliki kolom `text` dan `label`.")

    df = df[["text", "label"]].dropna()
    df["text_clean"] = df["text"].apply(clean_text)
    df = df[df["text_clean"].str.len() > 0]

    labels = sorted(df["label"].unique().tolist())
    if len(labels) != 2:
        raise ValueError(f"Model ini dibuat untuk binary sentiment. Label ditemukan: {labels}")

    # Dataset contoh memakai 0 = negatif, 1 = positif.
    y = df["label"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        df["text_clean"].values,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y
    )

    tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding="post", truncating="post")

    model = Sequential([
        Embedding(input_dim=NUM_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
        SpatialDropout1D(0.25),
        LSTM(128, dropout=0.25, recurrent_dropout=0.25),
        Dense(64, activation="relu"),
        Dropout(0.4),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, mode="max")
    ]

    history = model.fit(
        X_train_pad,
        y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Load model terbaik dari checkpoint jika ada.
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        model.save(MODEL_PATH)

    loss, accuracy = model.evaluate(X_test_pad, y_test, verbose=0)
    y_prob = model.predict(X_test_pad, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    print("\n=== Evaluation ===")
    print(f"Test Loss     : {loss:.4f}")
    print(f"Test Accuracy : {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)

    config = {
        "max_len": MAX_LEN,
        "num_words": NUM_WORDS,
        "positive_label": 1,
        "negative_label": 0,
        "test_accuracy": float(accuracy)
    }
    with open(CONFIG_PATH, "wb") as f:
        pickle.dump(config, f)

    print("\nFile berhasil dibuat:")
    print(f"- {MODEL_PATH}")
    print(f"- {TOKENIZER_PATH}")
    print(f"- {CONFIG_PATH}")

if __name__ == "__main__":
    main()
