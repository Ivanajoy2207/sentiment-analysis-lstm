# Text Sentiment Analysis LSTM - Streamlit

Aplikasi ini memakai dataset `data.csv` dengan kolom:

- `text`
- `label`

Label:
- `0` = negatif
- `1` = positif

## Cara menjalankan lokal

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## File penting

- `data.csv` = dataset
- `train_model.py` = script training LSTM
- `app.py` = UI Streamlit
- `requirements.txt` = dependency untuk Streamlit Cloud

## Deploy ke Streamlit Community Cloud

1. Upload semua file ini ke repository GitHub.
2. Jalankan training lokal dulu dengan:
   ```bash
   python train_model.py
   ```
3. Pastikan file berikut ikut ter-upload ke GitHub:
   - `sentiment_lstm_model.h5`
   - `tokenizer.pkl`
   - `config.pkl`
   - `app.py`
   - `requirements.txt`
4. Buka https://streamlit.io/cloud
5. Klik **New app**.
6. Pilih repository GitHub Anda.
7. Isi **Main file path** dengan:
   ```text
   app.py
   ```
8. Klik **Deploy**.
9. Copy link aplikasi dan kirim ke forum.

## Catatan

Saya tidak bisa melakukan deploy langsung ke akun Streamlit Anda dari sini, tetapi kode ini sudah siap untuk di-upload dan di-deploy.
