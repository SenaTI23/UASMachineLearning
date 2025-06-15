import streamlit as st
import pandas as pd
import joblib

# Set page config
st.set_page_config(page_title="Deteksi Penipuan Transaksi 💳", page_icon="🛡️", layout="wide")

# Load model
model = joblib.load("rf_model.joblib")

# Judul halaman utama
st.title("🛡️ Deteksi Penipuan Transaksi Online")
st.markdown("""
Aplikasi ini menggunakan model **Machine Learning (Random Forest)** untuk memprediksi apakah suatu transaksi online bersifat **penipuan** atau **aman** berdasarkan data numerik yang ditransformasi sebelumnya (PCA features V1-V28 + Amount).

**Silakan pilih metode input:**
""")

# Pilihan tab
tab1, tab2, tab3 = st.tabs(["📊 Input Manual", "📁 Upload CSV", "ℹ️ Penjelasan Fitur"])

# ================================
# 📊 TAB 1 - Input Manual
# ================================
with tab1:
    st.subheader("📊 Input Data Transaksi Manual")
    st.info("Masukkan nilai untuk fitur V1 hingga V28 dan jumlah transaksi (Amount)")

    # Input fitur
    input_data = {}
    cols = st.columns(3)
    for i in range(1, 29):
        col = cols[(i - 1) % 3]
        input_data[f'V{i}'] = col.number_input(f'V{i}', value=0.0, step=0.1)

    amount = st.number_input("Amount 💰", value=0.0, step=1.0)
    input_data["Amount"] = amount
    input_df = pd.DataFrame([input_data])

    if st.button("🔍 Prediksi Transaksi"):
        prediction = model.predict(input_df)
        proba = model.predict_proba(input_df)[0][1]

        st.markdown("### 🔎 Hasil Prediksi")
        if prediction[0] == 1:
            st.error(f"⚠️ Transaksi ini **TERDETEKSI SEBAGAI PENIPUAN** dengan probabilitas: **{proba:.2f}**")
        else:
            st.success(f"✅ Transaksi ini **AMAN** dengan probabilitas: **{1 - proba:.2f}**")

# ================================
# 📁 TAB 2 - Upload CSV
# ================================
with tab2:
    st.subheader("📁 Upload File CSV")
    st.info("Upload file CSV berisi satu atau lebih baris data transaksi (harus punya kolom: V1-V28 + Amount)")

    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("🧾 Data yang diupload:")
        st.dataframe(data)

        if st.button("🔍 Prediksi dari CSV"):
            prediction = model.predict(data)
            probabilities = model.predict_proba(data)[:, 1]

            results = data.copy()
            results['Prediction'] = prediction
            results['Fraud Probability'] = probabilities

            st.success("✅ Prediksi selesai. Hasil ditampilkan di bawah.")
            st.dataframe(results)

            # Download hasil
            csv_download = results.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Hasil Prediksi CSV", data=csv_download, file_name='hasil_prediksi.csv', mime='text/csv')

# ================================
# ℹ️ TAB 3 - Penjelasan Fitur
# ================================
with tab3:
    st.subheader("ℹ️ Penjelasan Fitur V1–V28 dan Amount")
    st.markdown("""
Dataset ini merupakan hasil transformasi PCA (Principal Component Analysis) dari fitur asli transaksi ke fitur-fitur baru yang lebih aman dan bebas identitas. Berikut penjelasannya:

- **V1 - V28**: Fitur hasil transformasi PCA yang merepresentasikan pola perilaku transaksi, seperti:
  - Pola waktu pengguna
  - Lokasi kartu
  - Jumlah transaksi per waktu
  - Korelasi akun atau merchant
  - Tidak bisa diinterpretasikan langsung (fitur anonim)
  
- **Amount**: Nilai nominal transaksi (dalam Euro atau mata uang asli)

Model dilatih dengan fitur ini dan label:
- `Class = 0` (transaksi normal)
- `Class = 1` (transaksi penipuan)
""")
