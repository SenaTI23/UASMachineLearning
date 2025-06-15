import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("rf_model.joblib")

# Konfigurasi halaman
st.set_page_config(page_title="Deteksi Penipuan Transaksi 💳", page_icon="🛡️", layout="wide")

st.title("🛡️ Deteksi Penipuan Transaksi Online")
st.markdown("""
Aplikasi ini memprediksi apakah suatu transaksi online bersifat **penipuan** atau **normal** menggunakan model Machine Learning `Random Forest`.

Silakan pilih metode input data:
""")

# Tab
tab1, tab2, tab3 = st.tabs(["📊 Input Manual", "📁 Upload CSV", "ℹ️ Penjelasan Fitur"])

# ===============================
# 📊 TAB 1: Input Manual
# ===============================
with tab1:
    st.subheader("📊 Input Data Transaksi Manual")
    st.info("Masukkan nilai untuk V1 hingga V28 dan Amount")

    input_data = {}
    cols = st.columns(3)
    for i in range(1, 29):
        col = cols[(i - 1) % 3]
        input_data[f"V{i}"] = col.number_input(f"V{i}", value=0.0, step=0.1)

    input_data["Amount"] = st.number_input("Amount 💰", value=0.0, step=1.0)

    input_df = pd.DataFrame([input_data])

    # Urutkan kolom agar cocok dengan model
    expected_cols = [f'V{i}' for i in range(1, 29)] + ['Amount']
    input_df = input_df[expected_cols]

    if st.button("🔍 Prediksi Transaksi"):
        prediction = model.predict(input_df)
        proba = model.predict_proba(input_df)[0][1]

        st.markdown("### 🔎 Hasil Prediksi")
        if prediction[0] == 1:
            st.error(f"⚠️ Transaksi ini **TERDETEKSI SEBAGAI PENIPUAN** dengan probabilitas: **{proba:.2f}**")
        else:
            st.success(f"✅ Transaksi ini **AMAN** dengan probabilitas: **{1 - proba:.2f}**")

# ===============================
# 📁 TAB 2: Upload CSV
# ===============================
with tab2:
    st.subheader("📁 Upload File CSV")
    st.info("Unggah file CSV berisi kolom: V1–V28 + Amount")

    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        try:
            expected_cols = [f'V{i}' for i in range(1, 29)] + ['Amount']
            df = df[expected_cols]  # pastikan urutan kolom sesuai

            st.write("📄 Data transaksi yang diupload:")
            st.dataframe(df)

            if st.button("🔍 Prediksi dari CSV"):
                preds = model.predict(df)
                probas = model.predict_proba(df)[:, 1]

                df["Prediction"] = preds
                df["Fraud Probability"] = probas

                st.success("✅ Prediksi selesai. Hasil ditampilkan di bawah.")
                st.dataframe(df)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Hasil Prediksi", data=csv, file_name="hasil_prediksi.csv", mime="text/csv")

        except Exception as e:
            st.error("❌ Format kolom tidak sesuai. Pastikan kolom: V1–V28 dan Amount.")

# ===============================
# ℹ️ TAB 3: Penjelasan Fitur
# ===============================
with tab3:
    st.subheader("ℹ️ Penjelasan Fitur Dataset")
    st.markdown("""
**Penjelasan singkat fitur:**

- `V1` sampai `V28`: Fitur hasil transformasi PCA (Principal Component Analysis) dari data asli transaksi. Ini digunakan untuk menyembunyikan data sensitif (nama, lokasi, waktu, dll).
- `Amount`: Jumlah nilai transaksi dalam satuan mata uang asli (misal: Euro).
- Tidak ada fitur waktu/identitas karena privasi data.

**Label asli (saat training model):**

- `Class = 0` → Transaksi Normal  
- `Class = 1` → Transaksi Penipuan

""")
