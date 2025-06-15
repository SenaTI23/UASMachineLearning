
import streamlit as st
import pandas as pd
import joblib

# Load model dari file joblib
model = joblib.load('rf_model.joblib')

st.title("Prediksi Penipuan Transaksi Online")

def user_input_features():
    data = {}
    for i in range(1, 29):
        data[f'V{i}'] = st.number_input(f'V{i}', value=0.0)
    data['Amount'] = st.number_input('Amount', value=0.0)
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

if st.button('Prediksi'):
    prediction = model.predict(input_df)
    proba = model.predict_proba(input_df)[0][1]
    if prediction[0] == 1:
        st.error(f"Transaksi TERDETEKSI sebagai PENIPUAN dengan probabilitas {proba:.2f}")
    else:
        st.success(f"Transaksi DIPREDIKSI AMAN dengan probabilitas {1-proba:.2f}")
