import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Setup Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Prediksi Harga Daging Ayam Jabar", layout="wide")
st.title("📊 Prediksi Harga Daging Ayam Provinsi Jawa Barat")
st.write("Aplikasi ini memprediksi tren harga daging ayam menggunakan integrasi arsitektur Deep Learning **CNN - LSTM**.")

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Sembunyikan warning tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Konfigurasi Awal Global Seed
np.random.seed(42)
tf.random.set_seed(42)

CSV_FILE = 'komoditas_daging_ayam_2022_2026.csv'
LOOK_BACK = 30


# ---------------------------------------------------------
# 1. TAMPILAN AWAL: INTERFASE KONTROL FORM UPDATE DATA & KONFIGURASI
# ---------------------------------------------------------
col_input, col_config = st.columns(2)

with col_input:
    st.header("📥 Tambah / Update Data CSV")
    with st.form(key='update_csv_form', clear_on_submit=True):
        input_date = st.date_input("Pilih Tanggal Baru/Update")
        input_price = st.number_input("Harga Daging Ayam (IDR)", min_value=0, step=500)
        submit_btn = st.form_submit_button(label="Simpan ke Dataset")
        
        if submit_btn:
            formatted_date = input_date.strftime('%d/%m/%Y')
            date_scraped = pd.Timestamp.now().strftime('%d/%m/%Y') 
            
            # Baca CSV asli
            df_raw = pd.read_csv(CSV_FILE, sep=';')
            df_raw.columns = df_raw.columns.str.strip()
            
            df_raw['Date_Param_Parsed'] = pd.to_datetime(df_raw['Date_Param'], format='%d/%m/%Y', errors='coerce')
            dup_idx = df_raw[(df_raw['Date_Param_Parsed'] == pd.to_datetime(input_date)) & (df_raw['Province_Name'] == 'Jawa Barat')].index
            
            if not dup_idx.empty:
                df_raw.loc[dup_idx, 'Price'] = input_price
                df_raw = df_raw.drop(columns=['Date_Param_Parsed'])
                df_raw.to_csv(CSV_FILE, index=False, sep=';')
                st.success(f"🔄 Data Jabar pada {formatted_date} berhasil diperbarui!")
            else:
                columns_order = [col for col in df_raw.columns if col != 'Date_Param_Parsed']
                df_append = pd.DataFrame(columns=columns_order)
                
                df_append.loc[0, 'Date_Scraped'] = date_scraped
                df_append.loc[0, 'Date_Param'] = formatted_date
                df_append.loc[0, 'Commodity_ID'] = 2
                df_append.loc[0, 'Commodity_Name'] = 'Daging Ayam'
                df_append.loc[0, 'Province_ID'] = 12
                df_append.loc[0, 'Province_Name'] = 'Jawa Barat'
                df_append.loc[0, 'Price'] = float(input_price)
                df_append.loc[0, 'Price_Type'] = 1
                
                df_raw = df_raw.drop(columns=['Date_Param_Parsed'])
                df_append.to_csv(CSV_FILE, mode='a', header=False, index=False, sep=';')
                st.success(f"✅ Data baru pada {formatted_date} berhasil ditambahkan!")
            
            st.cache_resource.clear()
            st.rerun()

with col_config:
    st.header("🔮 Konfigurasi Peramalan Masa Depan")
    pilihan_bulan = st.selectbox(
        "Pilih Jangka Waktu Prediksi:",
        options=[1, 2, 3],
        format_func=lambda x: f"{x} Bulan ({x * 30} Hari)"
    )
    FUTURE_DAYS = pilihan_bulan * 30

st.markdown("---")


# ---------------------------------------------------------
# CACHING MODEL & PREPROCESSING
# ---------------------------------------------------------
@st.cache_resource
def load_data_and_train_model():
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"File '{CSV_FILE}' tidak ditemukan di direktori aktif.")
        
    df = pd.read_csv(CSV_FILE, sep=';')
    df.columns = df.columns.str.strip()
    
    df['Date_Param'] = pd.to_datetime(df['Date_Param'], format='%d/%m/%Y', errors='coerce')
    df = df.dropna(subset=['Date_Param'])
    
    df_jabar = df[df['Province_Name'] == 'Jawa Barat'][['Date_Param', 'Price']].sort_values(by='Date_Param')
    df_jabar = df_jabar.groupby('Date_Param').mean().reset_index()
    df_jabar.set_index('Date_Param', inplace=True)
    
    data_prices = df_jabar[['Price']].values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(data_prices)
    
    train_size = int(len(scaled_prices) * 0.8)
    train_data, test_data = scaled_prices[0:train_size], scaled_prices[train_size:]
    
    def create_dataset(dataset, look_back=30):
        X, Y = [], []
        for i in range(len(dataset) - look_back):
            a = dataset[i:(i + look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)
    
    X_train, y_train = create_dataset(train_data, LOOK_BACK)
    X_test, y_test = create_dataset(test_data, LOOK_BACK)
    
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(LOOK_BACK, 1)),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        LSTM(units=64, return_sequences=False),
        Dropout(0.2),
        Dense(units=32, activation='relu'),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=0
    )
    
    test_predict = model.predict(X_test, verbose=0)
    predictions_actual = scaler.inverse_transform(test_predict)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test_actual, predictions_actual)),
        'mae': mean_absolute_error(y_test_actual, predictions_actual),
        'mape': mean_absolute_percentage_error(y_test_actual, predictions_actual) * 100,
        'r2': r2_score(y_test_actual, predictions_actual)
    }
    
    return model, df_jabar, scaled_prices, scaler, train_size, history.history, test_predict, metrics


with st.spinner("Mengunduh data dan melatih model Deep Learning CNN-LSTM... (Harap tunggu)"):
    try:
        model, df_jabar, scaled_prices, scaler, train_size, history_loss, test_predict, metrics = load_data_and_train_model()
        st.success("Model AI CNN-LSTM Berhasil Dimuat dan Dilatih!")
    except Exception as e:
        st.error(f"Gagal memproses data atau model. Error: {e}")
        st.stop()

# ---------------------------------------------------------
# 2. SETELAH INPUT: MENAMPILKAN GRAFIK TERLEBIH DAHULU, LALU EVALUASI METRIK
# ---------------------------------------------------------
st.header("📊 Analisis & Performa Model CNN-LSTM")

# Baris Grafik Perbandingan Loss & Hasil Testing (Sekarang di paling atas)
col_loss, col_eval = st.columns(2)

with col_loss:
    st.subheader("📉 Grafik Pergerakan Loss Function")
    fig_loss, ax_loss = plt.subplots(figsize=(8, 4))
    ax_loss.plot(history_loss['loss'], label='Training Loss', color='blue', linewidth=2)
    ax_loss.plot(history_loss['val_loss'], label='Validation Loss', color='orange', linewidth=2)
    ax_loss.set_xlabel('Epochs')
    ax_loss.set_ylabel('Loss (MSE)')
    ax_loss.legend()
    ax_loss.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig_loss)
    
with col_eval:
    st.subheader("📉 Hasil Prediksi vs Data Aktual Sebenarnya (Data Test)")
    fig_eval, ax_eval = plt.subplots(figsize=(8, 4))
    test_dates = df_jabar.index[train_size + LOOK_BACK:]
    predictions_actual = scaler.inverse_transform(test_predict)
    ax_eval.plot(df_jabar.index, df_jabar['Price'], label='Harga Aktual Sebenarnya', color='blue', alpha=0.4)
    ax_eval.plot(test_dates, predictions_actual, label='Prediksi Model (Data Test)', color='red', linestyle='--')
    ax_eval.set_xlabel("Tanggal")
    ax_eval.set_ylabel("Harga (IDR)")
    ax_eval.legend()
    ax_eval.grid(True, linestyle=':', alpha=0.6)
    st.pyplot(fig_eval)

st.markdown("---")

# Evaluasi Akurasi Model (Sekarang diletakkan di bawah grafik)
st.subheader("📋 Evaluasi Akurasi Model")
m1, m2, m3, m4 = st.columns(4)
m1.metric("R2 Score (Akurasi)", f"{metrics['r2']:.4f} ({metrics['r2']*100:.2f}%)")
m2.metric("MAPE (Error %)", f"{metrics['mape']:.2f}%")
m3.metric("MAE", f"Rp {metrics['mae']:.2f}")
m4.metric("RMSE", f"Rp {metrics['rmse']:.2f}")

st.markdown("---")


# ---------------------------------------------------------
# 3. TOMBOL EKSEKUSI PREDIKSI MASA DEPAN
# ---------------------------------------------------------
st.header("🚀 Peramalan Nilai Harga Masa Depan")
pred_button = st.button("Hitung Estimasi Peramalan Masa Depan", use_container_width=True)


# ---------------------------------------------------------
# 4. AKHIR: MENAMPILKAN GRAFIK DAN TABEL PREDIKSI (JIKA TOMBOL DIKLIK)
# ---------------------------------------------------------
if pred_button:
    last_30_days = scaled_prices[-LOOK_BACK:]
    current_batch = last_30_days.reshape((1, LOOK_BACK, 1))
    
    future_predictions_scaled = []
    
    for i in range(FUTURE_DAYS):
        current_pred = model.predict(current_batch, verbose=0)
        future_predictions_scaled.append(current_pred[0, 0])
        new_input = np.append(current_batch[0, 1:, 0], current_pred[0, 0])
        current_batch = new_input.reshape((1, LOOK_BACK, 1))
        
    future_predictions_actual = scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))
    
    last_date = df_jabar.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=FUTURE_DAYS)
    df_future = pd.DataFrame(data=future_predictions_actual, index=future_dates, columns=['Price_Forecast'])
    
    # Tampilkan Visualisasi Hasil Integrasi Masa Depan
    st.subheader(f"📈 Hasil Proyeksi Grafik {pilihan_bulan} Bulan ke Depan")
    fig_future, ax_future = plt.subplots(figsize=(15, 5))
    
    df_jabar_zoom = df_jabar.loc[df_jabar.index >= '2025-06-01']
    
    ax_future.plot(df_jabar_zoom.index, df_jabar_zoom['Price'], label='Harga Historis Aktual', color='blue', linewidth=2)
    ax_future.plot(df_future.index, df_future['Price_Forecast'], label=f'Estimasi Prediksi {pilihan_bulan} Bulan ke Depan', color='crimson', linestyle='--', linewidth=2.5)
    ax_future.axvline(x=last_date, color='black', linestyle=':', alpha=0.7)
    ax_future.text(last_date, df_jabar['Price'].iloc[-1], ' Mulai Masa Depan', color='black', fontsize=10, fontweight='bold')
    
    ax_future.set_title(f"Proyeksi Tren Harga Daging Ayam Jawa Barat Menggunakan CNN - LSTM", fontsize=12, fontweight='bold')
    ax_future.set_xlabel("Tanggal / Waktu")
    ax_future.set_ylabel("Harga (IDR)")
    ax_future.legend(loc='upper left')
    ax_future.grid(True, linestyle=':', alpha=0.6)
    st.pyplot(fig_future)
    
    # Tampilkan Tabel Hasil Peramalan (Paling bawah)
    st.subheader(f"📋 Tabel Riwayat Estimasi Harga ({pilihan_bulan} Bulan Masa Depan)")
    st.dataframe(df_future.rename(columns={'Price_Forecast': 'Estimasi Harga Komoditas (IDR)'}), use_container_width=True)