import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def extract_numeric(value):
    if isinstance(value, str):
        match = re.search(r'\d+', value)
        return int(match.group()) if match else 0
    return int(value)

df = pd.read_excel("DATA IB DeepLearning - Januari-Juni.xlsx", sheet_name='Sheet1')

sales_columns = [col for col in df.columns if 'Terjual Bulan' in col]
df[sales_columns] = df[sales_columns].applymap(extract_numeric)

df['Stok Awal'] = df['Stok barang Tiap Bulan'].apply(extract_numeric)


def create_sequences(data, look_back=3):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

def forecast_lstm(item_sales, look_back=3, n_future=12):
    scaler = MinMaxScaler()
    item_scaled = scaler.fit_transform(item_sales.reshape(-1, 1))

    X, y = create_sequences(item_scaled, look_back)
    if len(X) == 0:
        return np.array([0] * n_future)

    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(look_back, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=100, verbose=0)

    forecast = []
    input_seq = item_scaled[-look_back:].reshape(1, look_back, 1)
    for _ in range(n_future):
        pred = model.predict(input_seq, verbose=0)[0][0]
        forecast.append(pred)
        input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

    return scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()


look_back = 3
n_future = 12
forecast_data = []

for idx, row in df.iterrows():
    item_name = row['Item']
    stok_awal = row['Stok Awal']
    item_sales = row[sales_columns].values.astype('float32')


    forecast = forecast_lstm(item_sales, look_back, n_future)

    stok_dinamis = [stok_awal for _ in range(n_future)]

    status = []
    for i in range(n_future):
        pred = forecast[i]
        stok = stok_dinamis[i]
        if pred > stok * 1.1:
            status.append("Kekurangan Stok")
        elif pred < stok * 0.9:
            status.append("Kelebihan Stok")
        else:
            status.append("Stok Cukup")

   
    row_data = {
        'Item': item_name,
        **{f'Stok Bulan ke-{i+13}': stok_dinamis[i] for i in range(n_future)},
        **{f'Pred Bulan ke-{i+13}': int(round(forecast[i])) for i in range(n_future)},
        **{f'Status Bulan ke-{i+13}': status[i] for i in range(n_future)},
    }
    forecast_data.append(row_data)


output_df = pd.DataFrame(forecast_data)
output_df.to_csv('forecast_stok_dinamis_1_tahun.csv', index=False)
print("✅ Forecast 1 tahun dengan stok dinamis disimpan di 'forecast_stok_dinamis_1_tahun.csv'.")


from datetime import datetime


libur_nasional_2026 = [
    "2026-01-01", "2026-02-19", "2026-03-19", "2026-04-03",
    "2026-04-17", "2026-05-01", "2026-05-14", "2026-05-25", "2026-05-26",
    "2026-08-17", "2026-09-15", "2026-12-25"
]

tanggal_2026 = pd.date_range("2026-01-01", "2026-12-31", freq="D")
libur_dates = set(pd.to_datetime(libur_nasional_2026))

df_diskon = pd.DataFrame({"Tanggal": tanggal_2026})
df_diskon["Diskon Berlaku"] = df_diskon["Tanggal"].isin(libur_dates)
df_diskon["Diskon (%)"] = df_diskon["Diskon Berlaku"].apply(lambda x: 10 if x else 0)

df_diskon.to_csv("jadwal_diskon_2026.csv", index=False)
print("📁 Jadwal diskon hari libur nasional disimpan di 'jadwal_diskon_2026.csv'")


sample = forecast_data[0]
plt.plot([i+13 for i in range(n_future)],
         [sample[f'Pred Bulan ke-{i+13}'] for i in range(n_future)],
         marker='o', label='Prediksi')
plt.plot([i+13 for i in range(n_future)],
         [sample[f'Stok Bulan ke-{i+13}'] for i in range(n_future)],
         linestyle='--', label='Stok')
plt.title(f"Prediksi dan Stok: {sample['Item']}")
plt.xlabel("Bulan ke")
plt.ylabel("Jumlah")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()