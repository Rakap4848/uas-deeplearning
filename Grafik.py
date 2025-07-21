import pandas as pd
import matplotlib.pyplot as plt
import os
import re

df = pd.read_csv("forecast_stok_dinamis_1_tahun.csv")

os.makedirs("grafik_prediksi", exist_ok=True)

n_future = 12

for idx, row in df.iterrows():
    item_name = row['Item']

    forecast = [row[f'Pred Bulan ke-{i+13}'] for i in range(n_future)]
    stok = [row[f'Stok Bulan ke-{i+13}'] for i in range(n_future)]

    plt.figure(figsize=(10, 4))
    plt.plot([i+13 for i in range(n_future)], forecast, marker='o', label='Prediksi Penjualan')
    plt.plot([i+13 for i in range(n_future)], stok, linestyle='--', label='Stok (dinamis)')
    plt.title(f"Prediksi vs Stok: {item_name}")
    plt.xlabel("Bulan ke-")
    plt.ylabel("Jumlah")
    plt.legend()
    plt.grid(True)

    safe_name = re.sub(r'[^\w\-\. ]', '', item_name)
    plt.savefig(f"grafik_prediksi/{safe_name}.png")
    plt.close()

print("✅ Semua grafik berhasil dibuat dan disimpan di folder 'grafik_prediksi/'")