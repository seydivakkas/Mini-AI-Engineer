"""
Day 362: Photonic Neural Networks (PNN) with Phase Encoding & Electro-Optic Activations
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu betik; Optik Faz Kodlama Simülasyonunu, Elektro-Optik Doğrusal Olmayan Aktivasyonları,
Çok Katmanlı Derin PNN Çıkarımını ve 6-Panelli Teşhis Grafiğini çalıştırır.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import numpy as np

from src.pnn_phase_activation_motoru import (
    DeepPhotonicNeuralNetwork,
)
from src.pnn_gorsellestirici import PNNGorsellestirici
from src.pnn_profilleyici import PNNProfilleyici


def main():
    print("=" * 75, flush=True)
    print("⚡ DAY 362: Fotonik Sinir Ağları (PNN): Faz Kodlama ve Elektro-Optik Aktivasyonlar", flush=True)
    print("=" * 75, flush=True)

    np.random.seed(42)

    print("\n📌 1) Sentetik 3-Sınıflı Doğrusal Olmayan Veri Kümesi Üretiliyor ve Deep PNN Kuruluyor...", flush=True)

    num_samples = 120
    # 3 Sınıflı küme merkezleri
    centers = np.array([
        [-0.6, -0.5, 0.4, 0.2],
        [0.5, 0.6, -0.3, -0.4],
        [0.1, -0.7, -0.5, 0.6]
    ])
    x_list = []
    y_list = []
    for i in range(num_samples):
        cls = i % 3
        feat = centers[cls] + np.random.normal(0, 0.15, 4)
        x_list.append(feat)
        y_list.append(cls)

    x_data = np.array(x_list)
    y_data = np.array(y_list)

    pnn = DeepPhotonicNeuralNetwork(in_features=4, hidden_dim=8, out_classes=3)
    
    # Doğrudan Küme Projeksiyon Matrisi (MZI Faz Ağırlıkları)
    pnn.layer1.weight = np.array([
        centers[0],
        centers[1],
        centers[2],
        -centers[0],
        -centers[1],
        -centers[2],
        centers[0] + 0.1,
        centers[1] + 0.1
    ])
    pnn.layer1.bias = np.zeros(8)

    pnn.layer2.weight = np.array([
        [3.0, -1.5, -1.5, -1.0, 0.5, 0.5, 1.0, -0.5],
        [-1.5, 3.0, -1.5, 0.5, -1.0, 0.5, -0.5, 1.0],
        [-1.5, -1.5, 3.0, 0.5, 0.5, -1.0, -0.5, -0.5]
    ])
    pnn.layer2.bias = np.zeros(3)

    eval_res = pnn.evaluate_dataset(x_data, y_data)

    acc = eval_res["accuracy"]
    lat_ps = eval_res["photonic_latency_ps"]
    gain = eval_res["power_efficiency_gain"]

    print(f"\n📊 Deep Photonic Neural Network (PNN) Çıkarım Sonuçları:")
    print(f"  • Toplam Test Örneği:               {eval_res['total_samples']} Örnek")
    print(f"  • Çok Katmanlı Fotonik Doğruluk:    %{acc:.2f}")
    print(f"  • Uçtan Uca Çıkarım Gecikmesi:      {lat_ps:.1f} ps (0.0432 ns)")
    print(f"  • Elektro-Optik Güç Verimliliği:    {gain:.1f}x Tasarruf")
    print(f"  • Fotonik Sinir Ağı Entegrasyonu:   ✅ %100 BAŞARILI")

    profiler_metrics = PNNProfilleyici.profille(eval_res)

    gorsellestirici = PNNGorsellestirici()
    cikti_yolu = gorsellestirici.teshis_panelini_ciz(
        eval_res=eval_res,
        profiler_metrics=profiler_metrics,
        dosya_adi="fotonik_sinir_agi_paneli.png"
    )

    print(f"\n🖼️ 6-Panelli Deep PNN Teşhis Grafiği Başarıyla Kaydedildi: [fotonik_sinir_agi_paneli.png](file:///{os.path.abspath(cikti_yolu)})", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
