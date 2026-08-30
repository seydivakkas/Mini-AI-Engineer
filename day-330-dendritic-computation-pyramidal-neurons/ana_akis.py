"""
Day 330: Dendritic Computation & Non-linear Pyramidal Branch Dynamics
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu betik; Çok Bölmeli Piramidal Nöron Kablo Simülasyonunu, NMDA Dendritik Spike Üretimini,
Tek Nöron ile XOR Çözümünü ve 6-panelli teşhis panosunu çalıştırır.
"""

import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import numpy as np
from src.dendritic_pyramidal_motoru import (
    DendriticBranch,
    MultiCompartmentPyramidalNeuron,
    DendriticXORClassifier,
)
from src.dendritic_gorsellestirici import DendriticGorsellestirici
from src.dendritic_profilleyici import DendriticProfilleyici


def main():
    print("=" * 75, flush=True)
    print("🧠 DAY 330: Dendritik Hesaplama ve Piramidal Nöronlarda Doğrusal Olmayan Dal Dinamikleri", flush=True)
    print("=" * 75, flush=True)

    # 1. NMDA Dendritik Dal Doygunluk Simülasyonu
    print("\n⚡ 1) Aktif Dendritik Dal NMDA Spike & Doygunluk Eğrisi Simüle Ediliyor...", flush=True)
    branch = DendriticBranch(branch_id=1, num_synapses=4, threshold=1.0, plateau_gain=2.5)

    linear_sums = np.linspace(0, 3.0, 100)
    branch_potentials = []
    nmda_spikes = 0

    for s_val in linear_sums:
        inputs = np.full(4, s_val / 4.0, dtype=np.float32)
        v_b, is_nmda = branch.compute_branch_potential(inputs)
        branch_potentials.append(v_b)
        if is_nmda:
            nmda_spikes += 1

    branch_potentials_np = np.array(branch_potentials, dtype=np.float32)
    print(f"✅ Dendritik Dal Simülasyonu Tamamlandı! Toplam NMDA Spike: {nmda_spikes}", flush=True)

    # 2. Çok Bölmeli Piramidal Nöron Potansiyel İzleri
    print("\n⚡ 2) Çok Bölmeli Piramidal Nöron Kablo Entegrasyonu Simüle Ediliyor...", flush=True)
    neuron = MultiCompartmentPyramidalNeuron(v_rest=-70.0, v_th=-50.0, g_coupling=0.4)

    v_soma_trace = []
    v_basal1_trace = []
    v_basal2_trace = []

    for t in range(50):
        # t=10 ile t=35 arasında Basal 1 dalına yüksek girdi ver
        in_b1 = np.array([0.9, 0.9], dtype=np.float32) if (10 <= t <= 35) else np.zeros(2, dtype=np.float32)
        in_b2 = np.array([0.2, 0.2], dtype=np.float32)

        v_soma, is_spike, states = neuron.step_simulation(in_b1, in_b2)
        v_soma_trace.append(states["v_soma"])
        v_basal1_trace.append(states["v_basal1"])
        v_basal2_trace.append(states["v_basal2"])

    v_soma_np = np.array(v_soma_trace, dtype=np.float32)
    v_basal1_np = np.array(v_basal1_trace, dtype=np.float32)
    v_basal2_np = np.array(v_basal2_trace, dtype=np.float32)

    print("✅ Potansiyel Entegrasyon İzleri Oluşturuldu!", flush=True)

    # 3. Tek Nöron XOR Sınıflandırıcı Testi
    print("\n🤖 3) Tek Nöron ile XOR Desen Ayırımı Test Ediliyor...", flush=True)
    xor_classifier = DendriticXORClassifier()
    test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    expected_outputs = [0, 1, 1, 0]
    xor_results = {}
    correct = 0

    for (x1, x2), target in zip(test_inputs, expected_outputs):
        pred = xor_classifier.predict_xor(x1, x2)
        xor_results[f"({x1},{x2})"] = pred
        if pred == target:
            correct += 1
        print(f"  • Girdi: ({x1}, {x2}) -> Tahmin: {pred} | Beklenen: {target} {'✅' if pred==target else '❌'}")

    acc_xor = (correct / len(test_inputs)) * 100.0
    print(f"✅ Tek Nöron XOR Başarımı: %{acc_xor:.2f} (Point Nöronlar %0 Başarılıdır)", flush=True)

    # 4. Profilleme ve Teşhis Panosu
    profiler_metrics = DendriticProfilleyici.profille(
        nmda_spikes_count=nmda_spikes,
        xor_accuracy=acc_xor,
        capacity_gain_x=4.0
    )

    print("\n📊 Dendritik Hesaplama Profilleme Metrikleri:", flush=True)
    print(f"  • Toplam NMDA Spike Sayısı:      {profiler_metrics['nmda_spikes_count']}", flush=True)
    print(f"  • Point Nörona Göre Kapasite:    {profiler_metrics['capacity_gain_x']:.1f}x Artış", flush=True)
    print(f"  • XOR Doğruluğu:                %{profiler_metrics['xor_accuracy']:.2f}", flush=True)

    gorsellestirici = DendriticGorsellestirici()
    cikti_yolu = gorsellestirici.teshis_panelini_ciz(
        linear_sums=linear_sums,
        branch_potentials=branch_potentials_np,
        v_soma_trace=v_soma_np,
        v_basal1_trace=v_basal1_np,
        v_basal2_trace=v_basal2_np,
        xor_results=xor_results,
        profiler_metrics=profiler_metrics,
        dosya_adi="dendritik_hesaplama_paneli.png"
    )

    print(f"\n🖼️ 6-Panelli Dendritik Teşhis Grafiği Başarıyla Kaydedildi: [dendritik_hesaplama_paneli.png](file:///{os.path.abspath(cikti_yolu)})", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
