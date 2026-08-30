"""
Day 331: Astrocyte-Neuron Metabolic Interaction & Slow Neuromodulation
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu betik; Astrosit İçi Kalsiyum Salınımlarını, Yavaş Nöromodülasyon P_release Değişimini,
ANLS Laktat Mekiği Enerji İkmalini ve 6-panelli teşhis panosunu çalıştırır.
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
from src.astrocyte_neuron_motoru import (
    AstrocyteCalciumModel,
    TripartiteSynapse,
    AstrocyteMetabolicNetwork,
)
from src.astrocyte_gorsellestirici import AstrocyteGorsellestirici
from src.astrocyte_profilleyici import AstrocyteProfilleyici


def main():
    print("=" * 75, flush=True)
    print("🧠 DAY 331: Astrosit-Nöron Metabolik Etkileşimi ve Yavaş Nöromodülasyon", flush=True)
    print("=" * 75, flush=True)

    np.random.seed(42)
    num_neurons = 10
    num_steps = 100

    print(f"📌 Astrosit-Nöron Ağ Kurulumu: {num_neurons} Üçlü Sinaps (Tripartite Synapse), {num_steps} Simülasyon Adımı", flush=True)

    # 1. Simülasyon Verilerinin Toplanması
    print("\n⚡ 1) Astrosit İçi [Ca2+] ve Yavaş Nöromodülasyon Simülasyonu Başlatılıyor...", flush=True)
    network = AstrocyteMetabolicNetwork(num_neurons=num_neurons)

    ca_trace = []
    p_release_trace = []
    atp_trace = []
    spikes_history = []
    ca_spikes_count = 0

    for t in range(num_steps):
        # Rastgele nöronal ateşleme (t=20-60 arasında yüksek burst)
        if 20 <= t <= 60:
            spike_vec = np.random.rand(num_neurons) < 0.65
        else:
            spike_vec = np.random.rand(num_neurons) < 0.15

        res = network.simulate_step(spike_vec)
        
        ca_trace.append(res["mean_ca"])
        p_release_trace.append(res["mean_p_release"])
        atp_trace.append(res["mean_atp"])
        spikes_history.append(spike_vec)

        if res["mean_ca"] >= 0.35:
            ca_spikes_count += 1

    ca_trace_np = np.array(ca_trace, dtype=np.float32)
    p_release_np = np.array(p_release_trace, dtype=np.float32)
    atp_trace_np = np.array(atp_trace, dtype=np.float32)
    spikes_history_np = np.array(spikes_history, dtype=np.float32)

    print(f"✅ Simülasyon Tamamlandı! Toplam Kalsiyum Dalga Tepe Noktası: {ca_spikes_count}", flush=True)

    # 2. Profilleme ve Teşhis Panosu
    profiler_metrics = AstrocyteProfilleyici.profille(
        ca_spikes_count=ca_spikes_count,
        mean_p_release=float(np.mean(p_release_np)),
        mean_atp_level=float(np.mean(atp_trace_np))
    )

    print("\n📊 Astrosit-Nöron Profilleme Metrikleri:", flush=True)
    print(f"  • Kalsiyum Dalga Sıklığı:         {profiler_metrics['ca_spikes_count']} Tepe", flush=True)
    print(f"  • Ortalama Sinaptik Salınım (P):  {profiler_metrics['mean_p_release']:.3f}", flush=True)
    print(f"  • ANLS ATP Enerji İkmalı:        %{profiler_metrics['mean_atp_level']:.2f}", flush=True)

    gorsellestirici = AstrocyteGorsellestirici()
    cikti_yolu = gorsellestirici.teshis_panelini_ciz(
        ca_trace=ca_trace_np,
        p_release_trace=p_release_np,
        atp_trace=atp_trace_np,
        spikes_history=spikes_history_np,
        profiler_metrics=profiler_metrics,
        dosya_adi="astrosit_noron_paneli.png"
    )

    print(f"\n🖼️ 6-Panelli Astrosit-Nöron Teşhis Grafiği Başarıyla Kaydedildi: [astrosit_noron_paneli.png](file:///{os.path.abspath(cikti_yolu)})", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
