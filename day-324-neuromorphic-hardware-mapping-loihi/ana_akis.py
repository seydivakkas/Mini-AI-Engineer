"""
Day 324: Neuromorphic Hardware Mapping (Intel Loihi 2 & SynSense)
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu betik; PyTorch SNN modelinin Intel Loihi 2 Neuro-Core çip mimarisine haritalanmasını,
INT8 sabitleştirilmiş kuantizasyonunu, AER paket yönlendirme simülasyonunu ve 6-panelli teşhis panosunu içerir.
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
import torch
from src.loihi_mapper import (
    NeuromorphicHardwareMapper,
    AERPacketRouter,
    AERPacket
)
from src.loihi_gorsellestirici import LoihiGorsellestirici
from src.loihi_profilleyici import LoihiProfilleyici


def main():
    print("=" * 75, flush=True)
    print("🧠 DAY 324: Intel Loihi 2 & SynSense Nöromorfik Donanım Eşleme (Hardware Mapping)", flush=True)
    print("=" * 75, flush=True)

    # Simülasyon Hiperparametreleri
    out_neurons = 220
    in_features = 128
    mesh_rows = 4
    mesh_cols = 4
    max_neurons_per_core = 64

    print(f"📌 Hedef Donanım: Intel Loihi 2 Mesh ({mesh_rows}x{mesh_cols} = {mesh_rows*mesh_cols} Çekirdek)", flush=True)
    print(f"📌 SNN Model Katmanı: {out_neurons} Nöron x {in_features} Girdi Sinapsı", flush=True)

    # 1. FP32 Sinaptik Ağırlık Matrisi Oluştur
    np.random.seed(42)
    weights_fp32 = np.random.randn(out_neurons, in_features).astype(np.float32) * 0.4

    # 2. Donanım Haritalama ve INT8 Kuantizasyonu
    print("\n⚡ PyTorch SNN Ağırlıkları Neuro-Core Mesh Çipine Haritalanıyor...", flush=True)
    start_time = time.time()

    mapper = NeuromorphicHardwareMapper(
        mesh_rows=mesh_rows,
        mesh_cols=mesh_cols,
        max_neurons_per_core=max_neurons_per_core
    )
    mapping_info = mapper.map_snn_weights(weights_fp32)
    elapsed_time = time.time() - start_time

    print(f"✅ Haritalama Tamamlandı! Süre: {elapsed_time*1000.0:.2f} ms", flush=True)
    print(f"  • Kullanılan Çekirdek Sayısı:  {mapping_info['used_cores']} / {mapping_info['total_cores']}")
    print(f"  • Donanım Doluluk Oranı:      %{mapping_info['core_utilization_pct']:.1f}")
    print(f"  • INT8 Kuantizasyon SQNR:     {mapping_info['sqnr_db']:.2f} dB")

    # 3. AER Paket Yönlendirme Simülasyonu
    print("\n📡 AER (Address Event Representation) Spike Paket Yönlendirmesi Simüle Ediliyor...", flush=True)
    aer_packets = []
    num_spikes = 150
    for i in range(num_spikes):
        src_idx = np.random.randint(0, mapping_info["used_cores"])
        dst_idx = np.random.randint(0, mapping_info["used_cores"])
        if src_idx != dst_idx:
            pkt = AERPacketRouter.route_spike(
                src_core=mapper.cores[src_idx],
                dst_core=mapper.cores[dst_idx],
                neuron_id=np.random.randint(0, out_neurons),
                timestamp_us=np.random.uniform(0, 10000)
            )
            aer_packets.append(pkt)

    # 4. Donanım Profilleme Metrikleri
    profiler_metrics = LoihiProfilleyici.profille(
        mapping_info=mapping_info,
        aer_packets=aer_packets,
        total_sops=180000,
        total_flops=600000
    )

    print("\n📊 Loihi 2 Donanım & Enerji Profilleme Metrikleri:", flush=True)
    print(f"  • Ortalama Manhattan Hop Mesafesi: {profiler_metrics['avg_hop_distance']:.2f} Hop", flush=True)
    print(f"  • Maksimum Hop Mesafesi:            {profiler_metrics['max_hop_distance']} Hop", flush=True)
    print(f"  • Tahmini Loihi 2 Enerjisi:         {profiler_metrics['loihi_energy_uj']:.3f} uJ", flush=True)
    print(f"  • Tahmini GPU Enerjisi:             {profiler_metrics['gpu_energy_uj']:.3f} uJ", flush=True)
    print(f"  • Donanım Enerji Tasarrufu Kazancı: {profiler_metrics['energy_saving_x']:.2f}x Verim", flush=True)

    # 5. Teşhis Panosunu Çizme ve Kaydetme
    gorsellestirici = LoihiGorsellestirici()
    cikti_yolu = gorsellestirici.teshis_panelini_ciz(
        mapping_info=mapping_info,
        weights_fp32=weights_fp32,
        weights_dequant=mapping_info["dequantized_weights"],
        aer_packets=aer_packets,
        profiler_metrics=profiler_metrics
    )
    print(f"\n🖼️ 6-Panelli Loihi 2 Teşhis Grafiği Başarıyla Kaydedildi: [loihi_donanim_paneli.png](file:///{os.path.abspath(cikti_yolu)})", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
