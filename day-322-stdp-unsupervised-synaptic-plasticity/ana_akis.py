"""
Day 322: Spike-Timing-Dependent Plasticity (STDP) & Unsupervised Learning
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu betik; Spike-Timing-Dependent Plasticity (STDP) Hebbian yerel denetimsiz öğrenme kuralının
etiketsiz nöromorfik kalıplar üzerindeki uzmanlaşmasını, sinaptik plastisitesini ve teşhis panosunu çalıştırır.
"""

import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import torch
from src.stdp_motoru import STDPUnsupervisedNetwork
from src.stdp_gorsellestirici import STDPGorsellestirici
from src.stdp_profilleyici import STDPProfilleyici


def sentetik_spiking_kalip_uretcisi(
    num_samples: int = 200,
    in_features: int = 16,
    time_steps: int = 30,
    seed: int = 42
) -> torch.Tensor:
    """
    STDP denetimsiz öğrenimi için 2 belirgin uzamsal kalıba sahip Poisson spike dizileri üretir.
    """
    torch.manual_seed(seed)
    # Kalıp A: İlk 8 nöron aktif
    pattern_a = torch.zeros(in_features)
    pattern_a[:8] = 0.85
    
    # Kalıp B: Son 8 nöron aktif
    pattern_b = torch.zeros(in_features)
    pattern_b[8:] = 0.85

    x = torch.zeros(num_samples, in_features)
    for i in range(num_samples):
        if i % 2 == 0:
            x[i] = pattern_a + torch.randn(in_features) * 0.05
        else:
            x[i] = pattern_b + torch.randn(in_features) * 0.05
    x = torch.clamp(x, 0.0, 1.0)

    # Poisson Spike Dizilerine Dönüştür
    rand_tensor = torch.rand((num_samples, time_steps, in_features))
    x_expanded = x.unsqueeze(1).expand(num_samples, time_steps, in_features)
    spikes_seq = (rand_tensor < x_expanded).float()
    return spikes_seq


def main():
    print("=" * 75, flush=True)
    print("🧠 DAY 322: Spike-Timing-Dependent Plasticity (STDP) & Denetimsiz Öğrenme", flush=True)
    print("=" * 75, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📌 Çalışma Donanımı: {device}", flush=True)

    in_features = 16
    out_features = 4
    time_steps = 30
    epochs = 5
    batch_size = 32

    # 1. Veri Hazırlığı
    spikes_dataset = sentetik_spiking_kalip_uretcisi(
        num_samples=320, in_features=in_features, time_steps=time_steps
    ).to(device)

    # 2. STDP Denetimsiz Ağ
    model = STDPUnsupervisedNetwork(
        in_features=in_features,
        out_features=out_features,
        time_steps=time_steps,
        stdp_lr=0.03
    ).to(device)

    initial_weights = model.weights.clone().detach().cpu().numpy()

    print("\n⚡ STDP Denetimsiz Hebbian Öğrenimi Başlatılıyor (Backprop Yok)...", flush=True)
    start_time = time.time()

    num_batches = len(spikes_dataset) // batch_size
    for epoch in range(1, epochs + 1):
        for b in range(num_batches):
            batch_spikes = spikes_dataset[b * batch_size : (b + 1) * batch_size]
            stdp_info = model(batch_spikes, train_stdp=True)

        w_current = model.weights.detach().cpu().numpy()
        mean_w = w_current.mean()
        std_w = w_current.std()
        print(f"  [Epoch {epoch:02d}/{epochs:02d}] Sinaptik Ağırlık Ortalama: {mean_w:.4f} | Std: {std_w:.4f}", flush=True)

    elapsed_time = time.time() - start_time
    print(f"\n✅ STDP Plastisite Eğitimi Tamamlandı! Toplam Süre: {elapsed_time:.2f} saniye", flush=True)

    # 3. Profilleme ve Teşhis Çıkarımı
    final_weights = model.weights.detach().cpu().numpy()
    eval_info = model(spikes_dataset[:batch_size], train_stdp=False)

    profiler_metrics = STDPProfilleyici.profille(
        initial_weights=initial_weights,
        final_weights=final_weights,
        spikes_seq=eval_info["spikes_seq"]
    )

    print("\n📊 STDP Plastisite Profilleme Metrikleri:", flush=True)
    print(f"  • Ortalama Ağırlık Kayması (Drift): {profiler_metrics['mean_weight_drift']:.4f}", flush=True)
    print(f"  • Maksimum Ağırlık Kayması:         {profiler_metrics['max_weight_drift']:.4f}", flush=True)
    print(f"  • Bimodal Kutupsallaşma Skoru:    {profiler_metrics['bimodality_score']:.4f}", flush=True)
    print(f"  • Ağırlık Dağılım Entropisi:        {profiler_metrics['weight_entropy']:.4f}", flush=True)
    print(f"  • Aktif Nöron Sayısı (WTA):        {profiler_metrics['active_neurons_count']} / {profiler_metrics['total_neurons_count']}", flush=True)
    print(f"  • WTA Uzmanlaşma Oranı:            %{profiler_metrics['specialization_ratio']*100:.1f}", flush=True)

    # 4. Teşhis Panosunu Çizme ve Kaydetme
    gorsellestirici = STDPGorsellestirici()
    cikti_yolu = gorsellestirici.teshis_panelini_ciz(
        initial_weights=initial_weights,
        final_weights=final_weights,
        stdp_info=eval_info,
        profiler_metrics=profiler_metrics
    )
    print(f"\n🖼️ 6-Panelli STDP Plastisite Grafiği Başarıyla Kaydedildi: [stdp_plastisite_paneli.png](file:///{os.path.abspath(cikti_yolu)})", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
