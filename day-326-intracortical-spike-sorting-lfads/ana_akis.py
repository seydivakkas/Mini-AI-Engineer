"""
Day 326: Intracortical Spike Sorting & LFADS Latent Dynamics
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu betik; MEA elektrot verisinden spike tespiti, PCA+GMM ayrıştırma (spike sorting)
ve LFADS VAE modeli ile nöral popülasyon latent dinamiklerinin öğrenilmesini simüle eder.
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
import torch.optim as optim

from src.lfads_spike_motoru import (
    MEAWaveformSimulator,
    SpikeSorter,
    LFADSRecurrentGenerator,
)
from src.lfads_gorsellestirici import LFADSGorsellestirici
from src.lfads_profilleyici import LFADSProfilleyici


def main():
    print("=" * 75, flush=True)
    print("🧠 DAY 326: İntrakortikal Spike Ayrıştırma ve LFADS Nöral Latent Dinamikleri", flush=True)
    print("=" * 75, flush=True)

    # 1. MEA Ham Elektrot Sinyali Üretimi
    print("⚡ 1) MEA Çoklu Elektrot Dizilimi Sinyali Üretiliyor (30 kHz Örnekleme)...", flush=True)
    sim = MEAWaveformSimulator(sampling_rate=30000, duration_sec=0.5)
    v_raw, spike_idx_true, labels_true = sim.uret_ham_elektrot_verisi(num_units=3, seed=42)
    print(f"✅ Üretim Tamamlandı! Toplam Örnek Sayısı: {len(v_raw)} | Gerçek Spike Sayısı: {len(spike_idx_true)}", flush=True)

    # 2. Sinyal Filtreleme ve Spike Tespiti
    print("\n🔍 2) 300Hz-3000Hz Bandpass Filtreleme & Eşik Tespiti Yapılıyor...", flush=True)
    v_filtered = SpikeSorter.bant_geciren_filtre(v_raw)
    detected_spikes = SpikeSorter.spike_tespit_et(v_filtered, th_multiplier=3.8)
    print(f"✅ Eşik Tespiti Tamamlandı! Tespit Edilen Spike Sayısı: {len(detected_spikes)}", flush=True)

    # 3. Waveform Çıkarımı & PCA + GMM Spike Sorting
    print("\n📐 3) Waveform Çıkarımı ve PCA + GMM Spike Sorting (Ayrıştırma) Çalıştırılıyor...", flush=True)
    waveforms = SpikeSorter.dalga_formu_cikar(v_filtered, detected_spikes, window_size=48)
    features_2d, cluster_labels, pca, gmm = SpikeSorter.sort_spikes_pca_gmm(waveforms, n_clusters=3)
    print(f"✅ Spike Sorting Tamamlandı! 3 Nöron Birimine Ayrıştırıldı.", flush=True)
    print(f"  • PCA Açıklanan Varyans Oranı: %{np.sum(pca.explained_variance_ratio_)*100.0:.2f}")

    # 4. LFADS Nöral Popülasyon Simülasyonu ve Eğitimi
    print("\n🤖 4) LFADS (Latent Factor Analysis via Dynamical Systems) VAE Eğitiliyor...", flush=True)
    batch_size = 16
    time_steps = 50
    num_neurons = 20

    # Sentetik Popülasyon Spike Sayımı Y in R^(B x T x N)
    np.random.seed(42)
    syn_spikes_np = np.random.poisson(lam=0.2, size=(batch_size, time_steps, num_neurons)).astype(np.float32)
    x_spikes = torch.tensor(syn_spikes_np, dtype=torch.float32)

    lfads_model = LFADSRecurrentGenerator(num_neurons=num_neurons, latent_dim=12, hidden_dim=48)
    optimizer = optim.Adam(lfads_model.parameters(), lr=0.005)

    num_epochs = 10
    start_train_time = time.time()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        log_rates, factors, mu, logvar = lfads_model(x_spikes)
        loss, pois_nll, kl_div = LFADSRecurrentGenerator.compute_poisson_loss(x_spikes, log_rates, mu, logvar)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 2 == 0 or epoch == num_epochs - 1:
            print(f"  [Epoch {epoch+1:02d}/{num_epochs:02d}] Poisson Kayıp: {pois_nll.item():.4f} | KL Div: {kl_div.item():.4f}")

    elapsed_train = time.time() - start_train_time
    print(f"✅ LFADS Eğitimi Tamamlandı! Toplam Süre: {elapsed_train:.2f} saniye", flush=True)

    # Rekonstrüksiyon Değerleri
    with torch.no_grad():
        log_rates, factors, _, _ = lfads_model(x_spikes)
        rates = torch.exp(log_rates).numpy()
        factors_np = factors.numpy()

    # 5. Profilleme Metrikleri ve Görselleştirme
    profiler_metrics = LFADSProfilleyici.profille(
        total_spikes=len(detected_spikes),
        num_sorted_units=3,
        pca_explained_variance_ratio=pca.explained_variance_ratio_,
        poisson_loss=pois_nll.item(),
        kl_div=kl_div.item(),
        latency_ms=elapsed_train * 1000.0 / num_epochs
    )

    print("\n📊 LFADS & Spike Sorting Profilleme Metrikleri:", flush=True)
    print(f"  • Toplam Ayrıştırılan Spike:     {profiler_metrics['total_spikes']} Adet", flush=True)
    print(f"  • PCA Varyans Açıklama Oranı:    %{profiler_metrics['pca_var_pct']:.2f}", flush=True)
    print(f"  • LFADS Poisson NLL Kayıp:       {profiler_metrics['poisson_loss']:.4f}", flush=True)
    print(f"  • BCI Dekodlama Hazır Bulunurluk: %{profiler_metrics['bci_decoding_readiness']:.1f}", flush=True)

    # 6-Panelli Görsel Teşhis Grafiği
    gorsellestirici = LFADSGorsellestirici()
    spikes_raster_sample = syn_spikes_np[0].T  # (Neurons, Time)
    reconstructed_rates_sample = rates[0]       # (Time, Neurons)
    latent_factors_sample = factors_np[0]       # (Time, Latent)

    cikti_yolu = gorsellestirici.teshis_panelini_ciz(
        raw_voltage=v_filtered,
        spike_indices=detected_spikes,
        waveforms_2d=features_2d,
        cluster_labels=cluster_labels,
        waveforms=waveforms,
        spikes_raster=spikes_raster_sample,
        latent_factors=latent_factors_sample,
        reconstructed_rates=reconstructed_rates_sample,
        profiler_metrics=profiler_metrics
    )

    print(f"\n🖼️ 6-Panelli LFADS Teşhis Grafiği Başarıyla Kaydedildi: [lfads_spike_paneli.png](file:///{os.path.abspath(cikti_yolu)})", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
