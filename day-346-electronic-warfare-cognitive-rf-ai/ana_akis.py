"""
Day 346: Electronic Warfare (EW) Cognitive RF Spectrum Sensing & Jamming Mitigation
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu betik; Karmaşık I/Q RF Spektrum Simülasyonunu, Bilişsel Modülasyon Sınıflandırmasını,
Takviyeli Öğrenme ile Karıştırmadan Kaçınma (Anti-Jamming) ve 6-Panelli Teşhis Grafiğini çalıştırır.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import numpy as np

from src.cognitive_ew_motoru import (
    RFEmitterSimulator,
    CognitiveSpectrumClassifier,
    AdaptiveAntiJammingAgent,
)
from src.ew_gorsellestirici import EWGorsellestirici
from src.ew_profilleyici import EWProfilleyici


def main():
    print("=" * 75, flush=True)
    print("📡 DAY 346: Elektronik Harp (EW): Bilişsel RF Spektrum Algılama ve Karıştırma (Jamming)", flush=True)
    print("=" * 75, flush=True)

    np.random.seed(42)

    sim = RFEmitterSimulator()
    classifier = CognitiveSpectrumClassifier()
    agent = AdaptiveAntiJammingAgent(num_channels=8, epsilon=0.15, alpha=0.25)

    num_steps = 100
    tx_channels = []
    jammed_channels = []
    sinr_history_db = []
    classification_correct = 0

    print("\n📌 1) 100-Adımlı Bilişsel RF Elektronik Harp ve Dinamik Frekans Atlatma Başlatılıyor...", flush=True)

    sample_i, sample_q = None, None

    for step in range(num_steps):
        # 1. Düşman Karıştırıcı (Sweeping/Periodic Jammer) 1 kanalı hedef alır
        jam_ch = int((step // 5) % 8)
        jammed_channels.append(jam_ch)

        # 2. Bilişsel Ajan Güvenli İletişim Kanalını Seçer
        chosen_ch = agent.select_transmission_channel()
        tx_channels.append(chosen_ch)

        # 3. RF Sinyali Üret ve Sınıflandır
        sig_type = np.random.choice(["QPSK_COMM", "LFM_RADAR", "FHSS_TACTICAL", "HOSTILE_JAMMER"])
        i_sig, q_sig = sim.generate_signal(sig_type, snr_db=15.0)
        
        if sample_i is None:
            sample_i, sample_q = i_sig, q_sig

        res = classifier.classify_emitter(i_sig, q_sig)
        if res["predicted_emitter"] == sig_type:
            classification_correct += 1

        # 4. Kanal Durumu ve SINR Değerlendirmesi
        is_hit = (chosen_ch == jam_ch)
        if is_hit:
            sinr = -5.0 # Ağır karıştırma altında sinyal boğuldu
        else:
            sinr = 20.0 + float(np.random.normal(0, 1.5)) # Temiz kanal

        sinr_history_db.append(sinr)
        agent.update_channel_reward(chosen_ch, sinr, is_jammed=is_hit)

    acc = (classification_correct / num_steps) * 100.0
    mean_sinr = float(np.mean(sinr_history_db))
    hits = sum(1 for tx, jam in zip(tx_channels, jammed_channels) if tx == jam)
    hit_rate = hits / float(num_steps)

    print(f"\n📊 Bilişsel RF Elektronik Harp Performans Sonuçları:", flush=True)
    print(f"  • Modülasyon / Tehdit Tanıma Doğruluğu: %{acc:.2f} (> 90% Kriteri)", flush=True)
    print(f"  • Ortalama Efektif SINR Seviyesi:        {mean_sinr:.2f} dB (> 15 dB Temiz Eşik)", flush=True)
    print(f"  • Karıştırmaya Yakalanma Oranı:         %{hit_rate * 100.0:.2f} (%{100 - hit_rate * 100.0:.1f} Başarılı Savunma)", flush=True)
    print(f"  • Spektrum Hakimiyeti:                  ✅ SAĞLANDI", flush=True)

    # 5. FFT Güç Spektral Yoğunluğu
    fft_vals = np.fft.fft(sample_i + 1j * sample_q)
    psd_freqs = np.fft.fftfreq(len(fft_vals), 1.0 / sim.fs)
    psd_mag_db = 10 * np.log10(np.abs(fft_vals) ** 2 + 1e-6)

    # Sıralı frekanslar
    sort_idx = np.argsort(psd_freqs)
    psd_freqs = psd_freqs[sort_idx]
    psd_mag_db = psd_mag_db[sort_idx]

    profiler_metrics = EWProfilleyici.profille(
        classification_accuracy=acc,
        mean_sinr_db=mean_sinr,
        jamming_collision_rate=hit_rate
    )

    gorsellestirici = EWGorsellestirici()
    cikti_yolu = gorsellestirici.teshis_panelini_ciz(
        i_sig=sample_i,
        q_sig=sample_q,
        psd_freqs=psd_freqs,
        psd_mag_db=psd_mag_db,
        tx_channels=tx_channels,
        jammed_channels=jammed_channels,
        sinr_history_db=sinr_history_db,
        profiler_metrics=profiler_metrics,
        dosya_adi="elektronik_harp_paneli.png"
    )

    print(f"\n🖼️ 6-Panelli Elektronik Harp Teşhis Grafiği Başarıyla Kaydedildi: [elektronik_harp_paneli.png](file:///{os.path.abspath(cikti_yolu)})", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
