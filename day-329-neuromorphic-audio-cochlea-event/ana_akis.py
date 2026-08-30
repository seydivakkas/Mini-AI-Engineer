"""
Day 329: Neuromorphic Auditory Cochlea Filters & Event-Based Acoustic Classification
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu betik; Gammatone Koklear Süzgeç Bankasını, Olay Tabanlı Silikon Koklea Spike Üretimini,
Spiking Sinir Ağı Komut Sınıflandırmasını ve 6-panelli teşhis panosunu içerir.
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
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from src.cochlea_audio_motoru import (
    GammatoneFilterBank,
    SiliconCochleaEventGenerator,
    SpikingAudioClassifier,
)
from src.cochlea_gorsellestirici import CochleaGorsellestirici
from src.cochlea_profilleyici import CochleaProfilleyici


def uret_akustik_komut_sinyali(command_id: int, fs: int = 16000, duration_sec: float = 1.0) -> np.ndarray:
    """
    4 Akustik Komut ("Evet"=0, "Hayır"=1, "Dur"=2, "Geç"=3) için sentetik ses dalga formu üretir.
    """
    total_samples = int(fs * duration_sec)
    t = np.linspace(0, duration_sec, total_samples)
    audio = np.random.randn(total_samples) * 0.05

    # Komuta özel frekans bileşenleri
    if command_id == 0:  # "Evet" -> Düşükten yükseğe sweep (300Hz -> 1500Hz)
        freq = 300.0 + 1200.0 * t
        audio += 0.8 * np.sin(2 * np.pi * freq * t)
    elif command_id == 1:  # "Hayır" -> Yüksekten düşüğe sweep (2500Hz -> 500Hz)
        freq = 2500.0 - 2000.0 * t
        audio += 0.8 * np.sin(2 * np.pi * freq * t)
    elif command_id == 2:  # "Dur" -> Sabit yüksek frekans (3500Hz burst)
        audio += 0.9 * np.sin(2 * np.pi * 3500.0 * t) * (t < 0.4)
    else:  # "Geç" -> Çift frekans harmonik (800Hz + 2200Hz)
        audio += 0.5 * np.sin(2 * np.pi * 800.0 * t) + 0.5 * np.sin(2 * np.pi * 2200.0 * t)

    return audio.astype(np.float32)


def main():
    print("=" * 75, flush=True)
    print("🧠 DAY 329: Nöromorfik Koklea Filtreleri ve Olay Tabanlı Akustik Sınıflandırma", flush=True)
    print("=" * 75, flush=True)

    fs = 16000
    num_channels = 16
    num_samples_per_class = 25
    num_classes = 4

    print(f"📌 Koklea Kurulumu: {num_channels} Gammatone ERB Kanalı (100Hz - 6000Hz), {fs} Hz Örnekleme", flush=True)
    print(f"📌 Akustik Komutlar: 'Evet' (0), 'Hayır' (1), 'Dur' (2), 'Geç' (3)", flush=True)

    # 1. Ses Sinyalleri ve Kokleogram Çıkarımı
    print("\n⚡ 1) Gammatone Koklear Süzgeç ve Silikon Koklea Spike Üretimi Başlatılıyor...", flush=True)
    start_time = time.time()
    
    filter_bank = GammatoneFilterBank(num_channels=num_channels, f_min=100.0, f_max=6000.0, fs=fs)
    event_gen = SiliconCochleaEventGenerator(threshold=0.08, time_bin_size=160)

    cochleograms = []
    labels = []
    total_events_count = 0

    for c_id in range(num_classes):
        for _ in range(num_samples_per_class):
            audio = uret_akustik_komut_sinyali(c_id, fs=fs, duration_sec=1.0)
            filtered = filter_bank.filtrele(audio)
            coch, events = event_gen.uret_kokleogram_spikelari(filtered)
            
            cochleograms.append(coch)
            labels.append(c_id)
            total_events_count += len(events)

    cochleograms_np = np.array(cochleograms, dtype=np.float32)  # (Total, Channels, Time_Bins)
    labels_np = np.array(labels, dtype=np.int64)

    elapsed_extract = time.time() - start_time
    print(f"✅ Çıkarım Tamamlandı! Toplam Örnek: {len(labels)} | Toplam Olay: {total_events_count} Event", flush=True)
    print(f"  • Kokleogram Boyutu: {cochleograms_np.shape}", flush=True)

    # Train / Test Bölünmesi
    x_train, x_test, y_train, y_test = train_test_split(cochleograms_np, labels_np, test_size=0.3, random_state=42, stratify=labels_np)

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    x_test_t = torch.tensor(x_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    # 2. SNN Akustik Sınıflandırıcı Eğitimi
    print("\n🤖 2) SNN Akustik Komut Sınıflandırıcısı Eğitiliyor...", flush=True)
    num_time_bins = cochleograms_np.shape[2]
    model = SpikingAudioClassifier(num_channels=num_channels, num_time_bins=num_time_bins, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.008)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 10
    start_train = time.time()
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 2 == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                t_logits = model(x_test_t)
                preds = torch.argmax(t_logits, dim=1)
                t_acc = float((preds == y_test_t).float().mean() * 100.0)
            print(f"  [Epoch {epoch+1:02d}/{num_epochs:02d}] Kayıp: {loss.item():.4f} | Test Doğruluğu: %{t_acc:.2f}")

    elapsed_train = time.time() - start_train
    print(f"✅ SNN Eğitimi Tamamlandı! Süre: {elapsed_train:.2f} saniye", flush=True)

    # 3. Değerlendirme ve Profilleme
    model.eval()
    with torch.no_grad():
        test_logits = model(x_test_t)
        probs = torch.softmax(test_logits, dim=1).numpy()[0]
        final_acc = float((torch.argmax(test_logits, dim=1) == y_test_t).float().mean() * 100.0)

    pcm_bytes = fs * 2  # 1 second 16-bit PCM = 32000 bytes
    profiler_metrics = CochleaProfilleyici.profille(
        total_events=int(total_events_count / len(labels)),
        pcm_bytes=pcm_bytes,
        snn_accuracy=final_acc,
        latency_ms=(elapsed_train * 1000.0) / num_epochs
    )

    print("\n📊 Nöromorfik Koklea Profilleme Metrikleri:", flush=True)
    print(f"  • Örnek Başına Olay Sayısı:       {profiler_metrics['total_events']} Event", flush=True)
    print(f"  • Veri Sıkıştırma Kazancı:       {profiler_metrics['compression_ratio_x']:.2f}x Tasarruf", flush=True)
    print(f"  • SNN Komut Tanıma Doğruluğu:   %{profiler_metrics['snn_accuracy']:.2f}", flush=True)

    # 4. 6-Panelli Görsel Teşhis Grafiği
    sample_audio = uret_akustik_komut_sinyali(0, fs=fs, duration_sec=1.0)
    sample_filtered = filter_bank.filtrele(sample_audio)
    sample_coch, _ = event_gen.uret_kokleogram_spikelari(sample_filtered)

    gorsellestirici = CochleaGorsellestirici()
    cikti_yolu = gorsellestirici.teshis_panelini_ciz(
        raw_audio=sample_audio,
        filtered_audio=sample_filtered,
        center_freqs=filter_bank.center_freqs,
        cochleogram=sample_coch,
        class_probs=probs,
        profiler_metrics=profiler_metrics,
        dosya_adi="koklea_isitsel_paneli.png"
    )

    print(f"\n🖼️ 6-Panelli Koklea Teşhis Grafiği Başarıyla Kaydedildi: [koklea_isitsel_paneli.png](file:///{os.path.abspath(cikti_yolu)})", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
