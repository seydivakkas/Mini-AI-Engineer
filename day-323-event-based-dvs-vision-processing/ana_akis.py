"""
Day 323: Dynamic Vision Sensors (DVS) & Event-Based Processing
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu betik; DVS nöromorfik kamera olay akışlarını simüle eder, SAE sönümlenme yüzeyini oluşturur,
Voxel Grid kodlaması ile Spiking ConvNet modelini eğitir ve 6-panelli teşhis panosunu oluşturur.
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
from torch.utils.data import DataLoader, TensorDataset

from src.dvs_motoru import (
    DVSEventStreamGenerator,
    SurfaceOfActiveEvents,
    VoxelGridEncoder,
    SpikingEventConvNet,
)
from src.dvs_gorsellestirici import DVSGorsellestirici
from src.dvs_profilleyici import DVSProfilleyici


def dvs_sentetik_veri_uretcisi(
    num_samples: int = 240,
    height: int = 32,
    width: int = 32,
    num_bins: int = 5,
    seed: int = 42
) -> Tuple[TensorDataset, TensorDataset, np.ndarray]:
    """
    4 farklı yönlü hareket ('sag', 'sol', 'yukari', 'asagi') içeren DVS Voxel Grid veri kümesi üretir.
    """
    np.random.seed(seed)
    gen = DVSEventStreamGenerator(height=height, width=width)
    encoder = VoxelGridEncoder(height=height, width=width, num_bins=num_bins)

    directions = ["sag", "sol", "yukari", "asagi"]
    voxels_list = []
    labels_list = []
    sample_events = None

    for i in range(num_samples):
        label_idx = i % 4
        dir_name = directions[label_idx]
        events = gen.uret_hareketli_cizgi(yon=dir_name, duration_us=50000, num_events=800, seed=seed + i)
        
        if sample_events is None:
            sample_events = events

        voxel = encoder.kodla(events, duration_us=50000.0)
        voxels_list.append(voxel)
        labels_list.append(label_idx)

    x_tensor = torch.stack(voxels_list, dim=0)  # (N, C=2*num_bins, H, W)
    y_tensor = torch.tensor(labels_list, dtype=torch.long)

    split = int(0.8 * num_samples)
    train_ds = TensorDataset(x_tensor[:split], y_tensor[:split])
    test_ds = TensorDataset(x_tensor[split:], y_tensor[split:])
    return train_ds, test_ds, sample_events


def main():
    print("=" * 75, flush=True)
    print("🧠 DAY 323: Dynamic Vision Sensors (DVS) & Olay Tabanlı Görsel İşleme", flush=True)
    print("=" * 75, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📌 Çalışma Donanımı: {device}", flush=True)

    height, width = 32, 32
    num_bins = 5
    in_channels = 2 * num_bins
    num_classes = 4
    epochs = 5
    batch_size = 32
    lr = 0.003

    # 1. Veri Hazırlığı
    train_ds, test_ds, sample_events = dvs_sentetik_veri_uretcisi(
        num_samples=240, height=height, width=width, num_bins=num_bins
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # SAE ve Voxel Kodlayıcı Örnekleme
    sae_builder = SurfaceOfActiveEvents(height=height, width=width, tau_us=10000.0)
    sample_sae = sae_builder.guncelle_ve_hesapla(sample_events, t_current=50000.0)
    sample_voxel = VoxelGridEncoder(height=height, width=width, num_bins=num_bins).kodla(sample_events)

    # 2. Model & Optimizer
    model = SpikingEventConvNet(
        in_channels=in_channels, num_classes=num_classes, beta=0.85, v_th=1.0
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_accs = []

    print("\n🚀 Spiking Event ConvNet Eğitimi Başlatılıyor...", flush=True)
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits, _ = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_b.size(0)

        epoch_loss = total_loss / len(train_ds)
        train_losses.append(epoch_loss)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x_b, y_b in test_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                logits, _ = model(x_b)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y_b).sum().item()
                total += y_b.size(0)

        test_acc = (correct / total) * 100.0
        test_accs.append(test_acc)
        print(f"  [Epoch {epoch:02d}/{epochs:02d}] Kayıp (Loss): {epoch_loss:.4f} | Test Doğruluğu: %{test_acc:.2f}", flush=True)

    elapsed_time = time.time() - start_time
    print(f"\n✅ Eğitim Tamamlandı! Toplam Süre: {elapsed_time:.2f} saniye", flush=True)

    # 3. Profilleme Metrikleri
    profiler_metrics = DVSProfilleyici.profille(
        events=sample_events, height=height, width=width, duration_us=50000.0
    )

    print("\n📊 DVS Nöromorfik Profilleme Metrikleri:", flush=True)
    print(f"  • Toplam Olay Sayısı:               {profiler_metrics['num_events']:,} Event", flush=True)
    print(f"  • DVS Veri Hacmi:                   {profiler_metrics['dvs_bytes']/1024.0:.2f} KB", flush=True)
    print(f"  • Standart Video Veri Hacmi:        {profiler_metrics['frame_bytes']/1024.0:.2f} KB", flush=True)
    print(f"  • Veri Sıkıştırma Kazancı:          {profiler_metrics['compression_ratio_x']:.2f}x Veri Tasarrufu", flush=True)
    print(f"  • Olay Akış Hızı (Throughput):       {profiler_metrics['throughput_events_per_sec']:,.0f} Events/sec", flush=True)

    # 4. Teşhis Panosunu Çizme ve Kaydetme
    gorsellestirici = DVSGorsellestirici()
    cikti_yolu = gorsellestirici.teshis_panelini_ciz(
        events=sample_events,
        sae_surface=sample_sae,
        voxel_grid=sample_voxel,
        train_losses=train_losses,
        test_accs=test_accs,
        profiler_metrics=profiler_metrics
    )
    print(f"\n🖼️ 6-Panelli DVS Teşhis Grafiği Başarıyla Kaydedildi: [dvs_isleme_paneli.png](file:///{os.path.abspath(cikti_yolu)})", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
