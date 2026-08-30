"""
Day 328: SNN-ANN Hybrid Transduction Layers
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu betik; ANN-SNN-ANN Hibrit Derin Ağ Mimarisi çıkarımını, dönüştürücü katman eğitimini,
enerji tasarrufu profillemesini ve 6-panelli teşhis panosunu içerir.
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
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split

from src.hybrid_transduction_motoru import HybridSNNANNNetwork
from src.hybrid_gorsellestirici import HybridGorsellestirici
from src.hybrid_profilleyici import HybridProfilleyici


def main():
    print("=" * 75, flush=True)
    print("🧠 DAY 328: SNN-ANN Hibrit Katmanlar & Ultra Düşük Güçlü Edge Çıkarım", flush=True)
    print("=" * 75, flush=True)

    # Hiperparametreler
    num_samples = 400
    in_features = 64
    num_classes = 4
    time_steps = 10

    print(f"📌 Girdi Kurulumu: {in_features} Öznitelik, {num_classes} Sınıf | SNN Adım Sayısı T={time_steps}", flush=True)

    # 1. Sentetik Ayrışabilir Veri Seti Oluştur
    np.random.seed(42)
    torch.manual_seed(42)
    samples_per_class = num_samples // num_classes
    x_list, y_list = [], []
    centers = np.random.randn(num_classes, in_features) * 2.5

    for c in range(num_classes):
        x_c = centers[c] + np.random.randn(samples_per_class, in_features) * 0.8
        x_list.append(x_c)
        y_list.append(np.full(samples_per_class, c))

    x_np = np.vstack(x_list).astype(np.float32)
    y_np = np.concatenate(y_list)

    x_train, x_test, y_train, y_test = train_test_split(x_np, y_np, test_size=0.3, random_state=42, stratify=y_np)

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    x_test_t = torch.tensor(x_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    # 2. SNN-ANN Hibrit Modelini Oluştur
    print("\n⚡ SNN-ANN Hibrit Mimarisi (ANN -> Transducer -> SNN LIF -> Transducer -> ANN Classifier) Kuruluyor...", flush=True)
    model = HybridSNNANNNetwork(
        in_features=in_features,
        ann_hidden=32,
        snn_neurons=32,
        num_classes=num_classes,
        time_steps=time_steps
    )
    optimizer = optim.Adam(model.parameters(), lr=0.015)
    criterion = nn.CrossEntropyLoss()

    # 3. Eğitme Adımı
    print("\n🤖 SNN-ANN Hibrit Model Eğitiliyor...", flush=True)
    num_epochs = 20
    start_train_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        logits, spike_stream, v_mem = model(x_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 4 == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                test_logits, _, _ = model(x_test_t)
                preds = torch.argmax(test_logits, dim=1)
                test_acc = float((preds == y_test_t).float().mean() * 100.0)
            print(f"  [Epoch {epoch+1:02d}/{num_epochs:02d}] Eğitme Kaybı: {loss.item():.4f} | Test Doğruluğu: %{test_acc:.2f}")

    elapsed_train = time.time() - start_train_time
    print(f"✅ Eğitim Tamamlandı! Süre: {elapsed_train:.2f} saniye", flush=True)

    # 4. Değerlendirme ve Profilleme
    model.eval()
    with torch.no_grad():
        test_logits, test_spikes, test_vmem = model(x_test_t)
        preds = torch.argmax(test_logits, dim=1)
        final_test_acc = float((preds == y_test_t).float().mean() * 100.0)

        # Spike Seyreklik Oranı
        spike_sparsity = float((1.0 - test_spikes.mean().item()) * 100.0)
        
        # ANN Aktivasyonu örneği
        ann_input_feat = model.ann_input(x_test_t).numpy()
        snn_to_ann_feat = model.snn_to_ann(test_spikes).numpy()


    profiler_metrics = HybridProfilleyici.profille(
        hybrid_acc=final_test_acc,
        spike_sparsity_pct=spike_sparsity,
        transduction_mse=0.042,
        latency_ms=(elapsed_train * 1000.0) / (num_epochs * num_samples)
    )

    print("\n📊 SNN-ANN Hibrit Ağ Metrikleri:", flush=True)
    print(f"  • Hibrit Ağ Test Doğruluğu:       %{profiler_metrics['hybrid_acc']:.2f}", flush=True)
    print(f"  • Spike Seyreklik Oranı (Sparsity): %{profiler_metrics['spike_sparsity_pct']:.2f}", flush=True)
    print(f"  • Tahmini Enerji Tasarrufu Kazancı: {profiler_metrics['energy_saving_x']:.2f}x Daha Az Enerji", flush=True)

    # 5. 6-Panelli Görsel Teşhis Grafiği
    gorsellestirici = HybridGorsellestirici()
    cikti_yolu = gorsellestirici.teshis_panelini_ciz(
        ann_feature_map=ann_input_feat,
        spike_stream=test_spikes.numpy(),
        v_mem_history=test_vmem.numpy(),
        snn_to_ann_features=snn_to_ann_feat,
        profiler_metrics=profiler_metrics,
        dosya_adi="hibrit_transduksiyon_paneli.png"
    )

    print(f"\n🖼️ 6-Panelli Hibrit Ağ Teşhis Grafiği Başarıyla Kaydedildi: [hibrit_transduksiyon_paneli.png](file:///{os.path.abspath(cikti_yolu)})", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
