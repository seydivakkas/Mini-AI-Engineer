"""
Day 321: Spiking Neural Networks (SNN) & Leaky Integrate-and-Fire (LIF) Neuron Mathematics
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu betik; Spiking Neural Network (SNN) ve Leaky Integrate-and-Fire (LIF) nöron modelinin
sentetik veriler üzerinde eğitimini, profillemesini ve 6-panelli teşhis panosunun üretimini gerçekleştirir.
"""

import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.lif_snn_motoru import SNNClassifier
from src.snn_gorsellestirici import SNNGorsellestirici
from src.snn_profilleyici import SNNProfilleyici


def sentetik_veri_uretcisi(
    num_samples: int = 800,
    in_features: int = 32,
    num_classes: int = 4,
    seed: int = 42
) -> Tuple[TensorDataset, TensorDataset]:
    """
    SNN sınıflandırma eğitimi için sentetik kümelenmiş veri kümesi oluşturur.
    """
    torch.manual_seed(seed)
    # Sınıf merkezleri oluştur
    centers = torch.randn(num_classes, in_features) * 2.0
    
    y = torch.randint(0, num_classes, (num_samples,))
    x = torch.zeros(num_samples, in_features)
    
    for i in range(num_samples):
        class_idx = y[i]
        noise = torch.randn(in_features) * 0.5
        x[i] = torch.sigmoid(centers[class_idx] + noise)  # Normalize [0, 1]
        
    # %80 Train, %20 Test Bölünmesi
    split = int(0.8 * num_samples)
    train_ds = TensorDataset(x[:split], y[:split])
    test_ds = TensorDataset(x[split:], y[split:])
    return train_ds, test_ds


def main():
    print("=" * 75)
    print("🧠 DAY 321: Spiking Neural Networks (SNN) & LIF Nöron Matematiği")
    print("=" * 75)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📌 Çalışma Donanımı: {device}")
    
    # Hyperparameterlar
    in_features = 32
    hidden_features = 64
    num_classes = 4
    time_steps = 25
    batch_size = 64
    epochs = 5
    lr = 0.005
    
    # 1. Veri Hazırlığı
    train_ds, test_ds = sentetik_veri_uretcisi(num_samples=800, in_features=in_features, num_classes=num_classes)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # 2. Model, Kayıp Fonksiyonu ve Optimizer
    model = SNNClassifier(
        in_features=in_features,
        hidden_features=hidden_features,
        num_classes=num_classes,
        time_steps=time_steps,
        beta=0.85,
        v_threshold=1.0
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    test_accuracies = []
    
    print("\n🚀 SNN Eğitimi Başlatılıyor (Surrogate Gradient Backpropagation)...", flush=True)
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x_batch.size(0)
            
        epoch_loss = total_loss / len(train_ds)
        train_losses.append(epoch_loss)
        
        # Test Değerlendirmesi
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits, _ = model(x_batch)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
                
        test_acc = (correct / total) * 100.0
        test_accuracies.append(test_acc)
        
        print(f"  [Epoch {epoch:02d}/{epochs:02d}] Kayıp (Loss): {epoch_loss:.4f} | Test Doğruluğu: %{test_acc:.2f}", flush=True)
        
    elapsed_time = time.time() - start_time
    print(f"\n✅ Eğitim Tamamlandı! Toplam Süre: {elapsed_time:.2f} saniye", flush=True)
    
    # 3. Son Test ve Profilleme Çıkarımı
    model.eval()
    sample_x, sample_y = next(iter(test_loader))
    sample_x = sample_x.to(device)
    with torch.no_grad():
        logits, info_dict = model(sample_x)
        
    profiler_metrics = SNNProfilleyici.profille(model, info_dict, time_steps)
    
    print("\n📊 SNN Donanım & Enerji Profilleme Metrikleri:")
    print(f"  • Toplam Sinaptik Operasyon (SOP): {profiler_metrics['total_sops']:,} SOP")
    print(f"  • Örnek Başına SOP (SNN):          {profiler_metrics['sops_per_sample']:,.1f}")
    print(f"  • Örnek Başına MAC (ANN):          {profiler_metrics['ann_macs_per_sample']:,.1f}")
    print(f"  • Tahmini SNN Enerjisi:           {profiler_metrics['snn_energy_pj']:.1f} pJ")
    print(f"  • Tahmini ANN Enerjisi:           {profiler_metrics['ann_energy_pj']:.1f} pJ")
    print(f"  • Enerji Verimliliği Kazancı:     {profiler_metrics['energy_gain_x']:.2f}x Tasarruf")
    print(f"  • Küresel Spike Seyrekliği:       %{profiler_metrics['global_sparsity']*100:.2f}")
    
    # 4. Teşhis Panosunu Çizme ve Kaydetme
    gorsellestirici = SNNGorsellestirici()
    cikti_yolu = gorsellestirici.teshis_panelini_ciz(
        info_dict=info_dict,
        train_losses=train_losses,
        test_accuracies=test_accuracies,
        profiler_metrics=profiler_metrics
    )
    print(f"\n🖼️ 6-Panelli Teşhis Grafiği Başarıyla Kaydedildi: [snn_lif_teshis_paneli.png](file:///{os.path.abspath(cikti_yolu)})")
    print("=" * 75)


if __name__ == "__main__":
    main()
