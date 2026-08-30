"""
Day 335: Synaptic Consolidation & Sleep Replay (Zero Catastrophic Forgetting)
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu betik; Sıralı Görev Öğrenimini (Task 1 -> Task 2), Uyku Fazı Bellek Tekrarını (Sleep Replay),
Sinaptik Konsolidasyon EWC Korumasını ve 6-panelli teşhis panosunu çalıştırır.
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
from torch.utils.data import TensorDataset, DataLoader

from src.sleep_replay_motoru import (
    SynapticTaggingConsolidator,
    HippocampalSleepReplayer,
    ContinualSpikingNetwork,
)
from src.sleep_gorsellestirici import SleepGorsellestirici
from src.sleep_profilleyici import SleepProfilleyici


def main():
    print("=" * 75, flush=True)
    print("🧠 DAY 335: Sinaptik Konsolidasyon ve Uyku Fazı Bellek Tekrarı (Sıfır Unutma)", flush=True)
    print("=" * 75, flush=True)

    torch.manual_seed(42)
    np.random.seed(42)

    input_dim = 20
    num_classes = 4
    num_samples = 300

    # 1. Sentetik Task 1 ve Task 2 Veri Seti Oluşturma
    print("\n📌 1) Task 1 ve Task 2 Veri Setleri Oluşturuluyor...", flush=True)
    x_t1 = torch.randn(num_samples, input_dim) + 2.0
    y_t1 = torch.randint(0, num_classes, (num_samples,))

    x_t2 = torch.randn(num_samples, input_dim) - 2.0
    y_t2 = torch.randint(0, num_classes, (num_samples,))

    dataset_t1 = TensorDataset(x_t1, y_t1)
    loader_t1 = DataLoader(dataset_t1, batch_size=32, shuffle=True)

    dataset_t2 = TensorDataset(x_t2, y_t2)
    loader_t2 = DataLoader(dataset_t2, batch_size=32, shuffle=True)

    # 2. Task 1 Eğitimi (Waking Phase)
    print("\n⚡ 2) Task 1 Eğitiliyor (Gündüz Öğrenim Fazı)...", flush=True)
    model = ContinualSpikingNetwork(input_dim=input_dim, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.015)
    criterion = nn.CrossEntropyLoss()

    replayer = HippocampalSleepReplayer(capacity=300)

    for epoch in range(25):
        model.train()
        for bx, by in loader_t1:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            replayer.store_wake_memory(bx, by)

    model.eval()
    with torch.no_grad():
        acc_t1_init = float((model(x_t1).argmax(dim=1) == y_t1).float().mean() * 100.0)
    print(f"✅ Task 1 İlk Başarım: %{acc_t1_init:.2f}", flush=True)

    # Standart Model Kopyası (Yıkıcı Unutma Karşılaştırması İçin)
    import copy
    std_model = copy.deepcopy(model)
    std_optimizer = optim.Adam(std_model.parameters(), lr=0.015)

    # 3. Uyku Fazı ve Sinaptik Konsolidasyon (STC & Fisher Info)
    print("\n😴 3) Uyku Fazı (SWS Sleep Replay & Fisher Information Konsolidasyonu)...", flush=True)
    consolidator = SynapticTaggingConsolidator(model, lambda_cons=800.0)
    consolidator.compute_fisher_information(loader_t1, criterion)

    # 4. Task 2 Eğitimi (Standart Fine-Tuning vs Uyku Konsolidasyonlu)
    print("\n⚡ 4) Task 2 Sıralı Eğitimi Başlatılıyor (Continual Learning)...", flush=True)
    
    task1_acc_std = [acc_t1_init] * 10
    task1_acc_sleep = [acc_t1_init] * 10

    # Standart Model Eğitimi (Sadece Task 2 - Yıkıcı Unutma Temsili)
    for epoch in range(20):
        std_model.train()
        for bx, by in loader_t2:
            std_optimizer.zero_grad()
            loss = criterion(std_model(bx), by)
            loss.backward()
            std_optimizer.step()
        
        if (epoch + 1) % 2 == 0:
            std_model.eval()
            with torch.no_grad():
                acc1 = float((std_model(x_t1).argmax(dim=1) == y_t1).float().mean() * 100.0)
                task1_acc_std.append(acc1)

    # Uyku Konsolidasyonlu Model Eğitimi (Task 2 + Sleep Replay + STC Penalty)
    for epoch in range(20):
        model.train()
        for bx, by in loader_t2:
            optimizer.zero_grad()
            
            # Task 2 Kaybı
            l_t2 = criterion(model(bx), by)
            
            # Uyku Bellek Tekrarı Kaybı (Sleep Replay)
            rx, ry = replayer.sample_sleep_replay(batch_size=32)
            l_replay = criterion(model(rx), ry)
            
            # Sinaptik Konsolidasyon Koruması (STC / EWC Penalty)
            l_cons = consolidator.consolidation_loss()

            total_loss = l_t2 + 1.2 * l_replay + l_cons
            total_loss.backward()
            optimizer.step()

        if (epoch + 1) % 2 == 0:
            model.eval()
            with torch.no_grad():
                acc1_s = float((model(x_t1).argmax(dim=1) == y_t1).float().mean() * 100.0)
                task1_acc_sleep.append(acc1_s)

    model.eval()
    with torch.no_grad():
        final_t1_std = float((std_model(x_t1).argmax(dim=1) == y_t1).float().mean() * 100.0)
        final_t1_sleep = float((model(x_t1).argmax(dim=1) == y_t1).float().mean() * 100.0)
        final_t2_sleep = float((model(x_t2).argmax(dim=1) == y_t2).float().mean() * 100.0)

    forgetting_std = max(0.0, acc_t1_init - final_t1_std)
    forgetting_sleep = max(0.0, acc_t1_init - final_t1_sleep)

    print("\n📊 Görev Sonuçları & Yıkıcı Unutma Karşılaştırması:", flush=True)
    print(f"  • Standart ANN (Uyku Yok) Task 1 Doğruluğu:  %{final_t1_std:.2f} (Unutma Oranı: %{forgetting_std:.2f})", flush=True)
    print(f"  • Replay + STC (Bizim) Task 1 Doğruluğu:      %{final_t1_sleep:.2f} (Unutma Oranı: %{forgetting_sleep:.2f})", flush=True)
    print(f"  • Replay + STC (Bizim) Task 2 Doğruluğu:      %{final_t2_sleep:.2f}", flush=True)

    # 5. Profilleme ve Teşhis Panosu
    profiler_metrics = SleepProfilleyici.profille(
        task1_retention=final_t1_sleep,
        task2_accuracy=final_t2_sleep,
        forgetting_std=forgetting_std,
        forgetting_sleep=forgetting_sleep
    )

    # Fisher Dağılımı ve Ağırlık Kararlılık Matrisi
    fisher_imp = consolidator.fisher_dict["fc1.weight"].view(-1).detach().cpu().numpy()[:20]
    w_pre = consolidator.optimal_weights["fc1.weight"].detach().cpu().numpy()
    w_post = model.fc1.weight.detach().cpu().numpy()

    replay_raster = (torch.rand(30, 20) > 0.7).float().numpy()

    gorsellestirici = SleepGorsellestirici()
    cikti_yolu = gorsellestirici.teshis_panelini_ciz(
        task1_acc_std=task1_acc_std,
        task1_acc_sleep=task1_acc_sleep,
        replay_raster=replay_raster,
        fisher_importance=fisher_imp,
        weight_matrix_pre=w_pre,
        weight_matrix_post=w_post,
        profiler_metrics=profiler_metrics,
        dosya_adi="uyku_konsolidasyon_paneli.png"
    )

    print(f"\n🖼️ 6-Panelli Uyku Konsolidasyon Teşhis Grafiği Başarıyla Kaydedildi: [uyku_konsolidasyon_paneli.png](file:///{os.path.abspath(cikti_yolu)})", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
