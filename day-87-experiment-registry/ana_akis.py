"""
Day 87: MLflow / Weights & Biases ile Merkezi Deney Takibi ve Artefakt Kayıt Laboratuvarı
-----------------------------------------------------------------------------------------
5 farklı mimari ve hiperparametre koşusunun (Runs) merkezi SQLite & Artefakt deposunda
otomatik olarak loglanması, metrik eğrilerinin analizi ve Liderlik Tablosunun çıkarılması.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.takip_motoru import MerkeziDeneyTakipMotoru
from src.model import DeneyVisionModeli
from src.karsilastirici import DeneyKarsilastirici
from src.gorsellestirici import MLOpsGorsellestirici


def tohum_belirle(tohum: int = 42):
    random.seed(tohum)
    np.random.seed(tohum)
    torch.manual_seed(tohum)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tohum)


def veri_olustur(ornek_sayisi: int = 600, sinif_sayisi: int = 10, gorsel_boyutu: int = 32):
    x = torch.randn(ornek_sayisi, 3, gorsel_boyutu, gorsel_boyutu) * 0.5
    y = torch.randint(0, sinif_sayisi, (ornek_sayisi,))
    for i in range(ornek_sayisi):
        c = y[i].item()
        r = (c % 3) * 9 + 2
        col = (c // 3) * 9 + 2
        x[i, :, r:r+8, col:col+8] += 2.0
    return x, y


def main():
    print("=" * 85)
    print("🚀 Day 87: MLflow / Weights & Biases ile Merkezi Deney Takibi ve Artefakt Kayıt Laboratuvarı")
    print("=" * 85)

    tohum_belirle(42)
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📌 Çalışma Ortamı Cihazı: {cihaz.upper()}")

    # 1. Veri Hazırlığı
    tr_x, tr_y = veri_olustur(600, 10)
    val_x, val_y = veri_olustur(200, 10)

    tr_loader = DataLoader(TensorDataset(tr_x, tr_y), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=32, shuffle=False)

    # 2. Merkezi Deney Takip Motorunun Başlatılması
    depo_yolu = os.path.join(os.path.dirname(__file__), ".deney_deposu")
    motor = MerkeziDeneyTakipMotoru(depo_dizini=depo_yolu)
    deney_adi = "VisionModel_Hiperparametre_Optimizasyonu"

    # 3. 5 Farklı Hiperparametre Konfigürasyonunun Tanımlanması
    deney_planlari = [
        {"run_name": "AdamW_lr_1e3_std", "lr": 1e-3, "opt": "adamw", "wd": 1e-4, "kanal": 32, "drop": 0.1, "epok": 8},
        {"run_name": "AdamW_lr_3e4_decay", "lr": 3e-4, "opt": "adamw", "wd": 1e-3, "kanal": 32, "drop": 0.1, "epok": 8},
        {"run_name": "SGD_Momentum_lr_1e2", "lr": 1e-2, "opt": "sgd", "wd": 1e-4, "kanal": 32, "drop": 0.1, "epok": 8},
        {"run_name": "Adam_WideNet_lr_1e3", "lr": 1e-3, "opt": "adam", "wd": 1e-4, "kanal": 48, "drop": 0.2, "epok": 8},
        {"run_name": "AdamW_lr_5e3_HighLR", "lr": 5e-3, "opt": "adamw", "wd": 1e-4, "kanal": 32, "drop": 0.1, "epok": 8},
    ]

    print(f"\n[1/3] {len(deney_planlari)} Farklı Model Koşusu Başlatılıyor ve Merkezi Depoya Loglanıyor...")

    for idx, cfg in enumerate(deney_planlari, 1):
        print(f"\n  ▶ Koşu {idx}/{len(deney_planlari)}: {cfg['run_name']} (Opt: {cfg['opt']}, LR: {cfg['lr']})...")
        tohum_belirle(42 + idx)
        kosu = motor.start_run(deney_adi=deney_adi, kosu_adi=cfg["run_name"])

        # Modeli oluştur
        model = DeneyVisionModeli(
            sinif_sayisi=10,
            taban_kanal=cfg["kanal"],
            dropout_orani=cfg["drop"]
        )

        # Eğit ve otomatik logla
        sonuc = DeneyVisionModeli.egit_ve_kaydet(
            model=model,
            train_loader=tr_loader,
            val_loader=val_loader,
            kosu=kosu,
            epok_sayisi=cfg["epok"],
            lr=cfg["lr"],
            optimizator_tipi=cfg["opt"],
            weight_decay=cfg["wd"],
            cihaz=cihaz
        )
        motor.end_run(durum="FINISHED")
        print(f"    ✓ Koşu Tamamlandı! En İyi Val Doğruluğu: %{sonuc['best_val_acc']:.2f}")

    # 4. Deney Koşularının Karşılaştırılması ve Lider Tablosu
    print(f"\n[2/3] Merkezi Veritabanından Tüm Koşular Çekiliyor ve Liderlik Tablosu Oluşturuluyor...")
    kosular = motor.tum_kosulari_getir(deney_adi=deney_adi)
    df_liderlik = DeneyKarsilastirici.karsilastirma_tablosu(kosular)

    print("\n" + "=" * 80)
    print("🏆 MERKEZİ DENEY LİDERLİK TABLOSU (LEADERBOARD)")
    print("=" * 80)
    gosterilecek_kolonlar = ["run_name", "p_optimizer", "p_learning_rate", "p_param_count", "m_val_acc", "m_val_loss", "sure_sn"]
    mevcut_kolonlar = [c for c in gosterilecek_kolonlar if c in df_liderlik.columns]
    print(df_liderlik[mevcut_kolonlar].to_string(index=False))
    print("=" * 80)

    en_iyi_model = df_liderlik.iloc[0]
    print(f"  🥇 Şampiyon Model: {en_iyi_model['run_name']} (Val Acc: %{en_iyi_model['m_val_acc']:.2f})")

    # 5. 6 Panelli MLOps Teşhis Panosunun Üretilmesi
    print(f"\n[3/3] 6 Panelli MLOps Deney Takip Teşhis Panosu Oluşturuluyor...")
    gorsellestirici = MLOpsGorsellestirici()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "experiment_registry_paneli.png")

    gorsellestirici.olustur_deney_paneli(
        kosular=kosular,
        df_liderlik=df_liderlik,
        kayit_yolu=cikti_yolu
    )

    print(f"  ✓ 6 Panelli Teşhis Panosu Kaydedildi: {cikti_yolu}")
    print("\n✅ Day 87: MLflow / W&B Merkezi Deney Takibi Laboratuvarı Başarıyla Tamamlandı!")


if __name__ == "__main__":
    main()
