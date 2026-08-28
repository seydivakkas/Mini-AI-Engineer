"""
Day 88: Optuna ile Otomatik Hiperparametre Optimizasyonu (TPE & Pruning) Laboratuvarı
-------------------------------------------------------------------------------------
Bayesyen TPE (Tree-structured Parzen Estimator) algoritması ve MedianPruner erken durdurma
ile otomatik arama uzayı taraması, hiperparametre önem analizi ve görselleştirme paneli.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
import sys
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.optuna_optimize import OptunaHPOVurucu
from src.gorsellestirici import OptunaGorsellestirici


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
    print("🚀 Day 88: Optuna ile Otomatik Hiperparametre Optimizasyonu (TPE & Pruning) Laboratuvarı")
    print("=" * 85)

    tohum_belirle(42)
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📌 Çalışma Ortamı Cihazı: {cihaz.upper()}")

    # 1. Veri Hazırlığı
    tr_x, tr_y = veri_olustur(600, 10)
    val_x, val_y = veri_olustur(200, 10)

    tr_loader = DataLoader(TensorDataset(tr_x, tr_y), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=32, shuffle=False)

    # 2. Optuna HPO Yöneticisinin Başlatılması
    print(f"\n[1/3] Optuna TPE Arama Uzayı ve MedianPruner Başlatılıyor...")
    hpo_motoru = OptunaHPOVurucu(
        calisma_adi="MiniViT_TPE_Study",
        hedef_yon="minimize",
        tohum=42,
        startup_deneme_sayisi=4,
        warmup_epok_sayisi=2
    )

    # 3. Optimizasyon Süreci
    deneme_adedi = 16
    print(f"\n[2/3] {deneme_adedi} Denemelik TPE Bayesyen Optimizasyonu Koşuluyor...")
    study = hpo_motoru.optimize_et(
        train_loader=tr_loader,
        val_loader=val_loader,
        deneme_sayisi=deneme_adedi,
        epok_sayisi=8,
        cihaz=cihaz
    )

    ozet = hpo_motoru.calisma_ozeti()

    print("\n" + "=" * 70)
    print("🏆 OPTUNA HPO ÇALIŞMA SONUÇLARI VE ŞAMPİYON HİPERPARAMETRELER")
    print("=" * 70)
    print(f"  Toplam Deneme Sayısı   : {ozet['toplam_deneme']}")
    print(f"  Tamamlanan Denemeler   : {ozet['tamamlanan_sayisi']}")
    print(f"  Erken Budanan Denemeler: {ozet['budanan_sayisi']} (Hesaplama Tasarrufu: %{(ozet['budanan_sayisi']/ozet['toplam_deneme'])*100:.1f})")
    print(f"  En İyi Validation Loss : {ozet['en_iyi_deger']:.4f}")
    print("-" * 70)
    print("  🥇 ŞAMPİYON HİPERPARAMETRELER:")
    for k, v in ozet["en_iyi_parametreler"].items():
        if isinstance(v, float):
            print(f"     • {k:<15}: {v:.6f}")
        else:
            print(f"     • {k:<15}: {v}")
    print("=" * 70)

    # 4. Teşhis Panosunun Oluşturulması
    print(f"\n[3/3] 6 Panelli Optuna HPO Teşhis Panosu Oluşturuluyor...")
    gorsellestirici = OptunaGorsellestirici()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "optuna_hpo_paneli.png")

    gorsellestirici.olustur_hpo_paneli(
        study=study,
        ozet=ozet,
        kayit_yolu=cikti_yolu
    )

    print(f"  ✓ 6 Panelli Teşhis Panosu Kaydedildi: {cikti_yolu}")
    print("\n✅ Day 88: Optuna HPO & TPE Laboratuvarı Başarıyla Tamamlandı!")


if __name__ == "__main__":
    main()
