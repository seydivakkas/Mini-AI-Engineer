"""
Day 86: Görsel Bozulmalar Altında Model Dayanıklılığı & Domain Shift Laboratuvarı
--------------------------------------------------------------------------------
8 farklı görsel bozulma tipi (Noise, Blur, Digital, Weather), 5 şiddet seviyesi,
mCE (Mean Corruption Error) ve Rel-mCE stres testi kıyaslama motoru.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.model import DayanikliVisionModeli
from src.dayaniklilik_olcucu import DayaniklilikOlcucu
from src.gorsellestirici import DayaniklilikGorsellestirici


def tohum_belirle(tohum: int = 42):
    random.seed(tohum)
    np.random.seed(tohum)
    torch.manual_seed(tohum)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tohum)


def veri_olustur(ornek_sayisi: int = 1000, sinif_sayisi: int = 10, gorsel_boyutu: int = 32):
    x = torch.randn(ornek_sayisi, 3, gorsel_boyutu, gorsel_boyutu) * 0.4
    y = torch.randint(0, sinif_sayisi, (ornek_sayisi,))
    for i in range(ornek_sayisi):
        c = y[i].item()
        # Temel Yapısal Sinyal (Core Shape / Spatial feature)
        r_start = (c % 3) * 9 + 2
        c_start = (c // 3) * 9 + 2
        x[i, :, r_start:r_start+8, c_start:c_start+8] += 2.2

        # Yüksek Frekanslı Kırılgan Kestirme (Spurious Texture Shortcut - Bozulmalarda kaybolur)
        grid_i = torch.arange(gorsel_boyutu).unsqueeze(0).repeat(gorsel_boyutu, 1).float()
        freq = (c + 1) * 0.7
        dalga = torch.sin(grid_i * freq) * 0.35
        x[i, 0, :, :] += dalga
    return x, y


def main():
    print("=" * 85)
    print("🚀 Day 86: Görsel Bozulmalar Altında Model Dayanıklılığı & Domain Shift Laboratuvarı")
    print("=" * 85)

    tohum_belirle(42)
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📌 Çalışma Ortamı Cihazı: {cihaz.upper()}")

    # 1. Veri Hazırlığı
    tr_x, tr_y = veri_olustur(800, 10)
    test_x, test_y = veri_olustur(200, 10)

    tr_loader = DataLoader(TensorDataset(tr_x, tr_y), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=32, shuffle=False)

    # 2. Standart Modelin Eğitilmesi
    print(f"\n[1/4] Model 1: Standart Model (Bozulmasız Temiz Veri) Eğitiliyor...")
    tohum_belirle(42)
    model_standart = DayanikliVisionModeli(sinif_sayisi=10, taban_kanal=32).to(cihaz)
    _ = DayanikliVisionModeli.egit(model_standart, tr_loader, epok_sayisi=10, lr=2e-3, dayanikli_egitim=False, cihaz=cihaz)

    # 3. Dayanıklı Modelin (Robust Augmentation) Eğitilmesi
    print(f"\n[2/4] Model 2: Dayanıklı Model (Perturbation / Robust Augmentation) Eğitiliyor...")
    tohum_belirle(42)
    model_dayanikli = DayanikliVisionModeli(sinif_sayisi=10, taban_kanal=32).to(cihaz)
    _ = DayanikliVisionModeli.egit(model_dayanikli, tr_loader, epok_sayisi=10, lr=2e-3, dayanikli_egitim=True, cihaz=cihaz)

    # 4. Kapsamlı Bozulma Stres Testi (8 Bozulma x 5 Şiddet = 40 Koşul)
    print(f"\n[3/4] 8 Bozulma Tipi ve 5 Şiddet Seviyesinde Kapsamlı Stres Testi Koşuluyor...")
    rapor_std = DayaniklilikOlcucu.kapsamli_stres_testi(model_standart, test_loader, cihaz)
    rapor_day = DayaniklilikOlcucu.kapsamli_stres_testi(model_dayanikli, test_loader, cihaz)

    print("\n" + "=" * 65)
    print("📊 MODEL DAYANIKLILIK (ROBUSTNESS) METRİK KARŞILAŞTIRMASI")
    print("=" * 65)
    print(f"  Metrik                       | Standart Model | Dayanıklı Model")
    print("-" * 65)
    print(f"  Temiz Test Doğruluğu (Clean) | %{rapor_std['temiz_dogruluk']:<12.2f} | %{rapor_day['temiz_dogruluk']:<12.2f}")
    print(f"  Bozulma Altı Ort. Doğruluk   | %{rapor_std['macc']:<12.2f} | %{rapor_day['macc']:<12.2f}")
    print(f"  Mean Corruption Error (mCE)  | %{rapor_std['mce']:<12.2f} | %{rapor_day['mce']:<12.2f}")
    print(f"  Relative mCE (Rel-mCE)       | %{rapor_std['rel_mce']:<12.2f} | %{rapor_day['rel_mce']:<12.2f}")
    print("=" * 65)

    fark_macc = rapor_day['macc'] - rapor_std['macc']
    print(f"  🚀 Dayanıklı Modelin Bozulmalar Altındaki Doğruluk Üstünlüğü: +%{fark_macc:.2f}")

    # 5. Teşhis Panosu
    print(f"\n[4/4] 6 Panelli Dayanıklılık Teşhis Panosu Oluşturuluyor...")
    gorsellestirici = DayaniklilikGorsellestirici()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "robustness_domain_shift_paneli.png")

    gorsellestirici.olustur_dayaniklilik_paneli(
        standart_rapor=rapor_std,
        dayanikli_rapor=rapor_day,
        kayit_yolu=cikti_yolu
    )

    print(f"  ✓ 6 Panelli Teşhis Panosu Kaydedildi: {cikti_yolu}")
    print("\n✅ Day 86: Model Robustness & Domain Shift Laboratuvarı Başarıyla Tamamlandı!")


if __name__ == "__main__":
    main()
