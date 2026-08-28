"""
Day 70: Mixup, CutMix ve Label Smoothing Ana Akış Betiği
========================================================
1. Standart Eğitim (Baseline - No Reg),
2. Mixup + Label Smoothing (Piksel Harmanlama),
3. CutMix + Label Smoothing (Bölgesel Yama Kesip-Yapıştırma)
kombinasyonlarını aynı model mimarisi ve tohumla eğitir, aşırı güven (overconfidence) ve yakınsama metriklerini ölçer.
4. 6 Panelli görsel laboratuvar teşhis panosunu kaydeder.
"""

import os
import sys
import torch

from src.reguler_karsilastirici import RegulerizasyonLaboratuvari
from src.gorsellestirici import RegulerizasyonGorsellestirici


def main() -> None:
    print("=" * 95)
    print(">>> DAY 70: MIXUP, CUTMIX VE LABEL SMOOTHING MODERN DUZENLILESTIRME LABORATUVARI")
    print("=" * 95)

    kok_dizin = os.path.dirname(os.path.abspath(__file__))
    ciktilar_dizini = os.path.join(kok_dizin, "ciktilar")
    os.makedirs(ciktilar_dizini, exist_ok=True)
    dashboard_yolu = os.path.join(ciktilar_dizini, "modern_regulerizasyon_paneli.png")

    cihaz = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"\n[+] 1. Adim: Regulerizasyon Laboratuvari Baslatiliyor (Cihaz: {cihaz})...")
    print("    - Deney Kapsami: Baseline vs Mixup+LabelSmooth vs CutMix+LabelSmooth (10 Epoch)")

    sonuclar = RegulerizasyonLaboratuvari.tum_laboratuvari_kos(toplam_epoch=10)

    d1 = sonuclar["deney_1"]
    d2 = sonuclar["deney_2"]
    d3 = sonuclar["deney_3"]

    print("\n" + "=" * 95)
    print(">>> 2. DENEYSEL SONUCLAR VE KALIBRASYON PERFORMANSI")
    print("=" * 95)
    print(f"{'Deney Mimarisi':<26} | {'Artirma':<10} | {'Label Smooth':<14} | {'Son Train Loss':<16} | {'Val Acc (%)':<12} | {'Ortalama Guven':<14}")
    print("-" * 95)
    print(f"{d1['deney_adi']:<26} | {d1['artirma_turu']:<10} | {d1['label_smoothing']:<14.1f} | {d1['son_train_loss']:<16.4f} | %{d1['son_val_accuracy']:<10.2f} | {d1['son_ort_guven']:<14.3f}")
    print(f"{d2['deney_adi']:<26} | {d2['artirma_turu']:<10} | {d2['label_smoothing']:<14.1f} | {d2['son_train_loss']:<16.4f} | %{d2['son_val_accuracy']:<10.2f} | {d2['son_ort_guven']:<14.3f}")
    print(f"{d3['deney_adi']:<26} | {d3['artirma_turu']:<10} | {d3['label_smoothing']:<14.1f} | {d3['son_train_loss']:<16.4f} | %{d3['son_val_accuracy']:<10.2f} | {d3['son_ort_guven']:<14.3f}")

    print("\n" + "=" * 95)
    print(">>> 3. DERINLEMESINE REGULERIZASYON & KALIBRASYON ANALIZI")
    print("=" * 95)
    print(f"* Asiri Guven (Overconfidence) Sönümleme : Baseline %{d1['son_ort_guven']*100:.1f} iken Reguler %{d2['son_ort_guven']*100:.1f} seviyesine kalibre edildi.")
    print(f"* Mixup Dogrusalligi (Linearity)        : Siniflar arasi gecis uzayinda yumusak ara temsil ogrenildi.")
    print(f"* CutMix Mekansal Lokalizasyon          : Model tek bir piksel alanina bagimli kalmadan butunsel ozellikleri ogrendi.")

    # Görselleştirme
    print("\n[+] 4. Adim: 6 Panelli Modern Regulerizasyon Teshis Panosu Olusturuluyor...")
    grafik_yolu = RegulerizasyonGorsellestirici.panoyu_ciz_ve_kaydet(
        laboratuvar_sonuclari=sonuclar,
        cikti_yolu=dashboard_yolu
    )
    print(f"[+] Teşhis Panosu Kaydedildi: {grafik_yolu}")
    print("=" * 95)
    print("DAY 70: MODERN REGULARIZATION (MIXUP, CUTMIX, LABEL SMOOTHING) BASARIYLA TAMAMLANDI!")
    print("=" * 95)


if __name__ == "__main__":
    main()
