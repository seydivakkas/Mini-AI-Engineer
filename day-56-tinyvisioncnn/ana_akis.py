"""
Day 56: Edge Cihazlar İçin Sıfırdan Hafif CNN, Depthwise Separable Conv, FLOPs Hesabı Ana Yürütme Betiği.
"""

import os
import sys

# Dizin yolu ekleme
MEVCUT_DIZIN = os.path.abspath(os.path.dirname(__file__))
if MEVCUT_DIZIN not in sys.path:
    sys.path.insert(0, MEVCUT_DIZIN)

from src.modeller import StandartCNN, TinyVisionCNN
from src.profil_motoru import FLOPsProfilMotoru
from src.gorsellestirici import TinyVisionGorsellestirici


def main():
    print("=" * 85, flush=True)
    print(">>> DAY 56: EDGE CİHAZLAR İÇİN SIFIRDAN HAFİF CNN & ANALİTİK FLOPS HESABI", flush=True)
    print("=" * 85, flush=True)

    # 1. Modellerin Başlatılması
    print("\n[+] 1. Adım: Standart CNN ve TinyVisionCNN Modelleri Başlatılıyor...", flush=True)
    standart_model = StandartCNN(in_channels=3, num_classes=10)
    tiny_model = TinyVisionCNN(in_channels=3, num_classes=10)

    # 2. Kapsamlı Karşılaştırmalı Profilleme
    print("\n[+] 2. Adım: Analitik FLOPs, MACs, Parametre ve Gecikme Ölçümü Yapılıyor...", flush=True)
    girdi_sekli = (1, 3, 64, 64)
    profil_sonuclari = FLOPsProfilMotoru.karsilastirmali_profil(
        standart_model=standart_model,
        tiny_model=tiny_model,
        girdi_sekli=girdi_sekli
    )

    std = profil_sonuclari["standart"]
    tiny = profil_sonuclari["tinyvision"]
    ozet = profil_sonuclari["ozet"]

    print("\n" + "-" * 85, flush=True)
    print(f"{'METRİK / PERFORMANS ÖLÇÜTÜ':<32} | {'STANDART CNN':<22} | {'TINYVISION CNN':<22}", flush=True)
    print("-" * 85, flush=True)
    print(f"{'Toplam Parametre Sayısı':<32} | {std['params']['toplam_param']:>12,}           | {tiny['params']['toplam_param']:>12,}", flush=True)
    print(f"{'Bellek Boyutu (KB / MB)':<32} | {std['params']['boyut_kb']:>8.1f} KB ({std['params']['boyut_mb']:.2f}MB) | {tiny['params']['boyut_kb']:>8.1f} KB ({tiny['params']['boyut_mb']:.2f}MB)", flush=True)
    print(f"{'Toplam MACs':<32} | {std['flops']['toplam_macs']:>12,}           | {tiny['flops']['toplam_macs']:>12,}", flush=True)
    print(f"{'Toplam MFLOPs':<32} | {std['flops']['toplam_mflops']:>10.2f} MFLOPs     | {tiny['flops']['toplam_mflops']:>10.2f} MFLOPs", flush=True)
    print(f"{'Ortalama Çıkarım Gecikmesi':<32} | {std['latency']['ort_gecikme_ms']:>10.2f} ms         | {tiny['latency']['ort_gecikme_ms']:>10.2f} ms", flush=True)
    print(f"{'Kare Hızı (Throughput FPS)':<32} | {std['latency']['fps']:>10.1f} FPS        | {tiny['latency']['fps']:>10.1f} FPS", flush=True)
    print("-" * 85, flush=True)

    print("\n[+] 3. Adım: Hesaplama ve Bellek Verimliliği Tasarrufları:", flush=True)
    print(f"    • Parametre Tasarrufu : %{ozet['param_tasarrufu_yuzde']:.1f} ({ozet['param_tasarruf_carpani']:.1f}x Daha Hafif)", flush=True)
    print(f"    • FLOPs Tasarrufu     : %{ozet['flops_tasarrufu_yuzde']:.1f} ({ozet['flops_tasarruf_carpani']:.1f}x Daha Az Hesaplama)", flush=True)

    # 4. 6 Panelli Teşhis Panosunun Üretilmesi
    print("\n" + "=" * 85, flush=True)
    print(">>> 4. 6 PANELLİ EDGE AI MODEL PERFORMANS PANOSUNUN ÜRETİLMESİ", flush=True)
    print("=" * 85, flush=True)

    hedef_pano = os.path.join(MEVCUT_DIZIN, "ciktilar", "tinyvision_profil_paneli.png")
    cikis_yolu = TinyVisionGorsellestirici.panel_ciz(
        karsilastirma_verisi=profil_sonuclari,
        hedef_path=hedef_pano
    )
    print(f"[+] 6 Panelli Teşhis Panosu Kaydedildi: {os.path.abspath(cikis_yolu)}", flush=True)
    print("=" * 85, flush=True)
    print("DAY 56: TINYVISIONCNN & ANALİTİK FLOPS MOTORU BAŞARIYLA TAMAMLANDI!", flush=True)
    print("=" * 85, flush=True)


if __name__ == "__main__":
    main()
