"""
Day 68: Albumentations ile Yuksek Performansli Veri Artirma & GPU Prefetching Ana Akis
=====================================================================================
1. Albumentations donusumlerini ornek gorsel uzerinde uygular ve gorselleri hazirlar.
2. Torchvision vs Albumentations CPU vs Albumentations + CUDA Prefetcher boru hatlarini benchmark eder.
3. FPS, Batch Gecikmesi ve Hizlanma carpanlarini konsola yazdirir.
4. 6 Panelli endustriyel karsilastirma panosunu kaydeder.
"""

import os
import sys
import numpy as np
import cv2
import torch
import albumentations as A

from src.veri_donusturucu import YuksekPerformansArtirici
from src.boru_hatti_karsilastirici import BoruHattiKarsilastirici
from src.gorsellestirici import VeriBoruHattiGorsellestirici


def ornek_gorselleri_hazirla() -> tuple[list[np.ndarray], list[str]]:
    """Görselleştirme paneli için sentetik renkli desenler ve artırma varyasyonları üretir."""
    # Renkli sentetik desen
    H, W = 128, 128
    img = np.zeros((H, W, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (108, 108), (220, 50, 50), -1)
    cv2.circle(img, (64, 64), 30, (50, 220, 50), -1)
    cv2.line(img, (10, 118), (118, 10), (50, 50, 220), 4)

    # Albumentations dönüşüm varyasyonları
    t_crop = A.RandomResizedCrop(size=(H, W), scale=(0.7, 0.9), p=1.0)(image=img)["image"]
    t_flip_rot = A.Compose([A.HorizontalFlip(p=1.0), A.Affine(rotate=30, p=1.0)])(image=img)["image"]
    t_jitter_noise = A.Compose([
        A.ColorJitter(brightness=0.4, contrast=0.4, p=1.0),
        A.GaussNoise(p=1.0)
    ])(image=img)["image"]

    gorseller = [img, t_crop, t_flip_rot, t_jitter_noise]
    basliklar = ["Orijinal Desen", "RandomResizedCrop", "Flip + Rotate 30°", "ColorJitter + GaussNoise"]
    return gorseller, basliklar


def main() -> None:
    print("=" * 95)
    print(">>> DAY 68: ALBUMENTATIONS ILE YUKSEK PERFORMANSLI VERI ARTIRMA & GPU PREFETCHING")
    print("=" * 95)

    kok_dizin = os.path.dirname(os.path.abspath(__file__))
    ciktilar_dizini = os.path.join(kok_dizin, "ciktilar")
    os.makedirs(ciktilar_dizini, exist_ok=True)
    dashboard_yolu = os.path.join(ciktilar_dizini, "veri_boru_hatti_paneli.png")

    # 1. Adım: Veri Artırma Örnekleri
    print("\n[+] 1. Adim: Albumentations C++ Goruntu Donusumleri Ilklendiriliyor...")
    ornek_gorseller, ornek_basliklar = ornek_gorselleri_hazirla()
    print(f"    - Uretilen Donusum Varyasyonu : {len(ornek_gorseller)} adet")

    # 2. Adım: Boru Hatları Benchmark Testi
    print("\n[+] 2. Adim: 3 Farkli Veri Boru Hatti Icin Throughput ve Gecikme Benchmark'i Baslatiliyor...")
    print("    - Ornek Sayisi: 1200 | Batch Size: 64 | Cozunurluk: 64x64")

    sonuclar = BoruHattiKarsilastirici.benchmark_kos(
        ornek_sayisi=1200,
        batch_size=64,
        gorsel_boyutu=(64, 64),
        tekrar_sayisi=3
    )

    tv = sonuclar["torchvision"]
    albu = sonuclar["albumentations_cpu"]
    pref = sonuclar["albumentations_prefetcher"]

    print("\n" + "=" * 95)
    print(">>> 3. VERI BORU HATTI PERFORMANS VE HIZLANMA KARSILASTIRMASI")
    print("=" * 95)
    print(f"{'Boru Hatti Mimarisi':<32} | {'Ort. Sure (sn)':<15} | {'Throughput (FPS)':<18} | {'Batch Gecikmesi':<16} | {'Hizlanma':<10}")
    print("-" * 95)
    print(f"{'1. Torchvision (PIL Baseline)':<32} | {tv['toplam_sure_sn']:<15.4f} | {tv['fps']:<18.1f} | {tv['batch_gecikmesi_ms']:<13.2f} ms | {tv['hizlanma_kat']:<8.2f}x")
    print(f"{'2. Albumentations (CPU)':<32} | {albu['toplam_sure_sn']:<15.4f} | {albu['fps']:<18.1f} | {albu['batch_gecikmesi_ms']:<13.2f} ms | {albu['hizlanma_kat']:<8.2f}x")
    print(f"{'3. Albu + CUDA Stream Prefetch':<32} | {pref['toplam_sure_sn']:<15.4f} | {pref['fps']:<18.1f} | {pref['batch_gecikmesi_ms']:<13.2f} ms | {pref['hizlanma_kat']:<8.2f}x")

    print("\n" + "=" * 95)
    print(">>> 4. KAZANIM VE DARBOGAZ ANALIZI")
    print("=" * 95)
    print(f"* Albumentations CPU Hizlanmasi           : {albu['hizlanma_kat']:.2f}x Kat ({albu['fps']:.1f} FPS vs {tv['fps']:.1f} FPS)")
    print(f"* Albumentations + CUDA Prefetch Hizlanma : {pref['hizlanma_kat']:.2f}x Kat ({pref['fps']:.1f} FPS vs {tv['fps']:.1f} FPS)")
    print(f"* Batch Gecikmesindeki Tasarruf           : %{((tv['batch_gecikmesi_ms'] - pref['batch_gecikmesi_ms']) / tv['batch_gecikmesi_ms'])*100:.1f} Zaman Tasarrufu")
    print(f"* GPU Veri Bekleme (Starvation) Durumu     : Asenkron CUDA Stream ile Overlap Edildi.")

    # 3. Adım: Görselleştirme Panosu
    print("\n[+] 5. Adim: 6 Panelli Veri Boru Hatti Teshis Panosu Olusturuluyor...")
    grafik_yolu = VeriBoruHattiGorsellestirici.panoyu_ciz_ve_kaydet(
        benchmark_sonuclari=sonuclar,
        ornek_gorseller=ornek_gorseller,
        ornek_basliklar=ornek_basliklar,
        cikti_yolu=dashboard_yolu
    )
    print(f"[+] Teşhis Panosu Kaydedildi: {grafik_yolu}")
    print("=" * 95)
    print("DAY 68: HIGH PERFORMANCE DATA PIPELINE BASARIYLA TAMAMLANDI!")
    print("=" * 95)


if __name__ == "__main__":
    main()
