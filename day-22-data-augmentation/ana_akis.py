"""Day 22 Ana Çalıştırma Akışı: Veri Çoğaltma (Data Augmentation) ve Veri Hikayeciliği.

Bu betik; Albumentations, torchvision.transforms, MixUp ve CutMix tekniklerini
uçtan uca uygular, 4 farklı strateji ile eğitilen CNN modellerinin genelleme ve gürültüye
dayanıklılık farklarını deneysel olarak kanıtlar ve veri hikayesi raporlarını üretir.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Proje kök dizinini ekle
proje_kok = Path(__file__).resolve().parent
if str(proje_kok) not in sys.path:
    sys.path.insert(0, str(proje_kok))

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch

from src.albumentations_donusturucu import AlbumentationsDonusturucu
from src.torchvision_donusturucu import TorchvisionDonusturucu
from src.mixup_cutmix import MixUpCutMixUygulayici, MixUpCutMixKayip
from src.karsilastirici import VeriCogaltmaKarsilastirici, StratejiSonucu
from src.gorsellestirici import VeriCogaltmaGorsellestirici


def baslik(metin: str) -> None:
    """Konsol çıktıları için formatlı bölüm başlığı basar."""
    cizgi = "=" * 80
    print(f"\n{cizgi}\n>>> {metin}\n{cizgi}")


def sentetik_veri_uret(sinif_basina: int = 50, H: int = 64, W: int = 64) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """4 sınıflı sentetik görsel veri seti üretir."""
    siniflar = ["Vazo", "Kumaş", "Rozet", "Ahşap"]
    X, y = [], []
    np.random.seed(42)

    for sinif_idx, sinif_adi in enumerate(siniflar):
        for i in range(sinif_basina):
            img = np.zeros((H, W, 3), dtype=np.float32)
            if sinif_adi == "Vazo":
                for r in range(H):
                    img[r, :, 0] = 0.1 + 0.6 * (r / H)
                    img[r, :, 1] = 0.3 + 0.4 * (r / H)
                    img[r, :, 2] = 0.8
                cv2.ellipse(img, (W // 2, H // 2), (W // 4, H // 3), 0, 0, 360, (0.9, 0.9, 0.2), -1)
            elif sinif_adi == "Kumaş":
                img[:, :, 2] = 0.6 + 0.3 * np.sin(np.linspace(0, 12 * np.pi, W))
                img[:, :, 0] = 0.2
                img[:, :, 1] = 0.2
                for x in range(0, W, 6):
                    cv2.line(img, (x, 0), (x, H), (0.1, 0.8, 0.8), 1)
            elif sinif_adi == "Rozet":
                img[:, :] = (0.1, 0.1, 0.1)
                cv2.circle(img, (W // 2, H // 2), W // 3, (0.1, 0.8, 0.9), 3)
                cv2.circle(img, (W // 2, H // 2), W // 5, (0.2, 0.9, 1.0), -1)
            elif sinif_adi == "Ahşap":
                for r in range(H):
                    c = 0.2 + 0.5 * (np.sin(r * 0.4) ** 2)
                    img[r, :, 0] = c * 0.3
                    img[r, :, 1] = c * 0.6
                    img[r, :, 2] = c * 0.9

            gurultu = np.random.normal(0, 0.03, (H, W, 3)).astype(np.float32)
            img = np.clip(img + gurultu, 0.0, 1.0)
            X.append(img)
            y.append(sinif_idx)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), siniflar


def main() -> None:
    """Day 22 veri çoğaltma ve karşılaştırma ana akışını yürütür."""
    baslik("AŞAMA 1: Görsel Veri Setinin Hazırlanması")
    X, y, sinif_isimleri = sentetik_veri_uret(sinif_basina=50, H=64, W=64)
    print(f"[+] Üretilen Toplam Görsel: {len(X)} adet")
    print(f"[+] Tensör Boyutları: {X.shape}")
    print(f"[+] Sınıflar: {sinif_isimleri}")

    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    print(f"[+] Eğitim Kümesi (Train): {len(X_train)} örnek")
    print(f"[+] Test Kümesi (Test)   : {len(X_test)} örnek")

    baslik("AŞAMA 2: Albumentations ve Torchvision Dönüşüm Boru Hatlarının Doğrulanması")
    albu = AlbumentationsDonusturucu((64, 64))
    tv = TorchvisionDonusturucu((64, 64))

    ornek_img = X_train[0]
    albu_img = albu.donustur_tekil(ornek_img, mod="agir")
    tv_img = tv.donustur_numpy(ornek_img)

    print(f"[+] Albumentations Çıktı Şekli: {albu_img.shape} | Piksel Aralığı: [{albu_img.min():.2f}, {albu_img.max():.2f}]")
    print(f"[+] Torchvision Transforms Çıktı: {tv_img.shape} | Piksel Aralığı: [{tv_img.min():.2f}, {tv_img.max():.2f}]")

    baslik("AŞAMA 3: İleri Düzey MixUp ve CutMix Dönüşümlerinin Test Edilmesi")
    ornekler_x = torch.from_numpy(np.transpose(X_train[:4], (0, 3, 1, 2))).float()
    ornekler_y = torch.from_numpy(y_train[:4]).long()

    mix_x, ya_m, yb_m, lam_m = MixUpCutMixUygulayici.uygula_mixup(ornekler_x, ornekler_y, alpha=0.8)
    cut_x, ya_c, yb_c, lam_c = MixUpCutMixUygulayici.uygula_cutmix(ornekler_x, ornekler_y, alpha=1.0)

    print(f"[+] MixUp Lambda: {lam_m:.3f} | Tensör: {list(mix_x.shape)}")
    print(f"[+] CutMix Lambda: {lam_c:.3f} | Tensör: {list(cut_x.shape)}")

    baslik("AŞAMA 4: Dört Farklı Strateji ile Karşılaştırmalı Ablation Deneyleri")
    karsilastirici = VeriCogaltmaKarsilastirici()
    stratejiler = ["Baseline (Dönüşümsüz)", "Albumentations", "MixUp", "CutMix"]
    sonuclar: List[StratejiSonucu] = []

    for strat in stratejiler:
        print(f"[*] Strateji Eğitiliyor: {strat}...")
        res = karsilastirici.egit_ve_test_et(
            strateji=strat,
            X_train_np=X_train,
            y_train_np=y_train,
            X_test_np=X_test,
            y_test_np=y_test,
            epochs=18,
            batch_size=16,
        )
        sonuclar.append(res)
        print(f"    - Temiz Test Doğruluğu   : %{res.test_acc * 100:.2f}")
        print(f"    - Gürültülü Test Doğruluğu: %{res.gurultulu_test_acc * 100:.2f}")
        print(f"    - F1-Macro Skoru         : {res.f1_macro:.4f}")

    baslik("AŞAMA 5: Veri Hikayesi (Data Storyteller) Yönetici Özeti")
    print(f"{'Strateji':<25} | {'Temiz Test Acc':<16} | {'Gürültülü Test Acc':<20} | {'F1-Macro':<10} | {'Dirençlilik (Robustness)':<24}")
    print("-" * 105)
    for s in sonuclar:
        direnc = "Yüksek (Dirençli)" if s.gurultulu_test_acc >= 0.85 else ("Orta" if s.gurultulu_test_acc >= 0.65 else "Düşük (Kırılgan)")
        print(f"{s.strateji_adi:<25} | %{s.test_acc * 100:<15.1f} | %{s.gurultulu_test_acc * 100:<19.1f} | {s.f1_macro:<10.4f} | {direnc:<24}")

    baslik("AŞAMA 6: Görselleştirme Çıktılarının Kaydedilmesi")
    cikti_klasor = proje_kok / "ciktilar"
    cikti_klasor.mkdir(parents=True, exist_ok=True)

    # 1. Galeri Çizimi
    galeri_dosyasi = cikti_klasor / "veri_cogaltma_galerisi.png"
    ornek_temsilciler = [X_train[y_train == i][0] for i in range(4)]
    ornek_etiketler = np.array([0, 1, 2, 3])
    VeriCogaltmaGorsellestirici.galeri_ciz(
        X_ornekler=np.array(ornek_temsilciler),
        y_ornekler=ornek_etiketler,
        sinif_isimleri=sinif_isimleri,
        hedef_dosya=galeri_dosyasi,
    )
    print(f"[+] Veri çoğaltma görselleştirme galerisi kaydedildi: {galeri_dosyasi}")

    # 2. Karşılaştırma Raporu
    rapor_dosyasi = cikti_klasor / "veri_cogaltma_karsilastirma_raporu.png"
    VeriCogaltmaGorsellestirici.karsilastirma_raporu_ciz(
        sonuclar=sonuclar,
        hedef_dosya=rapor_dosyasi,
    )
    print(f"[+] Karşılaştırmalı veri hikayesi raporu kaydedildi: {rapor_dosyasi}")


if __name__ == "__main__":
    main()
