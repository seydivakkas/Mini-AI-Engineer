"""
Day 73: Sıfırdan SimCLR Temsil Öğrenimi, Artırma Çiftleri, NT-Xent (InfoNCE) Kaybı
Ana Çalıştırma ve Eğitim Laboratuvarı

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA

from src.artirma_politikasi import TensorSimCLRArtirici
from src.simclr_model import SimCLRModeli
from src.nt_xent_loss import NTXentLoss
from src.egitim_dongusu import SimCLREgitimMotoru
from src.gorsellestirici import SimCLRGorsellestirici


class SentetikGorselKumesi(Dataset):
    """5 farklı görsel desene sahip sentetik görüntü veri kümesi."""
    def __init__(self, ornek_sayisi: int = 500, tohum: int = 42):
        torch.manual_seed(tohum)
        np.random.seed(tohum)
        
        self.goruntuler = []
        self.etiketler = []
        ornek_basina = ornek_sayisi // 5
        
        for sinif_idx in range(5):
            for _ in range(ornek_basina):
                # Her sınıf için karakteristik bir temel renk ve desen
                img = torch.zeros(3, 32, 32)
                if sinif_idx == 0:
                    img[0, :, :] = 0.8 + 0.1 * torch.randn(32, 32) # Kırmızı baskın
                elif sinif_idx == 1:
                    img[1, :, :] = 0.8 + 0.1 * torch.randn(32, 32) # Yeşil baskın
                elif sinif_idx == 2:
                    img[2, :, :] = 0.8 + 0.1 * torch.randn(32, 32) # Mavi baskın
                elif sinif_idx == 3:
                    # Çapraz çizgili
                    for r in range(32):
                        img[:, r, r] = 1.0
                    img += 0.1 * torch.randn(3, 32, 32)
                elif sinif_idx == 4:
                    # Dama tahtası deseni
                    img[:, ::4, ::4] = 0.9
                    img += 0.1 * torch.randn(3, 32, 32)
                    
                img = torch.clamp(img, 0.0, 1.0)
                self.goruntuler.append(img)
                self.etiketler.append(sinif_idx)
                
        self.artirici = TensorSimCLRArtirici(goruntu_boyutu=32)

    def __len__(self):
        return len(self.goruntuler)

    def __getitem__(self, idx):
        img = self.goruntuler[idx]
        label = self.etiketler[idx]
        v1, v2 = self.artirici.cift_uret(img)
        return v1, v2, label


def main():
    print("=" * 95)
    print(">>> DAY 73: SIFIRDAN SimCLR TEMSİL ÖĞRENİMİ, ARTIRMA ÇİFTLERİ & NT-Xent (InfoNCE) LABORATUVARI")
    print("=" * 95)

    # 1. Adım: Veri Kümesi ve DataLoader Hazırlığı
    print("\n[+] 1. Adım: Sentetik Çok Sınıflı Veri Kümesi ve Çift Görünüm Üreteci Oluşturuluyor (N=500, C=5)...")
    veri_kumesi = SentetikGorselKumesi(ornek_sayisi=500, tohum=42)
    veri_yukleyici = DataLoader(veri_kumesi, batch_size=64, shuffle=True, drop_last=True)
    print(f"    * Toplam Görüntü: {len(veri_kumesi)} | Batch Boyutu: 64 | Batch Sayısı: {len(veri_yukleyici)}")

    # 2. Adım: SimCLR Modeli ve Eğitim Motoru Kurulumu
    print("\n[+] 2. Adım: SimCLR Modeli (f: Temel Kodlayıcı + g: Non-lineer MLP Projeksiyon) Başlatılıyor...")
    model = SimCLRModeli(giris_kanali=3, temsil_boyutu=128, projeksiyon_boyutu=64)
    motor = SimCLREgitimMotoru(
        model=model,
        sicaklik=0.5,
        ogrenme_orani=2e-3,
        agirlik_cezasi=1e-4,
        toplam_epoch=8,
        cihaz="cpu"
    )

    # 3. Adım: Kendi Kendine Denetimli (Self-Supervised) Ön Eğitim
    print("\n[+] 3. Adım: SimCLR Ön Eğitimi Başlatılıyor (8 Epoch, NT-Xent Kaybı)...")
    print("-" * 95)
    print(f"{'Epoch':<6} | {'NT-Xent Kaybı':<15} | {'Alignment (||z1-z2||²)':<24} | {'Pozitif Cos':<12} | {'Negatif Cos':<12} | {'Ayrışma Marjini':<15}")
    print("-" * 95)

    for ep in range(1, 9):
        ep_metrik = motor.bir_epoch_egit(veri_yukleyici)
        motor.gecmis["epoch"].append(ep)
        motor.gecmis["loss"].append(ep_metrik["loss"])
        motor.gecmis["lr"].append(ep_metrik["lr"])
        motor.gecmis["alignment_loss"].append(ep_metrik["alignment_loss"])
        motor.gecmis["uniformity_loss"].append(ep_metrik["uniformity_loss"])
        motor.gecmis["pozitif_kosinus"].append(ep_metrik["pozitif_kosinus"])
        motor.gecmis["negatif_kosinus"].append(ep_metrik["negatif_kosinus"])
        motor.gecmis["kosinus_marjini"].append(ep_metrik["kosinus_marjini"])

        print(
            f"{ep:<6} | {ep_metrik['loss']:<15.4f} | {ep_metrik['alignment_loss']:<24.4f} | "
            f"{ep_metrik['pozitif_kosinus']:<12.4f} | {ep_metrik['negatif_kosinus']:<12.4f} | "
            f"{ep_metrik['kosinus_marjini']:<15.4f}"
        )

    # 4. Adım: Temsil Kalitesi Değerlendirmesi ve İzdüşüm
    print("\n[+] 4. Adım: Öğrenilen Temsiller (h vektörleri) Çıkarılıyor ve PCA ile İzdüşürülüyor...")
    temsiller_h, etiketler = motor.temsilleri_cikar(veri_yukleyici)
    pca = PCA(n_components=2, random_state=42)
    temsiller_2d = pca.fit_transform(temsiller_h.numpy())
    print(f"    * Temsil Matrisi Şekli: {temsiller_h.shape} | PCA Açıklanan Varyans: %{np.sum(pca.explained_variance_ratio_)*100:.2f}")

    # 5. Adım: Görsel Teşhis Panosunun Oluşturulması
    print("\n[+] 5. Adım: 6 Panelli Teşhis Panosu Çizdiriliyor...")
    
    # İlk 4 görüntüden örnek çiftler topla
    ornek_ciftler = []
    for i in range(4):
        v1, v2, _ = veri_kumesi[i]
        ornek_ciftler.append((v1.permute(1, 2, 0).numpy(), v2.permute(1, 2, 0).numpy()))
        
    cikti_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    kayit_yolu = os.path.join(cikti_dizini, "simclr_egitim_paneli.png")
    
    gorsellestirici = SimCLRGorsellestirici()
    gorsellestirici.olustur_teshis_paneli(
        ornek_ciftler=ornek_ciftler,
        egitim_gecmisi=motor.gecmis,
        temsiller_2d=temsiller_2d,
        etiketler=etiketler.numpy(),
        kayit_yolu=kayit_yolu
    )
    print(f"[+] Teşhis Panosu Başarıyla Kaydedildi: {kayit_yolu}")
    print("=" * 95)
    print("DAY 73: SimCLR FROM SCRATCH & NT-Xent BAŞARIYLA TAMAMLANDI!")
    print("=" * 95)


if __name__ == "__main__":
    main()
