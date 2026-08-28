"""
Day 75: Triplet Margin Loss ve Hard/Semi-Hard Negative Mining Ana Akış Scripti
-----------------------------------------------------------------------------
Metrik öğrenimi, mesafe ayrışması ve online madencilik dinamiklerini uçtan uca
çalıştıran yürütülebilir laboratuvar uygulaması.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
import sys
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.triplet_ag import MetrikOznitelikAgi
from src.egitim_dongusu import TripletEgitimMotoru
from src.gorsellestirici import TripletGorsellestirici


def tohum_belirle(tohum: int = 42):
    random.seed(tohum)
    np.random.seed(tohum)
    torch.manual_seed(tohum)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tohum)


def sentetik_metrik_veri_uret(ornek_sayisi: int = 600, sinif_sayisi: int = 6):
    """
    Belirgin renk ve dokusal geometriye sahip 6 sınıflı sentetik görsel kümesi.
    """
    veriler = []
    etiketler = []
    
    renk_paleti = [
        [0.85, 0.15, 0.15], # Sınıf 0: Kırmızı
        [0.15, 0.85, 0.15], # Sınıf 1: Yeşil
        [0.15, 0.15, 0.85], # Sınıf 2: Mavi
        [0.85, 0.85, 0.15], # Sınıf 3: Sarı
        [0.85, 0.15, 0.85], # Sınıf 4: Magenta
        [0.15, 0.85, 0.85], # Sınıf 5: Cyan
    ]
    
    for i in range(ornek_sayisi):
        sinif_id = i % sinif_sayisi
        baz_renk = np.array(renk_paleti[sinif_id])
        
        img = np.zeros((3, 32, 32), dtype=np.float32)
        for c in range(3):
            img[c, :, :] = baz_renk[c] + np.random.normal(0, 0.05, (32, 32))
            
        # Sınıfa özel desen varyasyonları
        if sinif_id == 0:
            img[:, 10:22, 10:22] += 0.2
        elif sinif_id == 1:
            img[:, :, 14:18] += 0.2
        elif sinif_id == 2:
            img[:, 14:18, :] += 0.2
        elif sinif_id == 3:
            np.fill_diagonal(img[0], 0.9)
        elif sinif_id == 4:
            img[:, 8:24, 8:24] -= 0.2
        elif sinif_id == 5:
            img[0, :, :16] += 0.15
            img[1, :, 16:] += 0.15

        img = np.clip(img, 0.0, 1.0)
        veriler.append(img)
        etiketler.append(sinif_id)
        
    return torch.tensor(np.array(veriler), dtype=torch.float32), torch.tensor(etiketler, dtype=torch.long)


def main():
    print("=" * 85)
    print("🚀 Day 75: Triplet Margin Loss & Hard/Semi-Hard Negative Mining Başlatılıyor")
    print("=" * 85)
    
    tohum_belirle(42)
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📌 Çalışma Ortamı Cihazı: {cihaz.upper()}")

    # 1. Veri Hazırlığı
    print("\n[1/4] Sentetik 6 Sınıflı Görsel Veri Kümesi Üretiliyor...")
    x, y = sentetik_metrik_veri_uret(ornek_sayisi=600, sinif_sayisi=6)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    print(f"  ✓ Toplam Örnek: {len(x)}, Sınıf Sayısı: 6, Batch Boyutu: 64")

    # 2. Model ve Triplet Motoru Kurulumu
    marjin = 0.3
    print(f"\n[2/4] Metrik Öznitelik Ağı ve Triplet Motoru Kuruluyor (Marjin α = {marjin})...")
    model = MetrikOznitelikAgi(giris_kanali=3, gomulme_boyutu=64)
    motor = TripletEgitimMotoru(
        model=model,
        marjin=marjin,
        strateji="batch_semi_hard",
        ogrenme_orani=1e-3,
        cihaz=cihaz
    )

    # 3. Triplet Metrik Eğitimi
    print("\n[3/4] Online Semi-Hard Negative Mining ile Triplet Eğitimi (8 Epoch)...")
    print("-" * 85)
    print(f"{'Epoch':^7} | {'Triplet Loss':^13} | {'d(a, p)':^9} | {'d(a, n)':^9} | {'Marjin':^9} | {'Aktif %':^8} | {'Hard %':^8} | {'Semi %':^8} | {'Easy %':^8}")
    print("-" * 85)
    
    gecmis = motor.egit(dataloader, toplam_epoch=8)
    for i in range(len(gecmis["epoch"])):
        ep = gecmis["epoch"][i]
        l = gecmis["loss"][i]
        d_ap = gecmis["d_ap"][i]
        d_an = gecmis["d_an"][i]
        mar = gecmis["marjin"][i]
        akt = gecmis["aktif_oran"][i]
        hrd = gecmis["zor_oran"][i]
        smh = gecmis["yari_zor_oran"][i]
        esy = gecmis["kolay_oran"][i]
        print(f"{ep:^7} | {l:^13.4f} | {d_ap:^9.4f} | {d_an:^9.4f} | {mar:^9.4f} | {akt:^7.1f}% | {hrd:^7.1f}% | {smh:^7.1f}% | {esy:^7.1f}%")

    # 4. Gömülme Çıkarımı, PCA ve 6 Panelli Teşhis Panosu
    print("\n[4/4] Gömülmeler Çıkarılıyor, PCA İzdüşümü ve Teşhis Panosu Üretiliyor...")
    eval_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    gomulmeler, etiketler_t = motor.gomulmeleri_cikar(eval_loader)
    
    pca = PCA(n_components=2, random_state=42)
    gomulmeler_2d = pca.fit_transform(gomulmeler.numpy())
    print(f"  ✓ PCA Açıklanan Varyans Oranı: %{pca.explained_variance_ratio_.sum()*100:.2f}")

    gorsellestirici = TripletGorsellestirici()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "triplet_mining_paneli.png")
    gorsellestirici.olustur_teshis_paneli(
        gecmis=gecmis,
        gomulmeler_2d=gomulmeler_2d,
        etiketler=etiketler_t.numpy(),
        marjin=marjin,
        kayit_yolu=cikti_yolu
    )
    print(f"  ✓ 6 Panelli Teşhis Panosu Kaydedildi: {cikti_yolu}")
    print("\n✅ Day 75: Triplet Metric Learning Başarıyla Tamamlandı!")


if __name__ == "__main__":
    main()
