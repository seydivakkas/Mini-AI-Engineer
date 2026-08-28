"""
Day 74: Supervised Contrastive (SupCon) Ana Yürütülebilir Akış Scripti
---------------------------------------------------------------------
Etiketli veride SupCon kaybı ile temsil öğrenimi ve ardından Linear Probing
doğrulaması gerçekleştiren uçtan uca mini laboratuvar uygulaması.

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

from src.artirma_politikasi import TensorSupConArtirici
from src.supcon_model import SupConModeli
from src.egitim_motoru import SupConEgitimMotoru
from src.gorsellestirici import SupConGorsellestirici


def tohum_belirle(tohum: int = 42):
    random.seed(tohum)
    np.random.seed(tohum)
    torch.manual_seed(tohum)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tohum)


def sentetik_sinifli_veri_uret(ornek_sayisi: int = 500, sinif_sayisi: int = 5):
    """
    Belirgin renk ve geometrik desenlere sahip 5 sınıflı sentetik görsel kümesi.
    """
    veriler = []
    etiketler = []
    
    renk_paleti = [
        [0.8, 0.1, 0.1],  # Sınıf 0: Kırmızı
        [0.1, 0.8, 0.1],  # Sınıf 1: Yeşil
        [0.1, 0.1, 0.8],  # Sınıf 2: Mavi
        [0.8, 0.8, 0.1],  # Sınıf 3: Sarı
        [0.8, 0.1, 0.8],  # Sınıf 4: Magenta
    ]
    
    for i in range(ornek_sayisi):
        sinif_id = i % sinif_sayisi
        baz_renk = np.array(renk_paleti[sinif_id])
        
        # 3x32x32 görsel
        img = np.zeros((3, 32, 32), dtype=np.float32)
        for c in range(3):
            img[c, :, :] = baz_renk[c] + np.random.normal(0, 0.05, (32, 32))
            
        # Sınıfa özel geometrik desenler
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
            
        img = np.clip(img, 0.0, 1.0)
        veriler.append(img)
        etiketler.append(sinif_id)
        
    return torch.tensor(np.array(veriler), dtype=torch.float32), torch.tensor(etiketler, dtype=torch.long)


def main():
    print("=" * 75)
    print("🚀 Day 74: Supervised Contrastive (SupCon) Temsil Öğrenimi Başlatılıyor")
    print("=" * 75)
    
    tohum_belirle(42)
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📌 Çalışma Ortamı Cihazı: {cihaz.upper()}")

    # 1. Veri Hazırlığı
    print("\n[1/5] Sentetik 5 Sınıflı Görsel Veri Kümesi Üretiliyor...")
    x, y = sentetik_sinifli_veri_uret(ornek_sayisi=500, sinif_sayisi=5)
    
    # 400 Eğitim, 100 Doğrulama
    indices = list(range(500))
    random.shuffle(indices)
    train_idx, val_idx = indices[:400], indices[400:]
    
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    
    artirici = TensorSupConArtirici(goruntu_boyutu=32)
    v1_train, v2_train = artirici.cift_uret(x_train)
    v1_val, v2_val = artirici.cift_uret(x_val)
    
    train_ds = TensorDataset(v1_train, v2_train, y_train)
    val_ds = TensorDataset(v1_val, v2_val, y_val)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    print(f"  ✓ Eğitim Örnek Sayısı: {len(x_train)}, Doğrulama: {len(x_val)}, Batch Boyutu: 64")

    # 2. Model ve Eğitim Motoru Kurulumu
    print("\n[2/5] SupCon Modeli (Encoder + Projection Head + Linear Classifier) Başlatılıyor...")
    model = SupConModeli(giris_kanali=3, temsil_boyutu=128, projeksiyon_boyutu=64, sinif_sayisi=5)
    motor = SupConEgitimMotoru(model=model, sicaklik=0.1, ogrenme_orani=1e-3, cihaz=cihaz)

    # 3. Stage 1: SupCon ile Temsil Ön Eğitimi
    print("\n[3/5] Stage 1: Supervised Contrastive (SupCon) Temsil Öğrenimi (8 Epoch)...")
    print("-" * 75)
    print(f"{'Epoch':^7} | {'SupCon Loss':^13} | {'Sınıf İçi (Pos)':^17} | {'Sınıflar Arası (Neg)':^21} | {'Marjin':^9}")
    print("-" * 75)
    
    stage1_gecmis = motor.egit_stage1_kontrastif(train_loader, toplam_epoch=8)
    for i in range(len(stage1_gecmis["epoch"])):
        ep = stage1_gecmis["epoch"][i]
        l = stage1_gecmis["loss"][i]
        pos = stage1_gecmis["sinif_ici_kosinus"][i]
        neg = stage1_gecmis["siniflar_arasi_kosinus"][i]
        mar = stage1_gecmis["ayrisma_marjini"][i]
        print(f"{ep:^7} | {l:^13.4f} | {pos:^17.4f} | {neg:^21.4f} | {mar:^9.4f}")

    # 4. Stage 2: Linear Probing (Dondurulmuş Omurga Üzerinde Doğrusal Sınıflandırma)
    print("\n[4/5] Stage 2: Linear Probing (Dondurulmuş Omurga ile Doğrusal Sınıflandırma)...")
    print("-" * 75)
    print(f"{'Epoch':^7} | {'CE Loss':^13} | {'Validation Accuracy':^23}")
    print("-" * 75)
    
    stage2_gecmis = motor.egit_stage2_dogrusal_siniflandirici(train_loader, val_loader, toplam_epoch=5)
    for i in range(len(stage2_gecmis["epoch"])):
        ep = stage2_gecmis["epoch"][i]
        l = stage2_gecmis["loss"][i]
        acc = stage2_gecmis["dogruluk"][i]
        print(f"{ep:^7} | {l:^13.4f} | %{acc:^21.2f}")

    # 5. Temsil Çıkarımı, PCA Boyut İndirgeme ve 6 Panelli Görselleştirme
    print("\n[5/5] Temsiller Çıkarılıyor, PCA İzdüşümü ve 6 Panelli Pano Üretiliyor...")
    full_ds = TensorDataset(v1_train, v2_train, y_train)
    full_loader = DataLoader(full_ds, batch_size=64, shuffle=False)
    h_temsiller, etiketler_t = motor.temsilleri_cikar(full_loader)
    
    pca = PCA(n_components=2, random_state=42)
    temsiller_2d = pca.fit_transform(h_temsiller.numpy())
    print(f"  ✓ PCA Açıklanan Varyans Oranı: %{pca.explained_variance_ratio_.sum()*100:.2f}")

    # Görselleştirme için örnek çiftler
    ornek_ciftler = []
    for i in range(6):
        v1_np = v1_train[i].permute(1, 2, 0).numpy()
        v2_np = v2_train[i].permute(1, 2, 0).numpy()
        c = y_train[i].item()
        ornek_ciftler.append((v1_np, v2_np, c))

    gorsellestirici = SupConGorsellestirici()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "supcon_egitim_paneli.png")
    gorsellestirici.olustur_teshis_paneli(
        ornek_ciftler=ornek_ciftler,
        stage1_gecmisi=stage1_gecmis,
        stage2_gecmisi=stage2_gecmis,
        temsiller_2d=temsiller_2d,
        etiketler=etiketler_t.numpy(),
        kayit_yolu=cikti_yolu
    )
    print(f"  ✓ 6 Panelli Teşhis Panosu Kaydedildi: {cikti_yolu}")
    print("\n✅ Day 74: Supervised Contrastive Learning Başarıyla Tamamlandı!")


if __name__ == "__main__":
    main()
