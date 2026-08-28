"""
Day 76: Temsil Kalitesi Değerlendirme Paketi Ana Akış Scripti
------------------------------------------------------------
Rastgele başlatılmış temel model ile kontrastif/metrik öğrenimle eğitilmiş
modeli Linear Probing, k-NN ve Geometrik Manifold metrikleriyle kıyaslayan laboratuvar.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.temsil_cikarici import TemsilCikarici
from src.benchmark_suite import TemsilDegerlendirmePaketi
from src.gorsellestirici import BenchmarkGorsellestirici


def tohum_belirle(tohum: int = 42):
    random.seed(tohum)
    np.random.seed(tohum)
    torch.manual_seed(tohum)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tohum)


class BasitOmurga(nn.Module):
    """32x32 görseli 64 boyutlu normalize embedding'e dönüştüren ağ."""
    def __init__(self, giris_kanali: int = 3, cikti_boyutu: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(giris_kanali, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, cikti_boyutu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        h = self.conv(x).view(b, -1)
        z = self.fc(h)
        return F.normalize(z, p=2, dim=1)


def sentetik_veri_uret(ornek_sayisi: int = 600, sinif_sayisi: int = 6):
    veriler = []
    etiketler = []
    renk_paleti = [
        [0.85, 0.15, 0.15],
        [0.15, 0.85, 0.15],
        [0.15, 0.15, 0.85],
        [0.85, 0.85, 0.15],
        [0.85, 0.15, 0.85],
        [0.15, 0.85, 0.85],
    ]
    for i in range(ornek_sayisi):
        c_id = i % sinif_sayisi
        base = np.array(renk_paleti[c_id])
        img = np.zeros((3, 32, 32), dtype=np.float32)
        for c in range(3):
            img[c, :, :] = base[c] + np.random.normal(0, 0.05, (32, 32))
            
        if c_id == 0:
            img[:, 10:22, 10:22] += 0.2
        elif c_id == 1:
            img[:, :, 14:18] += 0.2
        elif c_id == 2:
            img[:, 14:18, :] += 0.2
        elif c_id == 3:
            np.fill_diagonal(img[0], 0.9)
        elif c_id == 4:
            img[:, 8:24, 8:24] -= 0.2
        elif c_id == 5:
            img[0, :, :16] += 0.15

        img = np.clip(img, 0.0, 1.0)
        veriler.append(img)
        etiketler.append(c_id)
        
    return torch.tensor(np.array(veriler), dtype=torch.float32), torch.tensor(etiketler, dtype=torch.long)


def main():
    print("=" * 80)
    print("🚀 Day 76: Temsil Kalitesi Değerlendirme Paketi (Representation Benchmark)")
    print("=" * 80)
    
    tohum_belirle(42)
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📌 Çalışma Ortamı Cihazı: {cihaz.upper()}")

    # 1. Veri Hazırlığı
    print("\n[1/5] Sentetik 6 Sınıflı Veri Kümesi Hazırlanıyor...")
    x, y = sentetik_veri_uret(ornek_sayisi=600, sinif_sayisi=6)
    
    # 450 Train, 150 Val
    indices = list(range(600))
    random.shuffle(indices)
    train_idx, val_idx = indices[:450], indices[450:]
    
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    
    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    print(f"  ✓ Eğitim Örnek Sayısı: {len(x_train)}, Doğrulama Örnek Sayısı: {len(x_val)}")

    # 2. Modeller: Rastgele Baseline vs Önceden Eğitilmiş Model
    print("\n[2/5] Modeller Kuruluyor: Rastgele Ağırlıklar vs Eğitilmiş Temsil Modeli...")
    model_rastgele = BasitOmurga(3, 64)
    model_egitilmis = BasitOmurga(3, 64)
    
    # Eğitilmiş model için hızlı 5 epoch ön eğitim (simülasyon)
    opt = torch.optim.Adam(model_egitilmis.parameters(), lr=1e-3)
    train_sim_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    model_egitilmis.train()
    model_egitilmis.to(cihaz)
    
    for _ in range(6):
        for bx, by in train_sim_loader:
            bx, by = bx.to(cihaz), by.to(cihaz)
            opt.zero_grad()
            emb = model_egitilmis(bx)
            # Sınıf merkezlerine çeken prototip kaybı
            sim_mat = torch.matmul(emb, emb.T) / 0.1
            mask = torch.eq(by.view(-1, 1), by.view(1, -1)).float()
            mask.fill_diagonal_(0.0)
            if mask.sum() > 0:
                loss = - (mask * F.log_softmax(sim_mat, dim=1)).sum() / mask.sum()
                loss.backward()
                opt.step()

    # 3. Dondurulmuş Temsillerin Çıkarımı
    print("\n[3/5] Dondurulmuş Temsil Vektörleri Çıkarılıyor...")
    cikarici_rastgele = TemsilCikarici(model_rastgele, cihaz=cihaz)
    cikarici_egitilmis = TemsilCikarici(model_egitilmis, cihaz=cihaz)

    h_train_rand, _ = cikarici_rastgele.cikar(train_loader)
    h_val_rand, _ = cikarici_rastgele.cikar(val_loader)

    h_train_trained, _ = cikarici_egitilmis.cikar(train_loader)
    h_val_trained, _ = cikarici_egitilmis.cikar(val_loader)

    # 4. Kapsamlı Benchmark Süiti Koşturuluyor
    print("\n[4/5] Kapsamlı Benchmark Süiti Koşturuluyor (Linear Probe, k-NN, Few-Shot, Geometri)...")
    suite = TemsilDegerlendirmePaketi(temsil_boyutu=64, sinif_sayisi=6, cihaz=cihaz)

    sonuc_rastgele = suite.calistir_kapsamli_benchmark(h_train_rand, y_train, h_val_rand, y_val)
    sonuc_egitilmis = suite.calistir_kapsamli_benchmark(h_train_trained, y_train, h_val_trained, y_val)

    print("\n" + "=" * 80)
    print(f"{'Değerlendirme Protokolü':^35} | {'Rastgele Model':^18} | {'Eğitilmiş Model':^18}")
    print("=" * 80)
    print(f"{'Linear Probe (%100 Etiket)':<35} | %{sonuc_rastgele['linear_probe_100']:^16.2f} | %{sonuc_egitilmis['linear_probe_100']:^16.2f}")
    print(f"{'Linear Probe Few-Shot (%10 Etiket)':<35} | %{sonuc_rastgele['linear_probe_10']:^16.2f} | %{sonuc_egitilmis['linear_probe_10']:^16.2f}")
    print(f"{'Linear Probe Few-Shot (%2 Etiket)':<35} | %{sonuc_rastgele['linear_probe_fewshot']:^16.2f} | %{sonuc_egitilmis['linear_probe_fewshot']:^16.2f}")
    print(f"{'Non-Parametric k-NN (k=1)':<35} | %{sonuc_rastgele['knn_k_1']:^16.2f} | %{sonuc_egitilmis['knn_k_1']:^16.2f}")
    print(f"{'Non-Parametric k-NN (k=5)':<35} | %{sonuc_rastgele['knn_k_5']:^16.2f} | %{sonuc_egitilmis['knn_k_5']:^16.2f}")
    print(f"{'Non-Parametric k-NN (k=20)':<35} | %{sonuc_rastgele['knn_k_20']:^16.2f} | %{sonuc_egitilmis['knn_k_20']:^16.2f}")
    print("-" * 80)
    print(f"{'Silhouette Küme Skoru (-1 ila +1)':<35} | {sonuc_rastgele['silhouette_skoru']:^18.3f} | {sonuc_egitilmis['silhouette_skoru']:^18.3f}")
    print(f"{'İzotropi İndeksi (min/max s.v.)':<35} | {sonuc_rastgele['izotropi_indeksi']:^18.3f} | {sonuc_egitilmis['izotropi_indeksi']:^18.3f}")
    print(f"{'Efektif Boyut (SVD Entropisi)':<35} | {sonuc_rastgele['efektif_boyut']:^18.1f} | {sonuc_egitilmis['efektif_boyut']:^18.1f}")
    print(f"{'Sınıf Ayrışma Marjini':<35} | {sonuc_rastgele['ayrisma_marjini']:^18.3f} | {sonuc_egitilmis['ayrisma_marjini']:^18.3f}")
    print("=" * 80)

    # 5. Few-Shot Eğrisi ve 6 Panelli Görselleştirme
    print("\n[5/5] Few-Shot Eğrisi ve 6 Panelli Teşhis Panosu Üretiliyor...")
    few_shot_oranlar = [1, 5, 10, 25, 50, 100]
    few_shot_dogruluklar = []
    for r in few_shot_oranlar:
        res = suite.linear_probe.egit_ve_degerlendir(
            h_train_trained, y_train, h_val_trained, y_val, etiket_orani=r/100.0, epoch_sayisi=15
        )
        few_shot_dogruluklar.append(res["dogruluk_yuzdesi"])

    pca = PCA(n_components=2, random_state=42)
    h_2d = pca.fit_transform(h_val_trained.numpy())

    gorsellestirici = BenchmarkGorsellestirici()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "temsil_benchmark_paneli.png")
    gorsellestirici.olustur_teshis_paneli(
        egitilmis_sonuclar=sonuc_egitilmis,
        rastgele_sonuclar=sonuc_rastgele,
        few_shot_egrisi={"oranlar": few_shot_oranlar, "dogruluklar": few_shot_dogruluklar},
        gomulmeler_2d=h_2d,
        etiketler=y_val.numpy(),
        kayit_yolu=cikti_yolu
    )
    print(f"  ✓ 6 Panelli Teşhis Panosu Kaydedildi: {cikti_yolu}")
    print("\n✅ Day 76: Temsil Kalitesi Değerlendirme Paketi Başarıyla Tamamlandı!")


if __name__ == "__main__":
    main()
