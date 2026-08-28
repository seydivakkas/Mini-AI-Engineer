"""
Day 59: Transfer Learning ve Dondurulmuş Katmanlarla L2-Normalize Embedding Çıkarımı Ana Yürütme Betiği.
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

MEVCUT_DIZIN = os.path.abspath(os.path.dirname(__file__))
if MEVCUT_DIZIN not in sys.path:
    sys.path.insert(0, MEVCUT_DIZIN)

from src.vektor_ekstraktor import DondurulmusEmbeddingEkstraktoru, OmurgaModelFabrikasi
from src.embedding_analizoru import EmbeddingGeometriAnalizoru
from src.gorsellestirici import EmbeddingGorsellestirici


def sentetik_veri_olustur(num_samples: int = 1200, num_classes: int = 5, img_size: int = 32):
    """Sınıf ayrışmasını test etmek için sentetik kümelenmiş görsel tensörleri üretir."""
    torch.manual_seed(42)
    np.random.seed(42)

    X_list = []
    y_list = []

    for c in range(num_classes):
        n_c = num_samples // num_classes
        # Her sınıf için belirgin bir spektral frekans deseni
        frekans = (c + 1) * 1.5
        taban = torch.sin(torch.linspace(0, frekans * np.pi, img_size)).view(1, 1, 1, img_size)
        gurultu = torch.randn(n_c, 3, img_size, img_size) * 0.4
        ornekler = taban + gurultu
        X_list.append(ornekler)
        y_list.append(torch.full((n_c,), c, dtype=torch.long))

    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)

    # Karıştır
    perm = torch.randperm(len(X))
    X, y = X[perm], y[perm]

    ayrim = int(len(X) * 0.8)
    train_loader = DataLoader(TensorDataset(X[:ayrim], y[:ayrim]), batch_size=64, shuffle=False)
    val_loader = DataLoader(TensorDataset(X[ayrim:], y[ayrim:]), batch_size=64, shuffle=False)
    return train_loader, val_loader


def main():
    print("=" * 85, flush=True)
    print(">>> DAY 59: DONDURULMUŞ KATMANLARLA L2-NORMALIZE EMBEDDING ÇIKARIMI", flush=True)
    print("=" * 85, flush=True)

    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Kullanılan Cihaz: {cihaz.upper()}", flush=True)

    # 1. Veri Yükleyicileri ve Dondurulmuş Omurga Hazırlığı
    print("\n[+] 1. Adım: Veri Seti ve Dondurulmuş ResNet-512 Omurgası Oluşturuluyor...", flush=True)
    train_loader, val_loader = sentetik_veri_olustur()

    omurga = OmurgaModelFabrikasi.uret("resnet", feature_dim=512)
    ekstraktor = DondurulmusEmbeddingEkstraktoru(backbone=omurga, normalize=True, device=cihaz)

    # 2. Embedding Çıkarımı
    print("\n[+] 2. Adım: Toplu (Batched) L2-Normalize Embedding Çıkarımı Yapılıyor...", flush=True)
    train_emb, train_y = ekstraktor.cikart(train_loader)
    val_emb, val_y = ekstraktor.cikart(val_loader)
    print(f"    - Çıkarılan Eğitim Embeddingleri : {train_emb.shape} (float32)")
    print(f"    - Çıkarılan Doğrulama Embedding : {val_emb.shape} (float32)")

    # 3. L2 Norm Doğrulama ve Benzerlik Analizi
    print("\n[+] 3. Adım: Geometri ve Kosinüs Benzerliği Analizi...", flush=True)
    norm_bilgisi = EmbeddingGeometriAnalizoru.l2_norm_dogrula(train_emb)
    print(f"    - Ortalama L2 Normu           : {norm_bilgisi['ort_norm']:.6f}")
    print(f"    - Standart Sapma              : {norm_bilgisi['norm_std']:.6e}")
    print(f"    - Birim Hiperküre Doğrulandı mı: {norm_bilgisi['birim_kure_gecerli_mi']}")

    benzerlik_bilgisi = EmbeddingGeometriAnalizoru.benzerlik_analizi(train_emb, train_y)
    print(f"    - Sınıf-İçi Kosinüs Benzerliği : {benzerlik_bilgisi['ort_intra_benzerlik']:.4f}")
    print(f"    - Sınıf-Dışı Kosinüs Benzerliği: {benzerlik_bilgisi['ort_inter_benzerlik']:.4f}")
    print(f"    - Ayrışabilirlik Skoru (Ratio) : {benzerlik_bilgisi['ayrisabilirlik_orani']:.2f}x")

    # 4. SVD Spektrumu ve Linear Probing Testi
    print("\n[+] 4. Adım: SVD Spektrumu ve Linear Probe Sınıflandırıcı Eğitimi...", flush=True)
    svd_bilgisi = EmbeddingGeometriAnalizoru.svd_ve_izotropi_analizi(train_emb)
    print(f"    - Efektif Boyut (Participation): {svd_bilgisi['efektif_boyut']:.2f} / 512")
    print(f"    - İzotropi Skoru              : {svd_bilgisi['izotropi_skoru']:.4e}")

    probe_bilgisi = EmbeddingGeometriAnalizoru.linear_probing_egit(
        train_emb, train_y, val_emb, val_y, num_classes=5, epochs=15
    )
    print(f"    - Linear Probe Doğrulama Başarısı: %{probe_bilgisi['nihai_val_dogruluk']:.2f}")

    # 5. 6 Panelli Teşhis Panosunun Üretilmesi
    print("\n" + "=" * 85, flush=True)
    print(">>> 5. 6 PANELLİ EMBEDDING GEOMETRİSİ PANOSUNUN ÜRETİLMESİ", flush=True)
    print("=" * 85, flush=True)

    hedef_pano = os.path.join(MEVCUT_DIZIN, "ciktilar", "embedding_ekstraktor_paneli.png")
    cikis_yolu = EmbeddingGorsellestirici.panel_ciz(
        embeddings=train_emb,
        labels=train_y,
        norm_bilgisi=norm_bilgisi,
        benzerlik_bilgisi=benzerlik_bilgisi,
        svd_bilgisi=svd_bilgisi,
        probe_bilgisi=probe_bilgisi,
        omurga_adi="ResNet-512 (Frozen)",
        hedef_path=hedef_pano
    )
    print(f"[+] 6 Panelli Teşhis Panosu Kaydedildi: {os.path.abspath(cikis_yolu)}", flush=True)
    print("=" * 85, flush=True)
    print("DAY 59: TRANSFER LEARNING EMBEDDING EKSTRAKTOR BAŞARIYLA TAMAMLANDI!", flush=True)
    print("=" * 85, flush=True)


if __name__ == "__main__":
    main()
