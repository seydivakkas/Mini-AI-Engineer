"""
Day 72: t-SNE, UMAP Boyut İndirgeme, Temsil Uzayı Geometrisi & İzotropi Analizi
Ana Çalıştırma ve Karşılaştırma Laboratuvarı

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
import sys
import numpy as np

# Proje modüllerini içe aktar
from src.model_ozellik_cikarici import TemsilVeriUreteci, GorselTemsilAgi
from src.boyut_indirgeme import BoyutIndirgemeMotoru
from src.geometri_analizoru import TemsilGeometrisiAnalizoru
from src.gorsellestirici import TemsilGeometrisiGorsellestirici


def main():
    print("=" * 95)
    print(">>> DAY 72: t-SNE, UMAP BOYUT İNDİRGEME, TEMSİL UZAYI GEOMETRİSİ & İZOTROPİ ANALİZİ")
    print("=" * 95)

    # 1. Adım: Kontrollü Temsil Kümelerinin Üretimi
    print("\n[+] 1. Adım: Yapılandırılmış ve Çökmüş Temsil Kümeleri Üretiliyor (N=600, D=64, C=5)...")
    X_normal, y_normal, meta = TemsilVeriUreteci.uret_kontrollu_temsiller(
        ornek_sayisi=600, boyut=64, sinif_sayisi=5, gurultu=0.30, tohum=42
    )
    X_cokmus, _ = TemsilVeriUreteci.uret_boyutsal_cokmus_temsiller(
        ornek_sayisi=600, boyut=64, efektif_boyut=2, tohum=42
    )
    print(f"    * Normal Temsil Boyutu : {X_normal.shape} | Sınıf Dağılımı: {len(np.unique(y_normal))} Sınıf")
    print(f"    * Çökmüş Temsil Boyutu : {X_cokmus.shape} | Efektif Boyut: 2")

    # 2. Adım: Boyut İndirgeme İşlemleri (PCA, t-SNE, UMAP)
    print("\n[+] 2. Adım: Boyut İndirgeme Algoritmaları Çalıştırılıyor...")
    motor = BoyutIndirgemeMotoru(rastgele_tohum=42)
    
    print("    * PCA İzdüşümü Hesaplanıyor...")
    X_pca, pca_varyans = motor.uygula_pca(X_normal, bilesen_sayisi=2)
    
    print("    * t-SNE İzdüşümü Hesaplanıyor (Perplexity=30)...")
    X_tsne, tsne_kl = motor.uygula_tsne(X_normal, bilesen_sayisi=2, perplexity=30.0)
    
    print("    * UMAP İzdüşümü Hesaplanıyor (Metric=Cosine, Neighbors=15)...")
    X_umap = motor.uygula_umap(X_normal, bilesen_sayisi=2, komsu_sayisi=15, metrik="cosine")

    # 3. Adım: Geometri ve İzotropi Analizleri
    print("\n[+] 3. Adım: SVD Spektrumu, İzotropi ve Kosinüs Geometrisi Hesaplanıyor...")
    analizor = TemsilGeometrisiAnalizoru()
    
    izotropi_norm = analizor.hesapla_izotropi(X_normal)
    izotropi_cok = analizor.hesapla_izotropi(X_cokmus)
    kosinus_sonuclari = analizor.hesapla_kosinus_geometrisi(X_normal, y_normal)
    
    teshis_norm = analizor.teshis_boyutsal_cokus(X_normal, esik_varyans=0.85)
    teshis_cok = analizor.teshis_boyutsal_cokus(X_cokmus, esik_varyans=0.85)

    # 4. Adım: Metrik Raporu Tablosu
    print("\n" + "=" * 95)
    print(">>> 4. TEMSİL UZAYI GEOMETRİSİ VE İNDİRGEME METRİK RAPORU")
    print("=" * 95)
    print(f"{'Metrik Adı':<35} | {'Sağlıklı Temsil':<25} | {'Çökmüş Temsil':<25}")
    print("-" * 95)
    print(f"{'İzotropi İndeksi (exp(H) / d)':<35} | {izotropi_norm['izotropi_skoru']:<25.4f} | {izotropi_cok['izotropi_skoru']:<25.4f}")
    print(f"{'Min/Max Tekil Değer Oranı':<35} | {izotropi_norm['min_max_tekil_orani']:<25.6f} | {izotropi_cok['min_max_tekil_orani']:<25.6f}")
    print(f"{'Tekil Değer Entropisi (H)':<35} | {izotropi_norm['tekil_deger_entropisi']:<25.4f} | {izotropi_cok['tekil_deger_entropisi']:<25.4f}")
    print(f"{'Efektif Boyut (exp(H))':<35} | {izotropi_norm['efektif_boyut']:<25.2f} / 64      | {izotropi_cok['efektif_boyut']:<25.2f} / 64")
    print(f"{'İlk 3 Eksen Varyans Payı':<35} | %{izotropi_norm['kumulatif_varyans'][2]*100:<24.1f} | %{izotropi_cok['kumulatif_varyans'][2]*100:<24.1f}")
    print(f"{'Boyutsal Çöküş Teşhisi':<35} | {'HAYIR (Sağlıklı)':<25} | {'EVET (ÇÖKÜŞ!)':<25}")
    print("-" * 95)
    print(f"{'Sınıf İçi Kosinüs Benzerliği':<35} | {kosinus_sonuclari['sinif_ici_ortalama_kosinus']:<25.4f} | {'N/A':<25}")
    print(f"{'Sınıflar Arası Kosinüs':<35} | {kosinus_sonuclari['siniflar_arasi_ortalama_kosinus']:<25.4f} | {'N/A':<25}")
    print(f"{'Kosinüs Ayrışma Marjini':<35} | {kosinus_sonuclari['ayrisma_marjini']:<25.4f} | {'N/A':<25}")
    print(f"{'PCA Açıklanan Toplam Varyans':<35} | %{np.sum(pca_varyans)*100:<24.2f} | {'N/A':<25}")
    print(f"{'t-SNE KL Diverjans Kaybı':<35} | {tsne_kl:<25.4f} | {'N/A':<25}")

    # 5. Adım: Görsel Teşhis Panosunun Oluşturulması
    print("\n[+] 5. Adım: 6 Panelli Teşhis Panosu Çizdiriliyor...")
    cikti_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    kayit_yolu = os.path.join(cikti_dizini, "temsil_geometrisi_paneli.png")
    
    gorsellestirici = TemsilGeometrisiGorsellestirici()
    gorsellestirici.olustur_teshis_paneli(
        X_pca=X_pca,
        pca_varyans=pca_varyans,
        X_tsne=X_tsne,
        tsne_kl=tsne_kl,
        X_umap=X_umap,
        y=y_normal,
        izotropi_normal=izotropi_norm,
        izotropi_cokmus=izotropi_cok,
        kosinus_metrikleri=kosinus_sonuclari,
        kayit_yolu=kayit_yolu
    )
    print(f"[+] Teşhis Panosu Başarıyla Kaydedildi: {kayit_yolu}")
    print("=" * 95)
    print("DAY 72: EMBEDDING GEOMETRY & MANIFOLD VISUALIZATION BAŞARIYLA TAMAMLANDI!")
    print("=" * 95)


if __name__ == "__main__":
    main()
