"""
Day 131: Semantik Parçalama (Semantic Chunking) & Dinamik RAG Bölümleme Ana Akışı.
Sabit boyutlu parçalamaya karşı anlamsal kosinüs mesafesi ve dinamik eşikleme ile metin bölümleme.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.semantik_parcalayici import SemantikParcalayici
from src.rag_karsilastirici import RAGParcalamaKarsilastirici
from src.gorsellestirici import SemanticChunkingGorsellestirici


def main():
    print("=" * 105)
    print(">>> Day 131: Semantic Chunking & Dynamic Text Partitioning for Advanced RAG")
    print("=" * 105)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # Çok konulu uzun teknik belge
    teknik_belge = (
        "Bölüm 1: Derin Öğrenme ve Vision Transformer Mimarisi. "
        "Vision Transformer (ViT) görüntüleri 16x16 yamalara bölerek doğrusal projeksiyon uygular. "
        "Öz-dikkat (Self-Attention) mekanizması küresel piksel ilişkilerini başarıyla modeller. "
        "Büyük ölçekli veri kümelerinde ön eğitim yapıldığında evrişimli ağları geride bırakır. "
        "Bölüm 2: Dağıtık Veritabanları ve Raft Konsensüs Protokolü. "
        "Raft lider seçimi ve günlük çoğaltma (Log Replication) prensipleriyle çalışır. "
        "Ağ bölünmesi (Network Partition) anında çoğunluk (Quorum) kuralı veri tutarlılığını garanti eder. "
        "CAP teoremine göre Raft tabanlı sistemler genellikle CP (Tutarlılık ve Bölünme Toleransı) sınıfındadır. "
        "Bölüm 3: Yüksek Frekanslı Algoritmik Ticaret ve Emir Eşleştirme. "
        "Limit Emir Defteri (LOB) gelen alım ve satım emirlerini fiyat ve zaman önceliğine göre sıralar. "
        "Mikrosaniyelik FPGA hızlandırıcılar emir iletim gecikmesini sıfıra yaklaştırır. "
        "Piyasa yapıcı algoritmalar likidite sağlayarak alış-satış farkından kâr elde eder."
    )

    print("\n[1/3] Çok Konulu Teknik Belge Semantik Olarak Analiz Ediliyor...")

    parcalayici = SemantikParcalayici(
        vektor_boyutu=128,
        tampon_boyutu=1,
        esik_yontemi="standart_sapma",
        esik_katsayisi=0.45,
    )
    sonuc = parcalayici.parcala(teknik_belge)

    print(f"  [✓] Toplam Ayrıştırılan Cümle Sayısı : {sonuc['toplam_cumle']}")
    print(f"  [✓] Hesaplanan Kırılma Eşik Değeri    : {sonuc['esik']:.4f}")
    print(f"  [✓] Üretilen Semantik Parça Sayısı    : {sonuc['toplam_parca']}")

    # -------------------------------------------------------------
    # ADIM 2: Parça Detaylarını ve Sınırlarını Yazdır
    # -------------------------------------------------------------
    print("\n[2/3] Oluşturulan Semantik Parçaların (Semantic Chunks) Detayları:")
    print("=" * 95)
    print(f"{'PARÇA ID':<12} | {'CÜMLE':<8} | {'UZUNLUK':<10} | {'İÇERİK ÖZETİ'}")
    print("-" * 95)
    for p in sonuc["parcalar"]:
        print(f"{p['parca_id']:<12} | {p['cumle_sayisi']:<8} | {p['karakter_sayisi']:<10} | {p['metin'][:48]}...")
    print("=" * 95)

    # -------------------------------------------------------------
    # ADIM 3: Teşhis Panosu Çizimi
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli Semantic Chunking Teşhis Panosu Çiziliyor...")
    karsilastirma = RAGParcalamaKarsilastirici.benchmark_karsilastir()

    gorsellestirici = SemanticChunkingGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "semantic_chunking_paneli.png")
    gorsellestirici.pano_olustur(
        parcalama_sonucu=sonuc,
        karsilastirma=karsilastirma,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 105)
    print("[OK] Day 131: SEMANTIC CHUNKING & DYNAMIC RAG BAŞARIYLA TAMAMLANDI!")
    print("=" * 105)


if __name__ == "__main__":
    main()
