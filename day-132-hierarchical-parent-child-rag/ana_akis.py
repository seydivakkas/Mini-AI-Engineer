"""
Day 132: Hiyerarşik RAG (Parent-Child / Small-to-Big Retrieval) Ana Akışı.
Küçük çocuk parçalarla hassas vektör araması, büyük ebeveyn parçalarla zengin bağlam genişletme.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.parent_child_getirici import ParentChildRAGGetirici
from src.gorsellestirici import ParentChildGorsellestirici


def main():
    print("=" * 105)
    print(">>> Day 132: Hierarchical Parent-Child RAG (Small-to-Big Retrieval Architecture)")
    print("=" * 105)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # Kapsamlı mühendislik el kitabı belgesi
    dokuman = (
        "Bölüm 1: Transformer Dikkat Mekanizmaları. "
        "Self-Attention katmanı sorgu (Query), anahtar (Key) ve değer (Value) matrisleri üzerinden çalışır. "
        "Ölçeklenmiş nokta çarpımı (Scaled Dot-Product) dikkat skorlarını hesaplayarak token ilişkilerini modeller. "
        "Çok başlı dikkat (Multi-Head Attention), modelin farklı temsil alt uzaylarına aynı anda odaklanmasını sağlar. "
        "Bölüm 2: Vektör Veritabanları ve IVF-PQ İndeksleme. "
        "Ters Çevrilmiş Dosya (Inverted File - IVF) vektör uzayını Voronoi hücrelerine bölerek arama alanını daraltır. "
        "Ürün Kuantizasyonu (Product Quantization - PQ) ise yüksek boyutlu vektörleri bayt dizilerine sıkıştırarak bellek tasarrufu sağlar. "
        "IVF-PQ kombinasyonu milyarlarca vektörde milisaniyelik yaklaşık en yakın komşu (ANN) sorgusu sunar. "
        "Bölüm 3: Dağıtık GPU Tensör Paralelizmi ve Megatron-LM. "
        "Tensör paralelizmi matris çarpımlarını sütun ve satır bazında GPU'lar arasında böler. "
        "İletişim yükünü azaltmak için All-Reduce ve All-Gather CUDA çekirdekleri optimize edilir. "
        "Boru hattı paralelizmi (Pipeline Parallelism) ile birleştiğinde yüzlerce milyar parametreli modeller eğitilebilir."
    )

    getirici = ParentChildRAGGetirici(ebeveyn_boyutu=450, cocuk_boyutu=140)

    print("\n[1/3] Belge Hiyerarşik Ağaca Dönüştürülüyor (Parent-Child Indexing)...")
    istatistikler = getirici.belge_yukle(dokuman)
    print(f"  [✓] Üretilen Ebeveyn Parça Sayısı (DocStore) : {istatistikler['toplam_ebeveyn']}")
    print(f"  [✓] Üretilen Çocuk Parça Sayısı (Vektör İndeks): {istatistikler['toplam_cocuk']}")

    # -------------------------------------------------------------
    # ADIM 2: Small-to-Big Sorgulama ve Bağlam Genişletme
    # -------------------------------------------------------------
    sorgu = "IVF-PQ vektör indeksleme ve kuantizasyon nasıl çalışır?"
    print(f"\n[2/3] Sorgu İcra Ediliyor: '{sorgu}'\n")

    sonuc = getirici.sorgula_ve_genislet(sorgu, cocuk_top_k=3)

    print("--- [A] Vektör Aramasında Eşleşen Çocuk Parçalar (Small Vectors) ---")
    for c in sonuc["eslesen_cocuklar"]:
        print(f"  • {c['child_id']} (Skor: %{c['skor']*100:.1f}) -> Parent: {c['parent_id']}")
        print(f"    Metin: \"{c['metin'][:60]}...\"")

    print("\n--- [B] DocStore'dan Genişletilen Ebeveyn Parçalar (Big Context Expansion) ---")
    for p in sonuc["secilen_ebeveynler"]:
        print(f"  ✓ {p['parent_id']} ({p['karakter']} karakter) -> {p['metin_ozeti']}")

    print(f"\n  [⚡] Arama ve Bağlam Genişletme Süresi: {sonuc['arama_suresi_ms']:.2f} ms")

    # -------------------------------------------------------------
    # ADIM 3: Teşhis Panosu Çizimi
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli Parent-Child RAG Teşhis Panosu Çiziliyor...")
    karsilastirma = getirici.benchmark_karsilastir()

    gorsellestirici = ParentChildGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "hierarchical_parent_child_paneli.png")
    gorsellestirici.pano_olustur(
        arama_sonucu=sonuc,
        karsilastirma=karsilastirma,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 105)
    print("[OK] Day 132: HIERARCHICAL PARENT-CHILD RAG BAŞARIYLA TAMAMLANDI!")
    print("=" * 105)


if __name__ == "__main__":
    main()
