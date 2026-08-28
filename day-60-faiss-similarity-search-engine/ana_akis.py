"""
Day 60: FAISS ile Milyonluk Vektör İndeksleme ve Benzerlik Arama Motoru Ana Yürütme Betiği.
"""

import os
import sys
import numpy as np

MEVCUT_DIZIN = os.path.abspath(os.path.dirname(__file__))
if MEVCUT_DIZIN not in sys.path:
    sys.path.insert(0, MEVCUT_DIZIN)

from src.indeks_motoru import FAISSIndeksMotoru, IndeksTuru
from src.vektor_benchmark import VektorBenchmarkRunner
from src.gorsellestirici import FAISSGorsellestirici


def main():
    print("=" * 85, flush=True)
    print(">>> DAY 60: FAISS ILE MILYONLUK VEKTOR INDEKSLEME & BENZERLIK ARAMA MOTORU", flush=True)
    print("=" * 85, flush=True)

    # 1. Sentetik Vektör Seti ve Sorgu Üretimi
    num_vectors = 50_000
    num_queries = 1_000
    dim = 128
    top_k = 10

    print(f"\n[+] 1. Adım: {num_vectors:,} Adet {dim}-D Vektör ve {num_queries:,} Adet Sorgu Üretiliyor...", flush=True)
    vektorler, sorgular = VektorBenchmarkRunner.sentetik_vektor_olustur(
        num_vectors=num_vectors,
        num_queries=num_queries,
        dim=dim,
        num_clusters=25
    )
    print(f"    - Veritabanı Vektör Matrisi : {vektorler.shape} ({vektorler.nbytes / (1024*1024):.1f} MB)")
    print(f"    - Sorgu Vektör Matrisi      : {sorgular.shape} ({sorgular.nbytes / (1024*1024):.2f} MB)")

    # 2. Uçtan Uca İndeks Kıyaslama Çalıştırma
    print("\n[+] 2. Adım: IndexFlatIP, IndexIVFFlat ve IndexHNSWFlat İndeksleri Kıyaslanıyor...", flush=True)
    sonuclar = VektorBenchmarkRunner.calistir_karsilastirma(
        vektorler=vektorler,
        sorgular=sorgular,
        top_k=top_k
    )

    print("\n" + "=" * 85, flush=True)
    print(">>> 3. FAISS BENCHMARK VE ARAMA PERFORMANS TABLOSU", flush=True)
    print("=" * 85, flush=True)
    print(f"{'İndeks Yapılandırması':<28} | {'Recall@10':<11} | {'QPS':<11} | {'Gecikme (ms)':<14} | {'İnşa Süresi':<11} | {'Bellek (MB)':<10}")
    print("-" * 98)

    for ad, d in sonuclar.items():
        print(
            f"{ad:<28} | "
            f"%{d['recall']:>8.2f} | "
            f"{d['qps']:>9,.0f} | "
            f"{d['tekil_sorgu_ms']:>10.4f} ms | "
            f"{d['build_suresi_s']:>9.3f} s | "
            f"{d['bellek_tahmini_mb']:>8.1f} MB"
        )

    # 3. İndeks Serileştirme ve Geri Yükleme Testi
    print("\n[+] 4. Adım: HNSW İndeksinin Diske Kaydı ve Geri Yükleme Doğrulaması...", flush=True)
    kayit_yolu = os.path.join(MEVCUT_DIZIN, "checkpoints", "hnsw_indeksi.faiss")
    os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

    test_motor = FAISSIndeksMotoru(dim=dim, indeks_turu=IndeksTuru.HNSW_FLAT, M=32)
    test_motor.egit_ve_ekle(vektorler[:1000])
    test_motor.indeksi_kaydet(kayit_yolu)

    yuklenen_motor = FAISSIndeksMotoru.indeksi_yukle(kayit_yolu)
    assert yuklenen_motor.toplam_vektor == 1000
    print(f"    - İndeks Başarıyla Kaydedildi ve Geri Yüklendi ({yuklenen_motor.toplam_vektor} Vektör)")

    # 4. 6 Panelli Teşhis Panosunun Çizilmesi
    print("\n" + "=" * 85, flush=True)
    print(">>> 5. 6 PANELLİ FAISS TEŞHİS VE PERFORMANS PANOSUNUN ÜRETİLMESİ", flush=True)
    print("=" * 85, flush=True)

    hedef_pano = os.path.join(MEVCUT_DIZIN, "ciktilar", "faiss_benchmark_paneli.png")
    cikis_yolu = FAISSGorsellestirici.panel_ciz(
        benchmark_sonuclari=sonuclar,
        num_vectors=num_vectors,
        dim=dim,
        hedef_path=hedef_pano
    )
    print(f"[+] 6 Panelli FAISS Teşhis Panosu Kaydedildi: {os.path.abspath(cikis_yolu)}", flush=True)
    print("=" * 85, flush=True)
    print("DAY 60: FAISS VEKTÖR İNDEKSLEME VE BENZERLİK ARAMA BAŞARIYLA TAMAMLANDI!", flush=True)
    print("=" * 85, flush=True)


if __name__ == "__main__":
    main()
