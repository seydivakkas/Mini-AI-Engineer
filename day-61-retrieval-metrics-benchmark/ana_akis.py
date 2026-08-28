"""
Day 61: Vektör Arama ve Bilgi Erişimi Değerlendirme (NDCG@k, MRR, MAP, Gecikme) Ana Yürütme Betiği.
"""

import os
import sys
import numpy as np

MEVCUT_DIZIN = os.path.abspath(os.path.dirname(__file__))
if MEVCUT_DIZIN not in sys.path:
    sys.path.insert(0, MEVCUT_DIZIN)

from src.metrik_motoru import RetrievalMetrikMotoru
from src.arama_degerlendirici import AramaDegerlendirici
from src.gorsellestirici import RetrievalGorsellestirici


def main():
    print("=" * 95, flush=True)
    print(">>> DAY 61: VEKTOR VE SEMANTIK ARAMA DEGERLENDIRMESI (NDCG@k, MRR, MAP, LATENCY & QPS)", flush=True)
    print("=" * 95, flush=True)

    # 1. Sentetik Arama Benchmark Senaryosu Üretimi
    num_queries = 500
    katalog_boyutu = 10_000
    print(f"\n[+] 1. Adım: {num_queries:,} Adet Test Sorgusu ve Dereceli İlgi Etiketleri Üretiliyor...", flush=True)
    senaryolar = AramaDegerlendirici.sentetik_arama_senaryosu_uret(
        num_queries=num_queries,
        katalog_boyutu=katalog_boyutu
    )
    print(f"    - Katalog Büyüklüğü : {katalog_boyutu:,} Doküman/Görsel")
    print(f"    - Test Senaryosu    : {len(senaryolar)} Sorgu")

    # 2. Çoklu Arama Stratejilerini Kıyaslama
    print("\n[+] 2. Adım: Hybrid RRF, Dense Vector, Lexical BM25 ve IVF-Flat Stratejileri Kıyaslanıyor...", flush=True)
    sonuclar = AramaDegerlendirici.calistir_karsilastirma(senaryolar)

    print("\n" + "=" * 95, flush=True)
    print(">>> 3. RETRIEVAL VE ARAMA KALİTE DEĞERLENDİRME TABLOSU", flush=True)
    print("=" * 95, flush=True)
    print(f"{'Arama Stratejisi':<32} | {'NDCG@10':<9} | {'MRR':<8} | {'MAP@10':<8} | {'Prec@10':<8} | {'p50 Lat':<9} | {'p99 Lat':<9} | {'QPS':<8}")
    print("-" * 110)

    for strat, m in sonuclar.items():
        lat = m["gecikme_istatistikleri"]
        print(
            f"{strat:<32} | "
            f"{m['ndcg@10']:>9.4f} | "
            f"{m['mrr']:>8.4f} | "
            f"{m['map@10']:>8.4f} | "
            f"{m['precision@10']:>8.4f} | "
            f"{lat['p50_ms']:>6.2f} ms | "
            f"{lat['p99_ms']:>6.2f} ms | "
            f"{lat['qps']:>8.0f}"
        )

    # 3. 6 Panelli Teşhis Panosunun Çizilmesi
    print("\n" + "=" * 95, flush=True)
    print(">>> 4. 6 PANELLİ RETRIEVAL METRİKLERİ TEŞHİS VE PERFORMANS PANOSU", flush=True)
    print("=" * 95, flush=True)

    hedef_pano = os.path.join(MEVCUT_DIZIN, "ciktilar", "retrieval_metrics_paneli.png")
    cikis_yolu = RetrievalGorsellestirici.panel_ciz(
        benchmark_sonuclari=sonuclar,
        num_queries=num_queries,
        hedef_path=hedef_pano
    )
    print(f"[+] 6 Panelli Teşhis Panosu Kaydedildi: {os.path.abspath(cikis_yolu)}", flush=True)
    print("=" * 95, flush=True)
    print("DAY 61: RETRIEVAL VE VEKTÖR ARAMA DEĞERLENDİRMESİ BAŞARIYLA TAMAMLANDI!", flush=True)
    print("=" * 95, flush=True)


if __name__ == "__main__":
    main()
