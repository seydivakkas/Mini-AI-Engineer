"""
Day 157: Soru Zorluğuna Göre Dinamik Token Bütçesi ve Hesaplama Tahsisi (Dynamic Compute Allocation) Ana Akışı.
Easy vs Hard Routing, Token & Maliyet Tasarrufu ve Çıkarım Optimizasyonu.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.cikarim_simulasyonu import CikarimSimulasyonu
from src.gorsellestirici import DinamikComputeGorsellestirici


def main():
    print("=" * 105)
    print(">>> Day 157: Dynamic Compute & Token Budget Allocation Engine (FAZ 8)")
    print("=" * 105)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # -------------------------------------------------------------
    # 1. TEST SORU SETİ (KOLAY / ORTA / ZOR)
    # -------------------------------------------------------------
    sorular = [
        "Türkiye'nin başkenti neresidir?",
        "Python'da bir liste nasıl ters çevrilir?",
        "Ahmet 250 TL'lik ceketi %20 indirim ve %18 KDV ile kaça alır? Adım adım hesapla.",
        "Bir fırın sabah 120, öğlen 80 ekmek satıp tanesini 5 TL'den verdiğinde toplam karı hesapla.",
        "AIME 2024 Problem: Tüm n doğal sayıları için P(n) polinomunun asal köklerini teorem ile ispatlayınız.",
        "Dinamik programlama ile Travelling Salesman Problem (TSP) için optimal algoritma tasarlayıp karmaşıklığını optimize ediniz.",
    ]

    print(f"\n[1/3] {len(sorular)} Adet Farklı Zorlukta Soru İşleniyor...")

    # -------------------------------------------------------------
    # 2. DİNAMİK COMPUTE SİMÜLASYONU
    # -------------------------------------------------------------
    print("\n[2/3] Soru Zorlukları Analiz Ediliyor ve Dinamik Bütçeler Tahsis Ediliyor...")
    sonuc = CikarimSimulasyonu.calistir(sorular)

    print("\n" + "-" * 95)
    print(f"{'#':<3} | {'Kategori':<8} | {'Zorluk':<6} | {'Tahsis Token':<12} | {'Süre (ms)':<10} | {'Maliyet (TL)':<12} | {'Çıkarım Modu'}")
    print("-" * 95)
    for i, s in enumerate(sonuc["soru_sonuclari"], start=1):
        print(
            f"{i:<3} | {s['kategori']:<8} | {s['zorluk_skoru']:<6.2f} | {s['tahsis_edilen_token_butcesi']:<12} | "
            f"{s['tahmini_gecikme_ms']:<10.1f} | {s['tahmini_maliyet_tl']:<12.3f} | {s['cikarim_modu']}"
        )
    print("-" * 95)

    print("\nTOPLAM KIYASLAMA VE TASARRUF RAPORU:")
    print(f"  • Toplam Harcanan Token  : Sabit = {sonuc['toplam_sabit_token']:,} tok  vs  Dinamik = {sonuc['toplam_dinamik_token']:,} tok  (%%%s Tasarruf)" % sonuc['token_tasarrufu_yuzde'])
    print(f"  • Toplam Çıkarım Maliyeti: Sabit = {sonuc['toplam_sabit_maliyet_tl']:.3f} TL  vs  Dinamik = {sonuc['toplam_dinamik_maliyet_tl']:.3f} TL  (%%%s Tasarruf)" % sonuc['maliyet_tasarrufu_yuzde'])
    print(f"  • Toplam Çıkarım Süresi  : Sabit = {sonuc['toplam_sabit_sure_ms']/1000:.2f} sn  vs  Dinamik = {sonuc['toplam_dinamik_sure_ms']/1000:.2f} sn  (%sx Kat Hızlanma)" % sonuc['hizlanma_orani'])

    # -------------------------------------------------------------
    # 3. 6 PANELLİ TEŞHİS PANOSU ÜRETİMİ
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli Dinamik Compute Teşhis Panosu Üretiliyor...")
    gorsellestirici = DinamikComputeGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "dynamic_compute_allocation_paneli.png")
    gorsellestirici.pano_olustur(sonuc, kayit_yolu=cikis_resmi)

    print("\n" + "=" * 105)
    print("✓ Day 157: DYNAMIC COMPUTE ALLOCATION BAŞARIYLA TAMAMLANDI!")
    print("=" * 105)


if __name__ == "__main__":
    main()
