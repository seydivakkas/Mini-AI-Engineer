"""
Day 154: GSM8K & MATH Benchmark: Program-Aided Language Models (PAL / PoT) Ana Akışı.
Çok Adımlı Sözel Problemlerin Python Kodu Olarak Yürütülmesi ve Doğrulanması.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.aritmetik_ayristirici import AritmetikAyristirici
from src.pal_kod_ureticisi import PALKodUreticisi
from src.gsm8k_yurutucu import GSM8KYurutucu
from src.gorsellestirici import GSM8KGorsellestirici


def main():
    print("=" * 105)
    print(">>> Day 154: Multi-Step Arithmetic Reasoning & PAL Engine (GSM8K Benchmark) (FAZ 8)")
    print("=" * 105)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # -------------------------------------------------------------
    # 1. 4 FARKLI GSM8K SÖZEL MATEMATİK PROBLEMİ
    # -------------------------------------------------------------
    problemler = [
        {
            "id": "p1_elma",
            "ad": "Elma Dağıtımı",
            "metin": "Ayşe'nin 15 elması vardı. 3 arkadaşının her birine 2'şer elma verdi. Kalan elmaların yarısını da annesine verdi. Ayşe'nin elinde kaç elma kaldı?",
            "beklenen": 4.5,
            "raw_cot_tahmini": 6.0, # Zihinsel CoT genelde 15 - 6 = 9'un yarısını yanlış alabilir
        },
        {
            "id": "p2_firin",
            "ad": "Fırın Satışı",
            "metin": "Bir fırın sabah 120 ekmek, öğlen 80 ekmek üretti. Üretilen ekmeklerin 150 tanesini tanesi 5 TL'den sattı. Fırın ekmek satışından kaç TL gelir elde etti?",
            "beklenen": 750.0,
            "raw_cot_tahmini": 750.0,
        },
        {
            "id": "p3_hiz",
            "ad": "Yol & Mesafe",
            "metin": "Bir araba saatte 60 km hızla 3 saat gitti. Ardından hızını saatte 80 km'ye çıkarıp 2 saat daha gitti. Araba toplam kaç km yol almıştır?",
            "beklenen": 340.0,
            "raw_cot_tahmini": 340.0,
        },
        {
            "id": "p4_vergi",
            "ad": "İndirim & KDV",
            "metin": "Etiket fiyatı 250 TL olan bir cekete önce %20 indirim yapıldı. İndirimli fiyat üzerinden %18 KDV eklendi. Müşteri kasada kaç TL ödedi?",
            "beklenen": 236.0,
            "raw_cot_tahmini": 245.0, # Zihinsel CoT %18 KDV'yi kaba hesaplayabilir
        },
    ]

    print(f"\n[1/3] {len(problemler)} Adet GSM8K Matematik Problemi Yükleniyor...")

    # -------------------------------------------------------------
    # 2. PAL KOD ÜRETİMİ VE İZOLE ÇALIŞTIRMA
    # -------------------------------------------------------------
    print("\n[2/3] Program of Thoughts (PoT) ile Python Kodu Üretiliyor ve Çalıştırılıyor...")
    ureticisi = PALKodUreticisi()
    karsilastirma_listesi = []

    print("-" * 100)
    for p in problemler:
        ayristirma = AritmetikAyristirici.ayristir(p["metin"])
        kod_bilgisi = ureticisi.kod_uret(p["id"], p["metin"])

        karsilastirma = GSM8KYurutucu.cozum_karsilastir(
            problem_adi=p["ad"],
            problem_metni=p["metin"],
            pal_kodu=kod_bilgisi["python_kodu"],
            beklenen_sonuc=p["beklenen"],
            raw_cot_tahmini=p["raw_cot_tahmini"],
        )
        karsilastirma_listesi.append(karsilastirma)

        print(f"\n[PROBLEM: {p['ad']}]: '{p['metin']}'")
        print(f"  • Tespit Edilen Sayılar : {ayristirma['tespit_edilen_sayilar']}")
        print(f"  • Beklenen Sonuç        : {p['beklenen']}")
        print(f"  • PAL (Python) Çıktısı  : {karsilastirma['pal_sonucu']} (Doğru mu: {karsilastirma['pal_dogru_mu']}) [Süre: {karsilastirma['calisma_suresi_ms']:.2f} ms]")
        print(f"  • Raw CoT (Mental Math) : {karsilastirma['raw_cot_tahmini']} (Doğru mu: {karsilastirma['raw_cot_dogru_mu']})")

    print("-" * 100)

    # -------------------------------------------------------------
    # 3. 6 PANELLİ TEŞHİS PANOSU ÜRETİMİ
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli GSM8K PAL Teşhis Panosu Üretiliyor...")
    gorsellestirici = GSM8KGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "multi_step_arithmetic_gsm8k_paneli.png")
    gorsellestirici.pano_olustur(karsilastirma_listesi, kayit_yolu=cikis_resmi)

    print("\n" + "=" * 105)
    print("✓ Day 154: MULTI-STEP ARITHMETIC (PAL / PoT) BAŞARIYLA TAMAMLANDI!")
    print("=" * 105)


if __name__ == "__main__":
    main()
