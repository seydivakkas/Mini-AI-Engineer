"""
Day 156: Chain of Verification (CoVe) Halüsinasyon Önleme & Fakt Kontrol Ana Akışı.
4 Aşamalı Doğrulama Boru Hattı: Taslak -> Soru Planı -> Bağımsız Doğrulama -> Sentez.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.cove_duzeltici_motor import CoVEDuzelticiMotor
from src.gorsellestirici import CoVEGorsellestirici


def main():
    print("=" * 105)
    print(">>> Day 156: Chain of Verification (CoVe) Fact-Checking & Self-Correction Engine (FAZ 8)")
    print("=" * 105)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # -------------------------------------------------------------
    # 1. KULLANICI SORGUSU
    # -------------------------------------------------------------
    soru = "Mehmet Akif Ersoy nerede doğdu, İstiklal Marşı'nı nerede yazdı ve marş hangi yıl TBMM'de kabul edildi?"
    print(f"\n[1/3] Kullanıcı Sorgusu: '{soru}'")

    # -------------------------------------------------------------
    # 2. 4 AŞAMALI CoVe BORU HATTI
    # -------------------------------------------------------------
    print("\n[2/3] 4 Aşamalı Chain of Verification (CoVe) Döngüsü Yürütülüyor...")
    cove_sonucu = CoVEDuzelticiMotor.calistir(soru)

    print("\n" + "-" * 85)
    print("AŞAMA 1: İLK TASLAK YANIT (Baseline - Halüsinasyon İçerir):")
    print(f"  '{cove_sonucu['ilk_taslak_yanit']}'")
    print("-" * 85)

    print("\nAŞAMA 2 & 3: PLANLANAN SORULAR VE BAĞIMSIZ FAKT KONTROLÜ:")
    for i, d in enumerate(cove_sonucu["dogrulama_raporu"], start=1):
        print(f"  [Soru {i}]: {d['soru']}")
        print(f"            Taslaktaki İddia  : '{d['taslak_iddia']}'")
        print(f"            Doğrulanan Gerçek : '{d['dogrulanmis_cevap']}' ({d['kanit']})")
        print(f"            Durum             : {d['durum']}")

    print("-" * 85)
    print("AŞAMA 4: DÜZELTİLMİŞ NİHAİ OLGUSAL YANIT (CoVe):")
    print(f"  '{cove_sonucu['duzeltilmis_yanit']}'")
    print("-" * 85)

    print(f"\n  • Toplam İddia Sayısı       : {cove_sonucu['toplam_iddia_sayisi']}")
    print(f"  • Düzeltilen Halüsinasyon   : {cove_sonucu['duzeltilen_iddia_sayisi']} adet")
    print(f"  • İlk Taslak Doğruluk Oranı : %{cove_sonucu['taslak_dogruluk_orani']:.1f}")
    print(f"  • CoVe Nihai Doğruluk Oranı : %{cove_sonucu['cove_dogruluk_orani']:.1f}")

    # -------------------------------------------------------------
    # 3. 6 PANELLİ TEŞHİS PANOSU ÜRETİMİ
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli CoVe Teşhis Panosu Üretiliyor...")
    gorsellestirici = CoVEGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "chain_of_verification_cove_paneli.png")
    gorsellestirici.pano_olustur(cove_sonucu, kayit_yolu=cikis_resmi)

    print("\n" + "=" * 105)
    print("✓ Day 156: CHAIN OF VERIFICATION (CoVe) BAŞARIYLA TAMAMLANDI!")
    print("=" * 105)


if __name__ == "__main__":
    main()
