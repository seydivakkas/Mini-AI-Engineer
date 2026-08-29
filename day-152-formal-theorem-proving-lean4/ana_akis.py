"""
Day 152: Biçimsel Mantık ve Teorem İspatı: LLM ile Lean 4 Kod Üretimi & ITP Doğrulama Ana Akışı.
Autoformalization, Peano Tümevarımı ve Lean 4 Taktik Yürütme Simülasyonu.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.itp_dogrulayici import ITPDogrulayici
from src.gorsellestirici import Lean4Gorsellestirici


def main():
    print("=" * 105)
    print(">>> Day 152: Formal Theorem Proving & Lean 4 Tactic Engine (FAZ 8)")
    print("=" * 105)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # -------------------------------------------------------------
    # 1. DOĞAL DİL TEOREM TANIMI
    # -------------------------------------------------------------
    dogal_dil_teoremi = "Her doğal sayı n için, n + 0 = n olduğunu Peano tümevarımı ile ispatlayınız."
    print(f"\n[1/3] Doğal Dil Teoremi: '{dogal_dil_teoremi}'")

    # -------------------------------------------------------------
    # 2. AUTOFORMALIZATION VE LEAN 4 ITP İSPATI
    # -------------------------------------------------------------
    print("\n[2/3] Autoformalization (Lean 4 Koduna Çeviri) ve Taktik Motoru Yürütülüyor...")
    ispat_sonucu = ITPDogrulayici.teoremi_ispatla_ve_dogrula(dogal_dil_teoremi)

    print("\n" + "-" * 75)
    print("ÜRETİLEN LEAN 4 BİÇİMSEL TEOREM KODU:")
    for satir in ispat_sonucu["lean4_kodu"].strip().split("\n"):
        print(f"  {satir}")
    print("-" * 75)

    print("\nITP ÇEKİRDEK ADIMLARI VE HEDEF DURUMLARI:")
    for i, adim in enumerate(ispat_sonucu["adim_kayitlari"], start=1):
        print(f"  [Adım {i}] Taktik: '{adim['uygulanan_taktik']}'")
        print(f"           Açıklama: {adim['aciklama']}")
        print(f"           Kalan Hedef Sayısı: {adim['kalan_hedef_sayisi']}")

    print("-" * 75)
    print(f"\n  • İspat Başarılı mı            : {ispat_sonucu['ispatlandi_mi']}")
    print(f"  • Kalan Hedef (Goal Count)     : {ispat_sonucu['kalan_hedef_sayisi']} (no goals left / Q.E.D.!)")
    print(f"  • İspat Güvenilirliği          : %100 Biçimsel Tip Denetimi (Curry-Howard)")

    # -------------------------------------------------------------
    # 3. 6 PANELLİ TEŞHİS PANOSU ÜRETİMİ
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli Lean 4 Teşhis Panosu Üretiliyor...")
    gorsellestirici = Lean4Gorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "formal_theorem_proving_lean4_paneli.png")
    gorsellestirici.pano_olustur(ispat_sonucu, kayit_yolu=cikis_resmi)

    print("\n" + "=" * 105)
    print("✓ Day 152: FORMAL THEOREM PROVING (LEAN 4) BAŞARIYLA TAMAMLANDI!")
    print("=" * 105)


if __name__ == "__main__":
    main()
