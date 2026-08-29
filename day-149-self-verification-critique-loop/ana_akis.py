"""
Day 149: Kendi Kendine Doğrulama (Self-Verification) ve İkili Eleştiri Döngüsü (Actor-Critic) Ana Akışı.
Ters sağlama (Back-Substitution) ve kesinlik kalibrasyonu simülasyonu.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.dogrulama_dongusu_yoneticisi import DogrulamaDongusuYoneticisi
from src.gorsellestirici import SelfVerificationGorsellestirici


def main():
    print("=" * 105)
    print(">>> Day 149: Self-Verification & Actor-Critic Critique Loop (FAZ 8)")
    print("=" * 105)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # -------------------------------------------------------------
    # 1. PROBLEM TANIMI (Modüler Denklem & Ters Sağlama)
    # -------------------------------------------------------------
    print("\n[1/3] Problem Tanımlanıyor: '3x + 7 = 2 (mod 5) denklemini sağlayan x değerini bulunuz.'")
    problem = {"tur": "moduler_aritmetik", "denklem": "3x + 7 = 2 (mod 5)"}

    # -------------------------------------------------------------
    # 2. ACTOR-CRITIC SELF-VERIFICATION DÖNGÜSÜ
    # -------------------------------------------------------------
    print("\n[2/3] Actor-Critic Doğrulama ve Eleştiri Döngüsü Başlatılıyor...")
    yonetici = DogrulamaDongusuYoneticisi(maks_dongu=3)
    sonuc = yonetici.calistir(problem)

    print("\n" + "-" * 75)
    print("DÖNGÜ GEÇMİŞİ VE ELEŞTİRİ KAYITLARI:")
    for k in sonuc["dongu_kayitlari"]:
        durum_str = "ONAYLANDI (OK)" if k["dogrulandi_mi"] else "REDDEDİLDİ (X)"
        print(f"\n  [TUR {k['tur']}] Aday Çözüm: x = {k['aday_x']} | Güven: %{k['guven_skoru']*100:.1f} | Durum: {durum_str}")
        print(f"         Eleştiri Notu: {k['elestiri_notu']}")
        if k["hata_mesaji"]:
            print(f"         Hata Mesajı  : {k['hata_mesaji']}")
    print("-" * 75)

    print(f"\n  • Toplam Tur Sayısı              : {sonuc['toplam_tur_sayisi']}")
    print(f"  • Doğrulama Başarılı mı          : {sonuc['basarili_dogrulandi_mi']}")
    print(f"  • Nihai Doğrulanmış Çözüm        : x = {sonuc['nihai_cozum']}")

    # -------------------------------------------------------------
    # 3. 6 PANELLİ TEŞHİS PANOSU ÜRETİMİ
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli Self-Verification Teşhis Panosu Üretiliyor...")
    gorsellestirici = SelfVerificationGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "self_verification_critique_loop_paneli.png")
    gorsellestirici.pano_olustur(sonuc, kayit_yolu=cikis_resmi)

    print("\n" + "=" * 105)
    print("✓ Day 149: SELF-VERIFICATION & CRITIQUE LOOP BAŞARIYLA TAMAMLANDI!")
    print("=" * 105)


if __name__ == "__main__":
    main()
