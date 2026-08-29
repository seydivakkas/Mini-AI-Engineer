"""
Day 145: Outcome (ORM) vs Process Reward Models (PRM): Adım Adım Mantıksal Doğruluk Puanlama Ana Akışı.
Lightman et al. (OpenAI PRM800K) step-level supervision ve Best-of-N yeniden sıralama simülasyonu.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.best_of_n_sirayici import BestOfNSirayici
from src.gorsellestirici import PRMvsORMGorsellestirici


def main():
    print("=" * 105)
    print(">>> Day 145: Outcome (ORM) vs Process Reward Models (PRM) (FAZ 8)")
    print("=" * 105)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # -------------------------------------------------------------
    # 1. TEST PROBLEMİ VE ADAY DÜŞÜNCE YOLLARI
    # -------------------------------------------------------------
    print("\n[1/3] Aday Akıl Yürütme Yolları Hazırlanıyor (CRT: Beyzbol Sopası ve Top Problemi)...")
    aday_yollar = [
        {
            "yol_id": "Yol #1 (Kusursuz Cebirsel İspat)",
            "adimlar": [
                "Sopa + Top = 1.10",
                "Sopa = Top + 1.00",
                "2 * Top = 0.10",
                "Top = 0.05",
            ],
            "nihai_cevap": "0.05",
        },
        {
            "yol_id": "Yol #2 (Şanslı Tahmin / Hatalı Ara İşlem)",
            "adimlar": [
                "Sopa + Top = 1.10",
                "1.10 - 1.00 = 0.10",  # Mantık hatası!
                "Top = 0.05",          # Ama sonuca tesadüfen doğru ulaşıyor
            ],
            "nihai_cevap": "0.05",
        },
        {
            "yol_id": "Yol #3 (Sezgisel Yanılgı)",
            "adimlar": [
                "Sopa + Top = 1.10",
                "1.10 - 1.00 = 0.10",
                "Top = 0.10",
            ],
            "nihai_cevap": "0.10",
        },
        {
            "yol_id": "Yol #4 (Rastgele Bölme Hatası)",
            "adimlar": [
                "1.10 / 2 = 0.55",
                "Top = 0.55",
            ],
            "nihai_cevap": "0.55",
        },
    ]

    # -------------------------------------------------------------
    # 2. BEST-OF-N ORM vs PRM SIRALAMASI
    # -------------------------------------------------------------
    print("\n[2/3] ORM ve PRM Modelleri ile N Aday Puanlanıyor ve Sıralanıyor...")
    sirayici = BestOfNSirayici(dogru_cevap="0.05")
    sonuc = sirayici.karsilastir_ve_sirala(aday_yollar)

    print("\n" + "-" * 95)
    print(f"{'ADAY YOL ADI':<40} | {'NİHAİ':<6} | {'ORM SKORU':<10} | {'PRM SKORU':<10} | {'DURUM'}")
    print("-" * 95)
    for y in sonuc["orm_sirali_adaylar"]:
        prm_bilgi = next(p for p in sonuc["prm_sirali_adaylar"] if p["yol_id"] == y["yol_id"])
        durum = "KUSURSUZ [GEÇERLİ]" if prm_bilgi["gecerli_yol_mu"] else "ŞANSLI TAHMİN / ELENDİ"
        if not y["nihai_cevap_dogru_mu"]:
            durum = "YANLIŞ YANIT / ELENDİ"
        print(f"{y['yol_id']:<40} | {y['nihai_cevap']:<6} | {y['orm_puani']:<10.2f} | {prm_bilgi['prm_carpim_puani']:<10.4f} | {durum}")
    print("-" * 95)

    print(f"\n  • ORM Seçimi (En Yüksek Sonuç Puanı) : {sonuc['orm_secimi']['yol_id']} (ORM Skoru: {sonuc['orm_secimi']['orm_puani']})")
    print(f"  • PRM Seçimi (En Yüksek Süreç Puanı)  : {sonuc['prm_secimi']['yol_id']} (PRM Skoru: {sonuc['prm_secimi']['prm_carpim_puani']})")
    print(f"  • Yakalanan Şanslı Tahmin Sayısı     : {sonuc['sansli_tahmin_sayisi']} adet yol PRM tarafından engellendi!")

    # -------------------------------------------------------------
    # 3. 6 PANELLİ TEŞHİS PANOSU ÜRETİMİ
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli PRM vs ORM Teşhis Panosu Üretiliyor...")
    gorsellestirici = PRMvsORMGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "process_reward_models_prm_paneli.png")
    gorsellestirici.pano_olustur(karsilastirma_sonucu=sonuc, kayit_yolu=cikis_resmi)

    print("\n" + "=" * 105)
    print("✓ Day 145: PROCESS REWARD MODELS (PRM vs ORM) BAŞARIYLA TAMAMLANDI!")
    print("=" * 105)


if __name__ == "__main__":
    main()
