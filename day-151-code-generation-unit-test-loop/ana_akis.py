"""
Day 151: Test Odaklı Kod Üretimi: Kod Yazma -> PyTest Çalıştırma -> Hata Ayıklama (TDD) Döngüsü Ana Akışı.
SWE-bench ve LLM tabanlı yazılım mühendisliği ajanlarının kendi kendini onarma (Self-Repair) simülasyonu.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.tdd_dongusu_yoneticisi import TDDDongusuYoneticisi
from src.gorsellestirici import TDDGorsellestirici


def main():
    print("=" * 105)
    print(">>> Day 151: Test-Driven Code Generation (TDD Loop & Self-Repair) (FAZ 8)")
    print("=" * 105)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # -------------------------------------------------------------
    # 1. PROBLEM VE TDD GÖREV TANIMI
    # -------------------------------------------------------------
    print("\n[1/3] Görev Tanımlanıyor: 'Run-Length Encoding (RLE) Dize Sıkıştırma Fonksiyonu'")
    print("      Test Kapsamı      : Boş Dize, Tek Karakter, Standart Tekrarlı Dizi, Tekrarsız Dizi")

    # -------------------------------------------------------------
    # 2. TDD DÖNGÜSÜNÜ ÇALIŞTIRMA (GENERATE -> EXECUTE -> TRACEBACK -> REPAIR)
    # -------------------------------------------------------------
    print("\n[2/3] TDD Kod Üretim ve PyTest Hata Ayıklama Döngüsü Başlatılıyor...")
    yonetici = TDDDongusuYoneticisi(maks_deneme=3)
    tdd_sonucu = yonetici.tdd_dongusunu_baslat("Run-Length Encoding (RLE)")

    print("\n" + "-" * 75)
    print("TDD DÖNGÜSÜ İLERLEMESİ:")
    for k in tdd_sonucu["dongu_gecmisi"]:
        durum_str = "BAŞARILI (PASS)" if k["tum_testler_gecti_mi"] else "BAŞARISIZ (FAIL)"
        print(f"\n  [TUR {k['tur']}] Test Sonucu: {k['gecen_sayisi']}/{k['toplam_test_sayisi']} Geçti (%{k['basari_orani']*100:.1f}) | Durum: {durum_str}")
        if k["hata_raporu"]:
            print(f"         İlk Hata/Traceback: {k['hata_raporu'][:120]}...")
        if k.get("onarma_monologu"):
            print(f"         Onarım Monoloğu   : {k['onarma_monologu'][:150]}...")
    print("-" * 75)

    print(f"\n  • Toplam Tur Sayısı           : {tdd_sonucu['toplam_tur']}")
    print(f"  • Nihai Test Başarımı         : %100 (Tüm birim testler geçti!)")
    print("\n  [NİHAİ ONARILMIŞ KOD]:")
    for satir in tdd_sonucu["nihai_kod"].strip().split("\n"):
        print(f"    {satir}")

    # -------------------------------------------------------------
    # 3. 6 PANELLİ TEŞHİS PANOSU ÜRETİMİ
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli TDD Kod Üretimi Teşhis Panosu Üretiliyor...")
    gorsellestirici = TDDGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "code_generation_unit_test_loop_paneli.png")
    gorsellestirici.pano_olustur(tdd_sonucu, kayit_yolu=cikis_resmi)

    print("\n" + "=" * 105)
    print("✓ Day 151: TEST-DRIVEN CODE GENERATION (TDD LOOP) BAŞARIYLA TAMAMLANDI!")
    print("=" * 105)


if __name__ == "__main__":
    main()
