"""
Day 230: Kendi Hatasını Düzelten (Self-Debugging) Kod Ajanı Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.hata_duzeltici_motoru import (
    TestCase,
    ExecutionFeedback,
    CodeExecutionHarness,
    SelfDebuggingAgent,
)
from src.debug_profilleyici import DebugProfilleyici
from src.gorsellestirici import DebugGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 230 (FAZ 12): KENDİ HATASINI DÜZELTEN (SELF-DEBUGGING) AJAN - TEST GERİ BİLDİRİMİ VE REFLEXION")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Test Senaryolarının Hazırlanması
    # -------------------------------------------------------------
    print("\n[1/4] Birim Test Senaryoları (Test Harness) Tanımlanıyor...")
    test_senaryolari = [
        TestCase(([1, 2, 3], [2, 3, 4]), [2, 3]),
        TestCase(([1, 2, 2, 3], [2, 2, 4]), [2]),
        TestCase(([10, 20], [30, 40]), []),
    ]

    for i, t in enumerate(test_senaryolari, 1):
        print(f"  • Test {i}: Girdi={t.girdi} -> Beklenen={t.beklenen}")

    # -------------------------------------------------------------
    # ADIM 2: Hatalı Kod ve Reflexion Adımlarının Tanımlanması
    # -------------------------------------------------------------
    print("\n[2/4] İlk Aday Kod (C_0) ve Reflexion Düşünüm Aşamaları...")

    hatali_kod = (
        "def kesisim(a, b):\n"
        "    return [x for x in a if x in b]\n"
    )
    reflexion1 = "HATA ANALİZİ: Liste üreteci mükerrer elemanları tekleştirmedi ([2, 2] döndü). set() küme kesişimi kullanmalıyım."

    duzeltilmis_kod = (
        "def kesisim(a, b):\n"
        "    return sorted(list(set(a) & set(b)))\n"
    )
    reflexion2 = "KOD GÜNCELLENDİ: set(a) & set(b) ile tekilleştirilmiş kesişim kümesi alındı ve liste olarak döndürüldü."

    # -------------------------------------------------------------
    # ADIM 3: Self-Debugging Döngüsünün Koşturulması
    # -------------------------------------------------------------
    print("\n[3/4] Otonom Self-Debugging Ajanı Kodu Onarıyor...")
    ajan = SelfDebuggingAgent(max_deneme=3)
    sonuc = ajan.onar_ve_coz(
        hedef_gorev="İki dizinin mükerrersiz kesişimini bulan fonksiyon",
        fonksiyon_adi="kesisim",
        aday_kod_adimlari=[
            (hatali_kod, reflexion1),
            (duzeltilmis_kod, reflexion2),
        ],
        test_senaryolari=test_senaryolari,
    )

    print(f"  • Çözüm Başarılı mı? : {sonuc['basarili_mi']}")
    print(f"  • Toplam Deneme Sayısı: {sonuc['toplam_deneme_sayisi']}")

    print("\n--- [Debug ve Reflexion Günlüğü] ---")
    for log in sonuc["debug_gunlugu"]:
        print("  " + log)

    print("\n--- [Nihai Onarılmış Kod] ---")
    print(sonuc["nihai_kod"])

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Self-Debugging Teşhis Panosu Oluşturuluyor...")
    profil_raporu = DebugProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "self_debugging_paneli.png")

    DebugGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Self-Debugging Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 230 (FAZ 12): KENDİ HATASINI DÜZELTEN (SELF-DEBUGGING) AJAN BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
