"""
Day 237: Ajan Öz-Yansıtma (Reflection) ve Öz-Değerlendirme Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.refleksiyon_ajani_motoru import (
    EvaluationScore,
    ReflectionCritic,
    SelfRefiningAgent,
)
from src.refleksiyon_profilleyici import RefleksiyonProfilleyici
from src.gorsellestirici import RefleksiyonGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 237 (FAZ 12): AJAN ÖZ-YANSITMA (SELF-REFLECTION) - RUBRİK DENETİMİ VE YİNELEMELİ İYİLEŞTİRME")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Öz-Yansıtma Ajanının Başlatılması
    # -------------------------------------------------------------
    print("\n[1/4] Self-Refine Öz-Yansıtma Ajanı ve Rubrik Denetçisi Başlatılıyor...")
    agent = SelfRefiningAgent(esik_puani=90.0, maks_iterasyon=3)
    print("  ✓ Kalite ve Güvenlik Onay Eşiği: >= 90.0 Puan")

    # -------------------------------------------------------------
    # ADIM 2: Kod Taslakları ve Görev Tanımı
    # -------------------------------------------------------------
    gorev = "Güvenli Kullanıcı Şifre Doğrulama Fonksiyonu (verify_password)"
    print(f"\n[2/4] Hedef Görev: '{gorev}'")

    taslak_1 = "def verify_password(plain_pwd, stored_pwd):\n    return plain_pwd == stored_pwd"
    taslak_2 = "import hashlib\ndef verify_password(plain_pwd: str, stored_hash: str) -> bool:\n    return hashlib.sha256(plain_pwd.encode()).hexdigest() == stored_hash"
    taslak_3 = "import bcrypt\ndef verify_password(plain_pwd: str, hashed_pwd: str) -> bool:\n    try:\n        return bcrypt.checkpw(plain_pwd.encode('utf-8'), hashed_pwd.encode('utf-8'))\n    except Exception:\n        return False"

    # -------------------------------------------------------------
    # ADIM 3: Yinelemeli Öz-Yansıtma ve Eleştiri Döngüsü
    # -------------------------------------------------------------
    print("\n[3/4] Yinelemeli İyileştirme Döngüsü Koşturuluyor...")
    sonuc = agent.iyilestir_ve_tamamla([taslak_1, taslak_2, taslak_3])

    for adim in sonuc["gecmis"]:
        print(f"\n  --- [İterasyon #{adim['iterasyon']}] ---")
        print(f"  • Kod:\n{adim['taslak']}")
        print(f"  • Rubrik Skorları  : Doğruluk={adim['dogruluk']}/40, Güvenlik={adim['guvenlik']}/40, Tamlık={adim['tamlik']}/20")
        print(f"  • Toplam Skor      : {adim['skor']}/100 -> Onaylandı mı? : {adim['onaylandi']}")
        print(f"  • Eleştiri/Tavsiye : {adim['elestiri']}")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Öz-Yansıtma Teşhis Panosu Oluşturuluyor...")
    profil_raporu = RefleksiyonProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "refleksiyon_ajani_paneli.png")

    RefleksiyonGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Öz-Yansıtma Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 237 (FAZ 12): AJAN ÖZ-YANSITMA VE ELEŞTİRİ DÖNGÜSÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
