"""
Day 228: SWE-Bench Otonom Kodlayıcı Ajan Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.swe_kodlayici_motoru import (
    GitHubIssue,
    CodebaseNavigator,
    SurgicalPatcher,
    AutonomousSWEAgent,
)
from src.swe_profilleyici import SWEProfilleyici
from src.gorsellestirici import SWEGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 228 (FAZ 12): SWE-BENCH OTONOM YAZILIM MÜHENDİSİ - CERRAHİ YAMA VE REPO ONARIMI")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: GitHub Issue ve Hata Bildirimi Yükleme
    # -------------------------------------------------------------
    print("\n[1/4] Gerçek Dünya GitHub Sorun Bildirimi (Issue) Yükleniyor...")
    issue = GitHubIssue(
        issue_id=892,
        title="ZeroDivisionError in calculate_variance when samples list is empty",
        description="Varyans hesaplama fonksiyonu boş liste gönderildiğinde ZeroDivisionError veriyor.",
        stack_trace="ZeroDivisionError: division by zero at stats.py:L5 in calculate_variance",
        target_file="src/stats.py",
    )

    orijinal_kod = (
        "def calculate_variance(data):\n"
        "    n = len(data)\n"
        "    mean = sum(data) / n\n"
        "    return sum((x - mean) ** 2 for x in data) / n\n"
    )

    print(f"  • Issue ID     : #{issue.issue_id}")
    print(f"  • Başlık       : {issue.title}")
    print(f"  • Hedef Dosya  : {issue.target_file}")
    print(f"  • Stack Trace  : {issue.stack_trace}")

    # -------------------------------------------------------------
    # ADIM 2: Otonom Kodlayıcı ile Hata Onarımı ve Cerrahi Yama
    # -------------------------------------------------------------
    print("\n[2/4] Otonom SWE Ajanı Sorunu Çözüyor ve Cerrahi Yama Uyguluyor...")

    hedef_kesit = "    n = len(data)\n    mean = sum(data) / n\n"
    duzeltilmis_kesit = (
        "    n = len(data)\n"
        "    if n == 0:\n"
        "        return 0.0\n"
        "    mean = sum(data) / n\n"
    )

    ajan = AutonomousSWEAgent()
    sonuc = ajan.sorunu_coz_ve_yamala(
        issue=issue,
        orijinal_dosya_icerigi=orijinal_kod,
        hedef_kod_kesiti=hedef_kesit,
        duzeltilmis_kod_kesiti=duzeltilmis_kesit,
    )

    print(f"  • Çözüm Başarılı mı? : {sonuc['cozum_basarili_mi']}")
    print("\n--- [Oluşturulan Unified Git Diff Yaması] ---")
    print(sonuc["unified_diff"])

    print("\n--- [Ajan İşlem Günlüğü] ---")
    for adim in sonuc["islem_adimlari"]:
        print("  " + adim)

    # -------------------------------------------------------------
    # ADIM 3: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[3/4] 6 Panelli SWE-Bench Teşhis Panosu Oluşturuluyor...")
    profil_raporu = SWEProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "swe_kodlayici_paneli.png")

    SWEGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ SWE-Bench Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 228 (FAZ 12): SWE-BENCH OTONOM YAZILIM MÜHENDİSİ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
