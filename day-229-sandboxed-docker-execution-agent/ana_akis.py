"""
Day 229: Güvenli Docker Sandbox Ajanı Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.sandbox_motoru import (
    SandboxConfig,
    ExecutionResult,
    DockerSandboxAgent,
)
from src.sandbox_profilleyici import SandboxProfilleyici
from src.gorsellestirici import SandboxGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 229 (FAZ 12): GÜVENLİ DOCKER SANDBOX AJANI - İZOLE KOD ÇALIŞTIRMA VE GÜVENLİK SINIRLARI")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Sandbox Ajanının Kurulumu
    # -------------------------------------------------------------
    print("\n[1/4] Güvenli Docker Sandbox Ajanı Başlatılıyor...")
    config = SandboxConfig(zaman_asimi_s=5.0, bellek_limiti_mb=512)
    ajan = DockerSandboxAgent(config=config)
    print("  ✓ Sandbox Güvenlik Duvarı Aktif (Bellek: 512MB, Zaman Aşımı: 5s, İzolasyon: Aktif).")

    # -------------------------------------------------------------
    # ADIM 2: Güvenli Veri İşleme Kodunun İzole İcrası
    # -------------------------------------------------------------
    print("\n[2/4] Güvenli Ajan Kodunun Sandbox Ortamında Çalıştırılması...")
    guvenli_kod = (
        "puanlar = [85, 92, 78, 95, 88]\n"
        "en_yuksek = max(puanlar)\n"
        "ortalama = sum(puanlar) / len(puanlar)\n"
        "print(f'Sınıf Ortalaması: {ortalama:.2f}, En Yüksek Not: {en_yuksek}')\n"
    )

    sonuc1 = ajan.kodu_izole_calistir(guvenli_kod)
    print(sonuc1.format_metni())

    # -------------------------------------------------------------
    # ADIM 3: Zararlı Sistem Çağrısının Engellenmesi
    # -------------------------------------------------------------
    print("\n[3/4] Ana Sisteme Zarar Vermeye Çalışan Tehlikeli Kod Test Ediliyor...")
    zararli_kod = (
        "import os\n"
        "os.system('del /f /q C:\\Windows\\System32')\n"
    )

    sonuc2 = ajan.kodu_izole_calistir(zararli_kod)
    print(sonuc2.format_metni())

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Docker Sandbox Teşhis Panosu Oluşturuluyor...")
    profil_raporu = SandboxProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "docker_sandbox_paneli.png")

    SandboxGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Docker Sandbox Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 229 (FAZ 12): GÜVENLİ DOCKER SANDBOX AJANI BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
