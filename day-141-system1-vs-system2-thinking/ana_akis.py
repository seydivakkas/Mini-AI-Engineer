"""
Day 141: System 1 (Hızlı/Sezgisel) vs System 2 (Yavaş/Akıl Yürüten) LLM Mimarisi Ana Akışı.
Bilişsel Yansıma Testi (CRT) ve Test-Time Compute simülasyonu.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.sistem2_motoru import Sistem2Motoru
from src.bilissel_karsilastirici import BilisselKarsilastirici
from src.gorsellestirici import System1VsSystem2Gorsellestirici


def main():
    print("=" * 105)
    print(">>> Day 141: System 1 vs System 2 Thinking - FAZ 8 (Reasoning LLMs & Test-Time Compute)")
    print("=" * 105)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    karsilastirici = BilisselKarsilastirici()
    motor2 = Sistem2Motoru()

    # -------------------------------------------------------------
    # ADIM 1: Bilişsel Yansıma Testi (CRT) Kıyaslaması
    # -------------------------------------------------------------
    print("\n[1/3] Bilişsel Yansıma Testi (CRT) Üzerinde Kıyaslama Yapılıyor...")
    sonuclar = karsilastirici.karsilastir()

    print(f"\n{'SİSTEM':<32} | {'DOĞRULUK':<12} | {'ORTALAMA GECİKME':<18} | {'DÜŞÜNME TOKENİ'}")
    print("-" * 80)
    print(
        f"{'System 1 (Hızlı / Sezgisel)':<32} | "
        f"%{sonuclar['sistem1']['dogruluk_orani']:<11.1f} | "
        f"{sonuclar['sistem1']['ortalama_gecikme_ms']:<15.2f} ms | "
        f"{sonuclar['sistem1']['toplam_dusunme_tokeni']}"
    )
    print(
        f"{'System 2 (Yavaş / Akıl Yürüten)':<32} | "
        f"%{sonuclar['sistem2']['dogruluk_orani']:<11.1f} | "
        f"{sonuclar['sistem2']['ortalama_gecikme_ms']:<15.2f} ms | "
        f"{sonuclar['sistem2']['toplam_dusunme_tokeni']}"
    )

    # -------------------------------------------------------------
    # ADIM 2: Örnek Soru Üzerinde System 2 Düşünme İzi (<think>)
    # -------------------------------------------------------------
    print("\n[2/3] Sopa ve Top Sorusu Üzerinde System 2 Düşünme İzi İnceleniyor:")
    ornek_soru_id = "sopave_top"
    ornek_soru_metin = "Bir sopa ve bir top toplamda $1.10 tutmaktadır. Sopa, toptan $1.00 daha pahalıdır. Top kaç paradır?"
    s2_detay = motor2.yanitla(ornek_soru_id, ornek_soru_metin, dusunme_butcesi=4)

    print(f"  • Soru: {ornek_soru_metin}")
    print("  • <think>")
    for adim in s2_detay["dusunme_izleri"]:
        print(f"      {adim}")
    print("    </think>")
    print(f"  • Nihai Yanıt: {s2_detay['yanit']}")
    print(f"  • Düşünme Tokeni: {s2_detay['dusunme_token_sayisi']} | Güven Skoru: %{s2_detay['guven_skoru']*100:.1f}")

    # -------------------------------------------------------------
    # ADIM 3: Test-Time Compute Ölçekleme ve Teşhis Panosu
    # -------------------------------------------------------------
    print("\n[3/3] Test-Time Compute Ölçekleme ve 6 Panelli Teşhis Panosu Üretiliyor...")
    compute_olcek = karsilastirici.test_time_compute_olceklemesi()

    gorsellestirici = System1VsSystem2Gorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "system1_vs_system2_paneli.png")
    gorsellestirici.pano_olustur(
        karsilastirma_sonucu=sonuclar,
        compute_olceklemesi=compute_olcek,
        ornek_sistem2_detayi=s2_detay,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 105)
    print("✓ Day 141: SYSTEM 1 vs SYSTEM 2 THINKING (FAZ 8 BAŞLANGICI) BAŞARIYLA TAMAMLANDI!")
    print("=" * 105)


if __name__ == "__main__":
    main()
