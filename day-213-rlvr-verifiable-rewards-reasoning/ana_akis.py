"""
Day 213: RLVR (Reinforcement Learning with Verifiable Rewards) ve Deterministik Akıl Yürütme Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.rlvr_motoru import (
    VerifiableTaskRegistry,
    GroundTruthVerifier,
    RLVRRewardCalculator,
    RLVRExplorationEngine,
    RLVRTrainer,
)
from src.rlvr_profilleyici import RLVRProfilleyici
from src.gorsellestirici import RLVRGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 213 (FAZ 11): RLVR (REINFORCEMENT LEARNING WITH VERIFIABLE REWARDS & FORMAL GROUND TRUTH)")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Biçimsel Doğrulanabilir Görevler
    # -------------------------------------------------------------
    print("\n[1/4] Doğrulanabilir Biçimsel Görevler (Verifiable Benchmarks) Yükleniyor...")
    for i in range(len(VerifiableTaskRegistry.GOREVLER)):
        g = VerifiableTaskRegistry.gorev_getir(i)
        print(f"  • [{g['id']}] ({g['kategori']:<10}) Soru: '{g['soru']}' -> Zemin Gerçeği: {g['hedef_cevap']}")
    print("  ✓ Biçimsel Görev Havuzu Başarıyla Yüklendi!")

    # -------------------------------------------------------------
    # ADIM 2: Kanıtlanabilir Ödül Hesabı (RLVR Reward)
    # -------------------------------------------------------------
    print("\n[2/4] Zemin Gerçekliği ve Bileşik Ödül (R_acc + R_fmt + R_len) Hesaplanıyor...")
    gorev = VerifiableTaskRegistry.gorev_getir(0)
    ornek_yanit = (
        "<think>\n"
        "1. 5*x - 7 = 38 denklemini çözelim.\n"
        "2. 5*x = 45 -> x = 9 bulunur.\n"
        "</think>\n"
        "Sonuç: \\boxed{9}"
    )
    odul_raporu = RLVRRewardCalculator.odul_hesapla(ornek_yanit, gorev["hedef_cevap"])

    print(f"  • İncelenen Soru      : {gorev['soru']}")
    print(f"  • Ayıklanan Cevap     : {odul_raporu['ayiklanan_cevap']} (Doğru mu: {'✅ EVET' if odul_raporu['dogru_mu'] else '❌ HAYIR'})")
    print(f"  • Doğruluk Ödülü (R_acc): {odul_raporu['r_acc']:.2f}")
    print(f"  • Biçim Ödülü (R_fmt)   : {odul_raporu['r_fmt']:.2f}")
    print(f"  • Uzunluk Cezası (R_len): {odul_raporu['r_len']:.4f}")
    print(f"  • Toplam RLVR Ödülü     : {odul_raporu['toplam_odul']:.2f} / 1.20")
    print("  ✓ RLVR Zemin Gerçekliği Başarıyla Onaylandı!")

    # -------------------------------------------------------------
    # ADIM 3: Sıfır Varyanslı RLVR Eğitim Adımı ve 'Aha Anı'
    # -------------------------------------------------------------
    print("\n[3/4] Sıfır Varyanslı RLVR Eğitimi ve Kendi Kendini Düzeltme ('Aha Moment') Simülasyonu...")
    egitim_sonucu = RLVRTrainer.egitim_adimi(gorev["soru"], gorev["hedef_cevap"])

    print("  • Üretilen Akıl Yürütme İzi (CoT Rollout):")
    for satir in egitim_sonucu["yanit"].split("\n")[:4]:
        print(f"    {satir}")
    print(f"  • Ödül Modeli Varyansı  : {egitim_sonucu['odul_varyansi']:.2f} (Sıfır Varyans, Sıfır Goodhart İstismarı)")
    print("  ✓ RLVR Politika Adımı Başarıyla Tamamlandı!")

    # -------------------------------------------------------------
    # ADIM 4: Profilleme ve 6 Panelli Görsel Teşhis Panosu
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli RLVR ve Deterministik Akıl Yürütme Teşhis Panosu Oluşturuluyor...")
    profil_raporu = RLVRProfilleyici.karsilastirma_raporu_uret()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "rlvr_reasoning_paneli.png")

    RLVRGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ RLVR Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 213 (FAZ 11): RLVR (REINFORCEMENT LEARNING WITH VERIFIABLE REWARDS) BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
