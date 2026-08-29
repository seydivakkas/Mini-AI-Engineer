"""
Day 202: GRPO (Group Relative Policy Optimization) ile Matematiksel Akıl Yürütme Ana Akışı.
"""

import os
import sys
import torch

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.grpo_motoru import (
    MathProblemEnvironment,
    RuleBasedMathRewardVerifier,
    GRPOTrainer,
)
from src.grpo_profilleyici import GRPOAkisProfilleyici
from src.gorsellestirici import GRPOGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 202 (FAZ 11 BAŞLANGICI): GRPO (GROUP RELATIVE POLICY OPTIMIZATION) MATH REASONING ENGINE")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Problem Ortamı ve Örnek Soru Üretimi
    # -------------------------------------------------------------
    print("\n[1/4] Deterministik Matematik Problemi Üretiliyor...")
    soru_dict = MathProblemEnvironment.rastgele_problem_uret()
    print(f"  • Problem Türü        : {soru_dict['tur'].capitalize()}")
    print(f"  • Soru Metni          : {soru_dict['soru']}")
    print(f"  • Kesin Doğru Cevap   : {soru_dict['dogru_cevap']}")

    # -------------------------------------------------------------
    # ADIM 2: DeepSeek-R1 Kural Tabanlı Ödül Doğrulayıcısı Testi
    # -------------------------------------------------------------
    print("\n[2/4] Kural Tabanlı Ödül Doğrulayıcısı (Biçim + Kesin Eşleşme) Test Ediliyor...")
    ornek_yanit = (
        f"<think>\n"
        f"1. Problem analiz edildi.\n"
        f"2. Adım adım aritmetik hesaplama yapıldı.\n"
        f"</think>\n"
        f"Sonuç: {soru_dict['dogru_cevap']}"
    )
    odul = RuleBasedMathRewardVerifier.odul_hesapla(ornek_yanit, soru_dict["dogru_cevap"])
    print(f"  • Format Ödülü (<think>...) : {odul['format_odulu']:.2f} / 0.20")
    print(f"  • Doğruluk Ödülü (Exact Ans): {odul['dogruluk_odulu']:.2f} / 0.80")
    print(f"  • Toplam Ödül Skoru        : {odul['toplam_odul']:.2f} / 1.00")
    print("  ✓ Kural Tabanlı Doğrulayıcı Başarıyla Çalıştı!")

    # -------------------------------------------------------------
    # ADIM 3: GRPO Eğitim Adımı ve Grup İçi Bağıl Avantaj
    # -------------------------------------------------------------
    print("\n[3/4] 10 Adımlık GRPO Post-Training Akışı Başlatılıyor (Sıfır Critic Modeli)...")
    profil_raporu = GRPOAkisProfilleyici.egitim_akisini_profili_cikar(adim_sayisi=10)

    print("-" * 115)
    print(f"{'Eğitim Adımı':<16} | {'Toplam Kayıp (Loss)':<22} | {'Ortalama Ödül':<18} | {'Doğruluk Oranı':<18} | {'Düşünce Uzunluğu'}")
    print("-" * 115)
    for adim, kayip, odul_val, acc, uz in zip(
        profil_raporu["adimlar"],
        profil_raporu["kayiplar"],
        profil_raporu["oduller"],
        profil_raporu["dogruluk_oranlari"],
        profil_raporu["dusunce_uzunluklari"],
    ):
        print(
            f"Adım #{adim:<10} | "
            f"{kayip:>18.4f}   | "
            f"{odul_val:>14.2f}   | "
            f"%{acc:>14.1f}   | "
            f"{uz:>12} Token"
        )
    print("-" * 115)
    print(f"  🏆 Nihai Doğruluk Seviyesi : %{profil_raporu['son_dogruluk']:.1f}")
    print(f"  ⚡ Bellek Tasarrufu (vs PPO): %{profil_raporu['ppo_vs_grpo']['bellek_tasarrufu_yuzde']:.0f} VRAM Tasarrufu")
    print(f"  🚀 Eğitim Hızlanması       : {profil_raporu['ppo_vs_grpo']['egitim_hizlanma_kat']}x Daha Yüksek Throughput")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Görsel Teşhis Panosu Üretimi
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli GRPO Matematiksel Akıl Yürütme Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "grpo_math_reasoning_paneli.png")

    GRPOGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ GRPO Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 202 (FAZ 11 BAŞLANGICI): GRPO MATEMATİKSEL AKIL YÜRÜTME BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
