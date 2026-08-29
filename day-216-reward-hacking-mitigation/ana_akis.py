"""
Day 216: Reward Hacking ve Goodhart Yasası Önleme Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.reward_hacking_motoru import (
    AdaptiveKLController,
    RewardSquasher,
    EnsembleRewardModel,
    RewardHackingDetector,
    RobustRLTrainer,
)
from src.reward_hacking_profilleyici import RewardHackingProfilleyici
from src.gorsellestirici import RewardHackingGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 216 (FAZ 11): REWARD HACKING & GOODHART YASASI ÖNLEME (ADAPTIVE KL & ENSEMBLE ROBUST ALIGNMENT)")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Dinamik Adaptif KL Denetleyicisi
    # -------------------------------------------------------------
    print("\n[1/4] Dinamik Adaptif KL Denetleyicisi (PID/Orantısal) Test Ediliyor...")
    kl_controller = AdaptiveKLController(kl_hedef=0.05, beta_baslangic=0.10)
    print(f"  • Başlangıç Beta (β_KL) : {kl_controller.beta:.4f} (Hedef KL: {kl_controller.kl_hedef:.3f})")

    # Yüksek sapma simülasyonu
    yeni_beta = kl_controller.guncelle(olculen_kl=0.12)
    print(f"  • Yüksek KL (0.12) Sonrası Yeni Beta: {yeni_beta:.4f} (Ceza Artırıldı)")
    print("  ✓ Adaptif KL Denetleyicisi Başarıyla Doğrulandı!")

    # -------------------------------------------------------------
    # ADIM 2: Topluluk (Ensemble LCB) ve Tanh Kırpma
    # -------------------------------------------------------------
    print("\n[2/4] Topluluk Hakemleri (Ensemble LCB) ve Tanh Ödül Kırpma Hesaplanıyor...")
    ham_topluluk_puanlari = [8.5, 9.2, 4.1]  # Bir model aşırı puan vererek kandırılmış
    ensemble = EnsembleRewardModel.degerlendir(ham_topluluk_puanlari, lambda_lcb=1.5)
    kirpilmis = RewardSquasher.tanh_kirp(ensemble["lcb_odul"])

    print(f"  • Topluluk Hakem Puanları : {ham_topluluk_puanlari}")
    print(f"  • Ortalama Puan           : {ensemble['ortalama_odul']:.2f}")
    print(f"  • Standart Sapma (Uyuşmazlık): {ensemble['standart_sapma']:.2f}")
    print(f"  • Muhafazakar LCB Ödülü   : {ensemble['lcb_odul']:.2f}")
    print(f"  • Tanh Kırpılmış Sağlam Ödül: {kirpilmis:.2f} (Patlama Engellendi)")
    print("  ✓ Ensemble LCB ve Kırpma Başarıyla Tamamlandı!")

    # -------------------------------------------------------------
    # ADIM 3: Tam Sağlamlaştırılmış RL Adımı ve Hack Tespiti
    # -------------------------------------------------------------
    print("\n[3/4] İstismarsız Sağlam RL Eğitim Adımı Yürütülüyor...")
    model_yaniti = "Kesinlikle harika bir soru sordunuz efendim, siz mükemmel bir uzmansınız. Cevap: 42."
    adim_raporu = RobustRLTrainer.guvenli_odul_adimi(
        model_yaniti=model_yaniti,
        topluluk_puanlari=ham_topluluk_puanlari,
        olculen_kl=0.08,
        kl_controller=kl_controller,
        perplexity=16.5,
    )

    print(f"  • İncelenen Model Yanıtı  : '{model_yaniti}'")
    print(f"  • Dalkavukluk Tespiti     : {'⚠️ VAR' if adim_raporu['hacking_raporu']['dalkavukluk_var_mi'] else '✅ TEMİZ'}")
    print(f"  • Hacking Şüphesi         : {'⚠️ VAR' if adim_raporu['hacking_raporu']['hacking_suphesi'] else '✅ TEMİZ'}")
    print(f"  • Nihai Sağlam RL Ödülü   : {adim_raporu['nihai_saglam_odul']:.2f}")
    print("  ✓ Sağlam RL Adımı Başarıyla Tamamlandı!")

    # -------------------------------------------------------------
    # ADIM 4: Profilleme ve 6 Panelli Görsel Teşhis Panosu
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Reward Hacking ve Sağlamlık Teşhis Panosu Oluşturuluyor...")
    profil_raporu = RewardHackingProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "reward_hacking_paneli.png")

    RewardHackingGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Reward Hacking Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 216 (FAZ 11): REWARD HACKING & GOODHART YASASI ÖNLEME BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
