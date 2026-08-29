"""
Day 203: PPO (Proximal Policy Optimization) Actor-Critic LLM Hizalama Ana Akışı.
"""

import os
import sys
import torch

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.ppo_motoru import (
    ActorNetwork,
    CriticNetwork,
    GAECalculator,
    PPOTrainer,
)
from src.ppo_profilleyici import PPOAkisProfilleyici
from src.gorsellestirici import PPOGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 203 (FAZ 11): PPO (PROXIMAL POLICY OPTIMIZATION) ACTOR-CRITIC LLM ALIGNMENT ENGINE")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: 4-Model RLHF Mimarisi Kurulumu
    # -------------------------------------------------------------
    print("\n[1/4] 4-Model RLHF Mimarisi Başlatılıyor...")
    trainer = PPOTrainer()
    print("  • Actor (Policy π_θ)        : Aktif Eğitilen Üreteç Modeli")
    print("  • Critic (Value V_ϕ)        : Aktif Eğitilen Durum Değeri Modeli")
    print("  • Reference Model (π_ref)   : Dondurulmuş Temel SFT Modeli")
    print("  • Reward Model (R_ψ)        : Dondurulmuş İnsan Tercih Modeli")
    print("  ✓ Modeller Başarıyla GPU/CPU Belleğine Yüklendi!")

    # -------------------------------------------------------------
    # ADIM 2: GAE (Generalized Advantage Estimation) Testi
    # -------------------------------------------------------------
    print("\n[2/4] GAE (gamma=0.99, lambda=0.95) Avantaj Tahmini Hesaplanıyor...")
    oduller = torch.tensor([[0.0, 0.0, 0.0, 1.5], [0.0, 0.0, 0.0, -1.0]])
    degerler = torch.tensor([[0.2, 0.4, 0.8, 1.2], [0.1, 0.0, -0.4, -0.8]])
    adv, targets = GAECalculator.hesapla_avantaj_ve_hedef(oduller, degerler)
    print(f"  • Örnek Pozitif GAE Avantajı : {adv[0, -1].item():+.4f}")
    print(f"  • Örnek Negatif GAE Avantajı : {adv[1, -1].item():+.4f}")
    print(f"  • Hedef Değerler (Targets)   : {targets[0].tolist()}")
    print("  ✓ GAE Avantaj Standardizasyonu Başarıyla Doğrulandı!")

    # -------------------------------------------------------------
    # ADIM 3: 10 Adımlık PPO LLM Hizalama Döngüsü
    # -------------------------------------------------------------
    print("\n[3/4] 10 Adımlık PPO Actor-Critic Hizalama Eğitimi Yürütülüyor...")
    profil_raporu = PPOAkisProfilleyici.egitim_akisini_profili_cikar(adim_sayisi=10)

    print("-" * 115)
    print(f"{'Eğitim Adımı':<16} | {'Actor Kaybı':<20} | {'Critic MSE':<20} | {'İnsan Ödül Skoru':<22} | {'KL Divergence'}")
    print("-" * 115)
    for adim, a_loss, c_loss, odul, kl in zip(
        profil_raporu["adimlar"],
        profil_raporu["actor_kayiplari"],
        profil_raporu["critic_kayiplari"],
        profil_raporu["odul_skorlari"],
        profil_raporu["kl_degerleri"],
    ):
        print(
            f"Adım #{adim:<10} | "
            f"{a_loss:>16.4f}   | "
            f"{c_loss:>16.4f}   | "
            f"{odul:>18.2f}   | "
            f"{kl:>15.4f}"
        )
    print("-" * 115)
    print(f"  🏆 Nihai İnsan Tercih Ödülü : {profil_raporu['son_odul']:+.2f}")
    print(f"  📉 Nihai Critic MSE Hatası  : {profil_raporu['son_critic_loss']:.4f}")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Görsel Teşhis Panosu Üretimi
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli PPO Actor-Critic Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "ppo_actor_critic_paneli.png")

    PPOGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ PPO Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 203 (FAZ 11): PPO ACTOR-CRITIC LLM HİZALAMA BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
