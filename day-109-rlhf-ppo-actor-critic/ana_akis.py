"""
Day 109: Proximal Policy Optimization (PPO) ile LLM Hizalama Ana Akışı.
4-Modelli RLHF orkestrasyonu, GAE avantajları, KL cezası ve 6 panelli teşhis panosu.
"""

import os
import sys
import torch

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.ppo_laboratuvari import PPOLaboratuvari
from src.gorsellestirici import PPOGorsellestirici


def main():
    print("=" * 95)
    print(">>> Day 109: RLHF with PPO (Actor, Critic, Reference Model, Reward Model)")
    print("=" * 95)

    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Çalışma Donanımı: {cihaz.type.upper()}")

    # -------------------------------------------------------------
    # ADIM 1: 4-Modelli PPO Laboratuvarının Başlatılması
    # -------------------------------------------------------------
    print("\n[1/3] 4-Modelli PPO RLHF Sistemi Başlatılıyor...")
    lab = PPOLaboratuvari(
        vocab_size=1000,
        dim=256,
        num_heads=4,
        num_layers=4,
        cihaz=cihaz,
    )
    actor_p = sum(p.numel() for p in lab.actor.parameters())
    critic_p = sum(p.numel() for p in lab.critic.parameters())
    print(f"  * 1. Actor (Politika) Parametreleri     : {actor_p:,}")
    print(f"  * 2. Critic (Değer Ağı) Parametreleri  : {critic_p:,}")
    print(f"  * 3. Reference Model                    : {actor_p:,} (Dondurulmuş)")
    print("  * 4. Reward Model                      : Dondurulmuş Skaler Puanlayıcı")

    # -------------------------------------------------------------
    # ADIM 2: PPO Hizalama Eğitimi (Rollout + GAE + Policy Update)
    # -------------------------------------------------------------
    print("\n[2/3] PPO Hizalama Eğitimi Koşturuluyor (15 İterasyon, KL Beta=0.05, Clip Eps=0.2)...")
    egitim_raporu = lab.ppo_egitim_dongusu(
        epok_sayisi=15,
        batch_size=32,
        prompt_len=12,
        max_new_tokens=14,
    )

    print("\n--- PPO ACTOR-CRITIC EĞİTİM GELİŞİM RAPORU ---")
    print(f"{'ADIM':<6} | {'ORT ÖDÜL':<12} | {'KL SAPMASI':<14} | {'POLİTİKA KAYBI':<16} | {'DEĞER KAYBI':<14} | {'KIRPMA (%)':<12}")
    print("-" * 95)
    for i in range(len(egitim_raporu["oduller"])):
        print(
            f"{i+1:<6} | "
            f"{egitim_raporu['oduller'][i]:>10.3f} | "
            f"{egitim_raporu['kl_sapmalari'][i]:>12.4f} | "
            f"{egitim_raporu['politika_kayiplari'][i]:>14.4f} | "
            f"{egitim_raporu['deger_kayiplari'][i]:>12.4f} | "
            f"%{egitim_raporu['kirpma_oranlari'][i]:>10.1f}"
        )
    print("-" * 95)

    ilk_odul = egitim_raporu["oduller"][0]
    son_odul = egitim_raporu["oduller"][-1]
    son_kl = egitim_raporu["kl_sapmalari"][-1]

    print("\n[-] PPO HİZALAMA BAŞARI ÖZETİ:")
    print(f"  * Başlangıç Ortalama Ödülü : {ilk_odul:+.3f}")
    print(f"  * Nihai Ortalama Ödül      : {son_odul:+.3f} (Önemli Artış!)")
    print(f"  * Nihai KL Sapması (D_KL)  : {son_kl:.4f} (Kontrol Altında)")

    # -------------------------------------------------------------
    # ADIM 3: 6 Panelli Teşhis Panosu Çizimi
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli PPO Actor-Critic Teşhis Panosu Çiziliyor...")
    gorsellestirici = PPOGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ciktilar",
        "ppo_actor_critic_paneli.png",
    )
    gorsellestirici.pano_olustur(
        egitim_raporu,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 95)
    print("[OK] Day 109: PPO ile LLM Hizalama Analizleri Başarıyla Tamamlandı!")
    print("=" * 95)


if __name__ == "__main__":
    main()
