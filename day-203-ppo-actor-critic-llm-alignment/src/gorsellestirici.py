"""
PPO Actor-Critic LLM Hizalama 6 Panelli Görselleştirici Modülü (Day 203 - FAZ 11).
"""

from typing import Dict, Any, List
import os
import matplotlib.pyplot as plt
import numpy as np


class PPOGorsellestirici:
    """PPO 6 Panelli Teşhis Panosu Motoru."""

    @classmethod
    def teshis_paneli_olustur(
        cls,
        profil_raporu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/ppo_actor_critic_paneli.png",
    ):
        """6 Panelli PPO Actor-Critic Teşhis Panosu."""
        os.makedirs(os.path.dirname(os.path.abspath(kayit_yolu)), exist_ok=True)

        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "DAY 203 (FAZ 11): PPO (PROXIMAL POLICY OPTIMIZATION) ACTOR-CRITIC LLM HİZALAMA VE GAE",
            fontsize=18,
            fontweight="bold",
            color="#38bdf8",
            y=0.98,
        )

        adimlar = profil_raporu["adimlar"]
        oduller = profil_raporu["odul_skorlari"]
        critic_loss = profil_raporu["critic_kayiplari"]
        kl_vals = profil_raporu["kl_degerleri"]

        # -------------------------------------------------------------
        # PANEL 1: 4-Modelli RLHF Mimarisi
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        modeller = ["1. Actor (Policy - π_θ)", "2. Critic (Value - V_ϕ)", "3. Ref Model (π_ref)", "4. Reward Model (R_ψ)", "5. GAE Avantaj Motoru"]
        bellek_orani = [1.0, 1.0, 1.0, 1.0, 0.4]
        renkler1 = ["#38bdf8", "#10b981", "#64748b", "#f59e0b", "#a855f7"]
        ax1.barh(modeller[::-1], bellek_orani[::-1], color=renkler1[::-1], height=0.45)
        ax1.set_xlabel("Gereken Bellek / İşlem Yükü Oranı", fontsize=10, color="#cbd5e1")
        ax1.set_title("1. Standart PPO RLHF 4-Model Mimarisi", fontsize=11, color="#38bdf8", fontweight="bold")
        ax1.grid(axis="x", linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 2: İnsan Tercihi / Hizalama Ödül Skoru Evrimi
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.plot(adimlar, oduller, marker="o", color="#10b981", lw=2.2, label="Ortalama Ödül (Reward)")
        ax2.axhline(0.0, color="#ffffff", linestyle="--", alpha=0.4)
        ax2.set_xlabel("PPO Eğitim Adımı", fontsize=10, color="#cbd5e1")
        ax2.set_ylabel("İnsan Tercih Ödülü (R_ψ)", fontsize=10, color="#cbd5e1")
        ax2.set_title(f"2. Ödül Skoru Gelişimi ({oduller[0]:.2f} -> {oduller[-1]:.2f})", fontsize=11, color="#38bdf8", fontweight="bold")
        ax2.legend(loc="lower right", fontsize=8)
        ax2.grid(True, linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 3: Critic Değer Ağı (Value MSE Loss) Yakınsaması
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.plot(adimlar, critic_loss, marker="s", color="#ef4444", lw=2.0, label="Critic MSE Kaybı")
        ax3.set_xlabel("Eğitim Adımı", fontsize=10, color="#cbd5e1")
        ax3.set_ylabel("Değer Tahmin Hatası (MSE)", fontsize=10, color="#cbd5e1")
        ax3.set_title("3. Critic Değer Yakınsama Eğrisi", fontsize=11, color="#38bdf8", fontweight="bold")
        ax3.legend(loc="upper right", fontsize=8)
        ax3.grid(True, linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 4: KL Divergence Güvenlik Bütçesi
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        bars4 = ax4.bar(adimlar, kl_vals, color="#f59e0b", width=0.55)
        ax4.axhline(0.20, color="#ef4444", linestyle="--", lw=1.5, label="Güvenlik Üst Sınırı (KL Max)")
        ax4.set_xlabel("Eğitim Adımı", fontsize=10, color="#cbd5e1")
        ax4.set_ylabel("D_KL(π_θ || π_ref)", fontsize=10, color="#cbd5e1")
        ax4.set_title("4. KL Divergence Güvenlik Takibi", fontsize=11, color="#38bdf8", fontweight="bold")
        ax4.legend(loc="upper right", fontsize=8)
        ax4.grid(axis="y", linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 5: GAE (γ=0.99, λ=0.95) Token Avantaj Profillemesi
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        token_adimlari = [f"Tok #{i+1}" for i in range(8)]
        ornek_adv = [0.12, -0.05, 0.45, 0.88, -0.30, 0.65, 1.20, 1.45]
        renkler5 = ["#10b981" if v >= 0 else "#ef4444" for v in ornek_adv]
        bars5 = ax5.bar(token_adimlari, ornek_adv, color=renkler5, width=0.5)
        ax5.axhline(0.0, color="#ffffff", linestyle="-", lw=1.0)
        ax5.set_ylabel("Standartlaştırılmış GAE Avantajı", fontsize=10, color="#cbd5e1")
        ax5.set_title("5. Token Başına GAE Avantaj Dağılımı", fontsize=11, color="#38bdf8", fontweight="bold")
        ax5.grid(axis="y", linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 6: GÜN 203 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")

        ozet_metin = (
            "GÜN 203: PPO ACTOR-CRITIC LLM HİZALAMA KARNESİ\n"
            "----------------------------------------------------\n"
            "• Algoritma           : Proximal Policy Optimization (PPO)\n"
            "• Kullanım Amacı      : RLHF (İnsan Geri Bildirimiyle Hizalama)\n"
            "• Model Mimarisi      : 4 Model (Actor + Critic + Ref + RM)\n"
            "• Avantaj Tahmini     : GAE (gamma=0.99, lambda=0.95)\n"
            "• Politika Kırpma     : Clip Epsilon = 0.20\n"
            "• Güvenlik Kilidi     : Token Düzeyinde KL Divergence Cezası\n"
            f"• Son Ödül Skoru      : {profil_raporu['son_odul']:+.2f} (Pozitif İnsan Uyumu)\n"
            "----------------------------------------------------\n"
            "SONUÇ: PPO Actor-Critic ve GAE mekanizması ile model\n"
            "halüsinasyonlardan arındırılarak insan değerlerine hizalandı!"
        )

        ax6.text(
            0.05,
            0.5,
            ozet_metin,
            fontsize=9.5,
            family="monospace",
            color="#f8fafc",
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#1e293b", edgecolor="#38bdf8", alpha=0.9),
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=300, bbox_inches="tight")
        plt.close()
