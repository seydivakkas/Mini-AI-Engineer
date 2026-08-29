"""
Day 286 (FAZ 15): Dünya Modelleri ve DreamerV3 6 Panelli Görselleştirici Modülü.
"""

from typing import Dict, Any
import os
import matplotlib.pyplot as plt
import numpy as np


class WorldModelGorsellestirici:
    """FAZ 15 Dünya Modelleri & DreamerV3 Teşhis Panosu Motoru."""

    @classmethod
    def teshis_paneli_olustur(
        cls,
        profil_raporu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/world_model_dreamerv3_paneli.png",
    ):
        """6 Panelli Dünya Modeli Teşhis Panosu."""
        os.makedirs(os.path.dirname(os.path.abspath(kayit_yolu)), exist_ok=True)

        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "DAY 286 (FAZ 15): DÜNYA MODELLERİ (WORLD MODELS) VE ÜRETKEN SİMÜLASYON — DREAMERV3 & RSSM",
            fontsize=15,
            fontweight="bold",
            color="#38bdf8",
            y=0.98,
        )

        karsilastirma = profil_raporu["karsilastirma"]
        modeller = ["1. Model-Free\n(PPO)", "2. Model-Based\n(MBPO)", "3. Dünya Modeli\n(DreamerV3)"]
        renkler = ["#ef4444", "#f59e0b", "#10b981"]

        # -------------------------------------------------------------
        # PANEL 1: Nihai Epizodik Ödül Skoru
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        rewards = [
            karsilastirma["nihai_epizodik_odul"]["Model_Free_PPO"],
            karsilastirma["nihai_epizodik_odul"]["Model_Based_MBPO"],
            karsilastirma["nihai_epizodik_odul"]["DreamerV3_WorldModel"],
        ]
        b1 = ax1.bar(modeller, rewards, color=renkler, width=0.45)
        ax1.set_ylabel("Epizodik Ödül Skoru", fontsize=10, color="#cbd5e1")
        ax1.set_title("1. Nihai Ajan Başarımı (740.0 -> 965.0 Skor)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax1.set_ylim(0, 1200)
        ax1.grid(axis="y", linestyle=":", alpha=0.3)

        for b in b1:
            h = b.get_height()
            ax1.text(b.get_x() + b.get_width() / 2.0, h + 20.0, f"{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 2: Gerekli Çevre Adımı (Log Ölçek - Düşük İyi)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        steps = [
            karsilastirma["gerekli_cevre_adimi"]["Model_Free_PPO"],
            karsilastirma["gerekli_cevre_adimi"]["Model_Based_MBPO"],
            karsilastirma["gerekli_cevre_adimi"]["DreamerV3_WorldModel"],
        ]
        b2 = ax2.bar(modeller, steps, color=["#ef4444", "#f59e0b", "#10b981"], width=0.45)
        ax2.set_ylabel("Gerekli Gerçek Adım (Log Ölçek)", fontsize=10, color="#cbd5e1")
        ax2.set_yscale("log")
        ax2.set_title("2. Örnek Verimliliği (1M -> 10K Adım | 100x Hızlı)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax2.set_ylim(1000, 3000000)
        ax2.grid(axis="y", linestyle=":", alpha=0.3)

        for b in b2:
            h = b.get_height()
            label = f"{int(h/1000)}k Adım" if h >= 1000 else f"{int(h)}"
            ax2.text(b.get_x() + b.get_width() / 2.0, h * 1.3, label, ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=9.5)

        # -------------------------------------------------------------
        # PANEL 3: Örnek Verimliliği Öğrenme Eğrisi
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        adimlar = profil_raporu["adimlar"]
        ax3.plot(adimlar, profil_raporu["ppo_curve"], "o--", color="#ef4444", label="Model-Free PPO", linewidth=2.0)
        ax3.plot(adimlar, profil_raporu["mbpo_curve"], "s-.", color="#f59e0b", label="MBPO", linewidth=2.0)
        ax3.plot(adimlar, profil_raporu["dreamer_curve"], "^-", color="#10b981", label="DreamerV3 Dünya Modeli", linewidth=2.5)

        ax3.set_xlabel("Gerçek Çevre Adımı (x1000)", fontsize=10, color="#cbd5e1")
        ax3.set_ylabel("Epizodik Ödül", fontsize=10, color="#cbd5e1")
        ax3.set_title("3. Yakınsama Hızı Kıyaslaması", fontsize=11, color="#38bdf8", fontweight="bold")
        ax3.set_ylim(0, 1100)
        ax3.legend(loc="lower right", facecolor="#1e293b", edgecolor="#38bdf8", fontsize=8.5)
        ax3.grid(True, linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 4: Hayal Gücü (Latent Imagination Horizon H=15)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        imag_res = profil_raporu["imagination_res"]
        horizon = imag_res["horizon"]
        imag_rewards = imag_res["imagined_rewards"]
        cum_rewards = np.cumsum(imag_rewards)

        ax4.plot(range(1, horizon + 1), cum_rewards, "o-", color="#38bdf8", linewidth=2.2, markersize=6)
        ax4.fill_between(range(1, horizon + 1), 0, cum_rewards, color="#38bdf8", alpha=0.2)
        ax4.set_xlabel("Gizil Hayal Gücü Adımı (Horizon H=15)", fontsize=10, color="#cbd5e1")
        ax4.set_ylabel("Kümülatif Tahmin Edilen Ödül", fontsize=10, color="#cbd5e1")
        ax4.set_title("4. Latent Imagination İç Simülasyonu", fontsize=11, color="#38bdf8", fontweight="bold")
        ax4.set_xticks(range(1, horizon + 1, 2))
        ax4.grid(True, linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 5: RSSM Durum Dinamiği (Deterministik & Stokastik)
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        x_pts = np.linspace(-3, 3, 100)
        p_prior = np.exp(-0.5 * (x_pts - 0.2)**2) / np.sqrt(2 * np.pi)
        q_post = np.exp(-0.5 * ((x_pts - 0.25) / 0.8)**2) / (0.8 * np.sqrt(2 * np.pi))

        ax5.plot(x_pts, p_prior, "--", color="#38bdf8", label="Öncül p(z_t | h_t) (Hayal Gücü)", linewidth=2.0)
        ax5.plot(x_pts, q_post, "-", color="#10b981", label="Posterior q(z_t | h_t, x_t) (Algı)", linewidth=2.0)
        ax5.fill_between(x_pts, p_prior, q_post, color="#f59e0b", alpha=0.2, label="KL Regülarizasyonu")

        ax5.set_xlabel("Stokastik Gizil Durum (z_t)", fontsize=10, color="#cbd5e1")
        ax5.set_ylabel("Olasılık Yoğunluğu", fontsize=10, color="#cbd5e1")
        ax5.set_title("5. RSSM Durum Uzayı ve KL Dağılımı", fontsize=11, color="#38bdf8", fontweight="bold")
        ax5.legend(loc="upper right", facecolor="#1e293b", edgecolor="#38bdf8", fontsize=7.5)
        ax5.grid(True, linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 6: DreamerV3 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")

        ozet_metin = (
            "DÜNYA MODELLERİ (DREAMERV3) RAPORU\n"
            "====================================================\n"
            "• Mimarî Yapı          : Recurrent State-Space Model (RSSM)\n"
            "• Durum Ayrışımı       : Deterministik h_t + Stokastik z_t\n"
            "• Hayal Gücü Ufku      : H = 15 Adım İleri Simülasyon\n"
            "• Örnek Verimliliği    : 100x Daha Hızlı Öğrenme (10k vs 1M)\n"
            "• Nihai Ödül Skoru     : 965.0 (Model-Free PPO: 740.0)\n"
            "• Fiziksel Güvenlik    : %0 Gerçek Kaza (Hayal Gücünde Deneme)\n"
            "• Politika Eğitimi     : Latent Actor-Critic + λ-Returns\n"
            "• Genel Yapay Zeka     : Çevre Dinamiğini İçsel Modelleme\n"
            "----------------------------------------------------\n"
            "FAZ 15 GÜN 286 DÜNYA MODELLERİ MODÜLÜ TAMAMLANDI!\n"
            "Sırada: Day 287 (Diffüzyon Tabanlı Planlayıcılar - Diffusion Policy)"
        )

        ax6.text(
            0.05,
            0.5,
            ozet_metin,
            fontsize=9.2,
            family="monospace",
            color="#f8fafc",
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#1e293b", edgecolor="#38bdf8", alpha=0.9),
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=300, bbox_inches="tight")
        plt.close()
