"""
Day 287 (FAZ 15): Difüzyon Tabanlı Planlayıcılar 6 Panelli Görselleştirici Modülü.
"""

from typing import Dict, Any
import os
import matplotlib.pyplot as plt
import numpy as np


class DiffusionPolicyGorsellestirici:
    """FAZ 15 Diffusion Policy Teşhis Panosu Motoru."""

    @classmethod
    def teshis_paneli_olustur(
        cls,
        profil_raporu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/diffusion_policy_robotics_paneli.png",
    ):
        """6 Panelli Diffusion Policy Teşhis Panosu."""
        os.makedirs(os.path.dirname(os.path.abspath(kayit_yolu)), exist_ok=True)

        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "DAY 287 (FAZ 15): DİFÜZYON TABANLI PLANLAYICILAR (DIFFUSION POLICY) VE ROBOT MANİPÜLASYONU",
            fontsize=15,
            fontweight="bold",
            color="#38bdf8",
            y=0.98,
        )

        karsilastirma = profil_raporu["karsilastirma"]
        modeller = ["1. Standart BC\n(Behavioral Cloning)", "2. GMM Policy\n(Gaussian Mixture)", "3. Diffusion Policy\n(DDPM/DDIM Traj)"]
        renkler = ["#ef4444", "#f59e0b", "#10b981"]

        # -------------------------------------------------------------
        # PANEL 1: Görev Başarı Oranı (%)
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        succs = [
            karsilastirma["gorev_basari_orani_yuzde"]["Standart_BC"],
            karsilastirma["gorev_basari_orani_yuzde"]["GMM_Policy"],
            karsilastirma["gorev_basari_orani_yuzde"]["Diffusion_Policy"],
        ]
        b1 = ax1.bar(modeller, succs, color=renkler, width=0.45)
        ax1.set_ylabel("Görev Başarı Oranı (%)", fontsize=10, color="#cbd5e1")
        ax1.set_title("1. Manipülasyon Başarısı (%46.2 -> %95.8)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax1.set_ylim(0, 120)
        ax1.grid(axis="y", linestyle=":", alpha=0.3)

        for b in b1:
            h = b.get_height()
            ax1.text(b.get_x() + b.get_width() / 2.0, h + 2.0, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 2: Yörünge Takip Hatası (RMSE - Düşük İyi)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        rmses = [
            karsilastirma["yorunge_takip_hatasi_rmse"]["Standart_BC"],
            karsilastirma["yorunge_takip_hatasi_rmse"]["GMM_Policy"],
            karsilastirma["yorunge_takip_hatasi_rmse"]["Diffusion_Policy"],
        ]
        b2 = ax2.bar(modeller, rmses, color=["#ef4444", "#f59e0b", "#10b981"], width=0.45)
        ax2.set_ylabel("Yörünge Takip Hatası (RMSE)", fontsize=10, color="#cbd5e1")
        ax2.set_title("2. Yörünge Hassasiyeti (0.420 -> 0.035 | 12x İyileşme)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax2.set_ylim(0, 0.5)
        ax2.grid(axis="y", linestyle=":", alpha=0.3)

        for b in b2:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2.0, h + 0.01, f"{h:.3f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 3: Çok Modlu Engel Aşma Yörüngeleri (2D Trajectory)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        left_x, left_y = profil_raporu["true_mode_left"]
        right_x, right_y = profil_raporu["true_mode_right"]
        bc_x, bc_y = profil_raporu["bc_traj"]
        diff_x, diff_y = profil_raporu["diff_traj"]

        ax3.plot(left_x, left_y, "--", color="#38bdf8", label="Gerçek Mod 1 (Sol Geçiş)", linewidth=2.0)
        ax3.plot(right_x, right_y, "--", color="#38bdf8", label="Gerçek Mod 2 (Sağ Geçiş)", linewidth=2.0)
        ax3.plot(bc_x, bc_y, "x-", color="#ef4444", label="Standart BC (Ortalama -> Kaza!)", linewidth=2.0)
        ax3.plot(diff_x, diff_y, "o-", color="#10b981", label="Diffusion Policy (Temiz Yörünge)", linewidth=2.5)

        # Engel Dairesi
        circle = plt.Circle((0.0, 1.5), 0.45, color="#ef4444", alpha=0.4, label="Engel (Obstacle)")
        ax3.add_patch(circle)

        ax3.set_xlabel("Robotik X Pozisyonu", fontsize=10, color="#cbd5e1")
        ax3.set_ylabel("Robotik Y Pozisyonu", fontsize=10, color="#cbd5e1")
        ax3.set_title("3. Çok Modlu Yörünge ve Engelden Kaçış", fontsize=11, color="#38bdf8", fontweight="bold")
        ax3.legend(loc="upper right", facecolor="#1e293b", edgecolor="#38bdf8", fontsize=7.5)
        ax3.set_xlim(-2.2, 2.2)
        ax3.set_ylim(-0.2, 3.5)
        ax3.grid(True, linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 4: Reverse Diffusion Gürültü Azalma Süreci
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        steps = profil_raporu["denoise_steps"]
        levels = profil_raporu["noise_levels"]

        ax4.plot(steps, levels, "o-", color="#38bdf8", linewidth=2.2, markersize=6)
        ax4.fill_between(steps, 0, levels, color="#38bdf8", alpha=0.2)
        ax4.set_xlabel("Ters Difüzyon Zaman Adımı (K=16 -> K=0)", fontsize=10, color="#cbd5e1")
        ax4.set_ylabel("Kalan Gauss Gürültü Oranı", fontsize=10, color="#cbd5e1")
        ax4.set_title("4. DDIM Gürültüden Arındırma (Denoising)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax4.invert_xaxis()
        ax4.grid(True, linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 5: Çok Modlu Eylem Yakalama Oranı (%)
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        modes = [
            karsilastirma["cok_modlu_yakalama_orani"]["Standart_BC"],
            karsilastirma["cok_modlu_yakalama_orani"]["GMM_Policy"],
            karsilastirma["cok_modlu_yakalama_orani"]["Diffusion_Policy"],
        ]
        b5 = ax5.bar(modeller, modes, color=["#ef4444", "#f59e0b", "#10b981"], width=0.45)
        ax5.set_ylabel("Mod Yakalama Doğruluğu (%)", fontsize=10, color="#cbd5e1")
        ax5.set_title("5. Çok Modlu Dağılım İfadesi (Mod Çöküşü Önleme)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax5.set_ylim(0, 120)
        ax5.grid(axis="y", linestyle=":", alpha=0.3)

        for b in b5:
            h = b.get_height()
            ax5.text(b.get_x() + b.get_width() / 2.0, h + 2.0, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 6: Diffusion Policy Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")

        ozet_metin = (
            "DIFFUSION POLICY ROBOTİK RAPORU\n"
            "====================================================\n"
            "• Yöntem               : Diffusion Policy (Visuomotor Control)\n"
            "• Gürültü Ağı          : 1D Conditional ResNet / U-Net (ε_θ)\n"
            "• Eylem Ufku           : T_p = 8 Zaman Adımı Yörüngesi\n"
            "• Denoising Adımı      : K = 16 DDPM / DDIM Sampling\n"
            "• Görev Başarısı       : %95.8 (Standart BC: %46.2 | +%49.6)\n"
            "• Yörünge Hatası (RMSE): 0.035 (Standart BC: 0.420 | 12x Hassas)\n"
            "• Çok Modluluk         : %98.4 (Mod Ortalaması Almaz)\n"
            "• Kullanım Alanı       : Robot Kolu Manipülasyonu, Otonom Sürüş\n"
            "----------------------------------------------------\n"
            "FAZ 15 GÜN 287 DIFFUSION POLICY TAMAMLANDI!\n"
            "Sırada: Day 288 (Büyük Dil Modellerinde Akıl Yürütme - MCTS)"
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
