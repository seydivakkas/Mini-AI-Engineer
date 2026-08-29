"""
DPO (Direct Preference Optimization) 6 Panelli Görselleştirici Modülü (Day 204 - FAZ 11).
"""

from typing import Dict, Any, List
import os
import matplotlib.pyplot as plt
import numpy as np


class DPOGorsellestirici:
    """DPO 6 Panelli Teşhis Panosu Motoru."""

    @classmethod
    def teshis_paneli_olustur(
        cls,
        profil_raporu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/dpo_preference_paneli.png",
    ):
        """6 Panelli DPO Kapalı Form Tercih Hizalama Teşhis Panosu."""
        os.makedirs(os.path.dirname(os.path.abspath(kayit_yolu)), exist_ok=True)

        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "DAY 204 (FAZ 11): DPO (DIRECT PREFERENCE OPTIMIZATION) KAPALI FORM TERCİH HİZALAMASI",
            fontsize=18,
            fontweight="bold",
            color="#38bdf8",
            y=0.98,
        )

        adimlar = profil_raporu["adimlar"]
        kayiplar = profil_raporu["kayiplar"]
        dogruluklar = profil_raporu["dogruluklar"]
        marjlar = profil_raporu["odul_marjlari"]

        # -------------------------------------------------------------
        # PANEL 1: DPO Kapalı Form Mimari Akışı
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        bloklar = ["1. İkili Veri (x, y_w, y_l)", "2. Politika LogP (π_θ)", "3. Referans LogP (π_ref)", "4. Örtük Ödül Farkı (Δr)", "5. Bradley-Terry Kaybı (-log σ)"]
        skorlar1 = [1.0, 1.3, 1.3, 1.8, 2.0]
        ax1.barh(bloklar[::-1], skorlar1[::-1], color=["#38bdf8", "#10b981", "#64748b", "#f59e0b", "#a855f7"], height=0.45)
        ax1.set_xlabel("Akış Basamakları", fontsize=10, color="#cbd5e1")
        ax1.set_title("1. DPO Kapalı Form Matematiksel Akışı", fontsize=11, color="#38bdf8", fontweight="bold")
        ax1.grid(axis="x", linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 2: DPO Kayıp (Loss) Yakınsama Eğrisi
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.plot(adimlar, kayiplar, marker="o", color="#ef4444", lw=2.2, label="DPO Kaybı (-log σ(Δ))")
        ax2.set_xlabel("DPO Eğitim Adımı", fontsize=10, color="#cbd5e1")
        ax2.set_ylabel("Kayıp Değeri (Loss)", fontsize=10, color="#cbd5e1")
        ax2.set_title(f"2. Kayıp Yakınsaması ({kayiplar[0]:.3f} -> {kayiplar[-1]:.3f})", fontsize=11, color="#38bdf8", fontweight="bold")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 3: Tercih Doğruluk Oranı (%) Evrimi
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.plot(adimlar, dogruluklar, marker="s", color="#10b981", lw=2.2, label="Tercih Doğruluğu (%)")
        ax3.axhline(50.0, color="#ffffff", linestyle="--", alpha=0.4, label="Rastgele Tahmin (%50)")
        ax3.set_xlabel("Eğitim Adımı", fontsize=10, color="#cbd5e1")
        ax3.set_ylabel("Doğruluk (%)", fontsize=10, color="#cbd5e1")
        ax3.set_title(f"3. Tercih Doğruluk Kazanımı (%50 -> %{dogruluklar[-1]:.1f})", fontsize=11, color="#38bdf8", fontweight="bold")
        ax3.legend(loc="lower right", fontsize=8)
        ax3.grid(True, linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 4: Örtük Ödül Marjı (Reward Margin Δr) Genişlemesi
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        bars4 = ax4.bar(adimlar, marjlar, color="#8b5cf6", width=0.55)
        ax4.set_xlabel("Eğitim Adımı", fontsize=10, color="#cbd5e1")
        ax4.set_ylabel("Ödül Marjı Δ = r(y_w) - r(y_l)", fontsize=10, color="#cbd5e1")
        ax4.set_title("4. Örtük Ödül Ayrışması (Margin Expansion)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax4.grid(axis="y", linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 5: PPO RLHF vs DPO Mimari Kıyası
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        kriterler = ["Gereken Model Sayısı", "RL Örnekleme Döngüsü", "Eğitim İstikrarsızlığı", "Gereken GPU Belleği"]
        ppo_skor = [4, 5, 4, 5]
        dpo_skor = [2, 0, 1, 2]

        x = np.arange(len(kriterler))
        w = 0.35
        ax5.bar(x - w/2, ppo_skor, width=w, label="PPO RLHF", color="#ef4444")
        ax5.bar(x + w/2, dpo_skor, width=w, label="DPO (Kapalı Form)", color="#10b981")
        ax5.set_xticks(x)
        ax5.set_xticklabels(kriterler, fontsize=8.5, rotation=12)
        ax5.set_ylabel("Karmaşıklık / Yük Seviyesi (1-5)", fontsize=9.5, color="#cbd5e1")
        ax5.set_title("5. PPO RLHF vs DPO Mimari Karşılaştırması", fontsize=11, color="#38bdf8", fontweight="bold")
        ax5.legend(loc="upper right", fontsize=8)
        ax5.grid(axis="y", linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 6: GÜN 204 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")

        ozet_metin = (
            "GÜN 204: DPO DIRECT PREFERENCE OPTIMIZATION KARNESİ\n"
            "----------------------------------------------------\n"
            "• Algoritma           : Direct Preference Optimization (DPO)\n"
            "• Referans Makale     : Rafailov et al. (NeurIPS 2023)\n"
            "• Ödül Modeli İhtiyacı: %0 (Sıfır Ayrı Ödül Modeli / Zero RM)\n"
            "• RL Eğitimi İhtiyacı : %0 (Sıfır Actor-Critic / Zero RL)\n"
            "• Bradley-Terry Eşleme: r_implicit = beta * log(pi / pi_ref)\n"
            f"• Son Tercih Doğruluğu: %{profil_raporu['son_dogruluk']:.1f}\n"
            f"• Son Ödül Marjı      : Δ = +{profil_raporu['son_marj']:.2f}\n"
            "----------------------------------------------------\n"
            "SONUÇ: Karmaşık RL döngüleri yerine doğrudan kapalı formda\n"
            "ikili insan tercihleri optimize edilerek SOTA hizalama sağlandı!"
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
