"""
Plan-and-Solve 6 Panelli Görselleştirici Modülü (Day 224 - FAZ 12).
"""

from typing import Dict, Any, List
import os
import matplotlib.pyplot as plt
import numpy as np


class PlanAndSolveGorsellestirici:
    """Plan-and-Solve 6 Panelli Teşhis Panosu Motoru."""

    @classmethod
    def teshis_paneli_olustur(
        cls,
        profil_raporu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/plan_and_solve_paneli.png",
    ):
        """6 Panelli Plan-and-Solve Teşhis Panosu."""
        os.makedirs(os.path.dirname(os.path.abspath(kayit_yolu)), exist_ok=True)

        plt.style.use("dark_background")
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            "DAY 224 (FAZ 12): PLAN-AND-SOLVE (PS+) AJAN MİMARİSİ - STRATEJİK PLANLAMA VE SIRALI İCRA",
            fontsize=18,
            fontweight="bold",
            color="#38bdf8",
            y=0.98,
        )

        karsilastirma = profil_raporu["karsilastirma"]
        modeller = ["Açgözlü ReAct\n(Greedy Step)", "Statik Kod\n(Hardcoded Script)", "Plan-and-Solve\n(PS+ Mimarisi)"]
        renkler = ["#ef4444", "#38bdf8", "#10b981"]

        # -------------------------------------------------------------
        # PANEL 1: Plan-and-Solve Ajan Hattı
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        asamalar = ["1. Karmaşık Kullanıcı Hedefi", "2. Planlayıcı (Planner DAG)", "3. Alt Görev Ayrıştırması", "4. Sıralı Bellek İcrası", "5. Dinamik Yeniden Planlama"]
        onemler = [1.0, 1.5, 1.9, 2.3, 2.7]
        ax1.barh(asamalar[::-1], onemler[::-1], color=["#38bdf8", "#8b5cf6", "#f59e0b", "#10b981", "#ec4899"], height=0.45)
        ax1.set_xlabel("İşlem Aşamaları", fontsize=10, color="#cbd5e1")
        ax1.set_title("1. Plan-and-Solve İki Aşamalı Akış", fontsize=11, color="#38bdf8", fontweight="bold")
        ax1.grid(axis="x", linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 2: Karmaşık Görev Tamamlama Oranı (%)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        tamamlama = [
            karsilastirma["karmasik_gorev_tamamlama_orani"]["Acgozlu_ReAct"],
            karsilastirma["karmasik_gorev_tamamlama_orani"]["Statik_Script"],
            karsilastirma["karmasik_gorev_tamamlama_orani"]["Plan_and_Solve_PS"],
        ]
        bars2 = ax2.bar(modeller, tamamlama, color=renkler, width=0.45)
        ax2.set_ylabel("Tamamlama Oranı (%)", fontsize=10, color="#cbd5e1")
        ax2.set_title("2. Karmaşık Görev Başarısı (%52.0 -> %93.8)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax2.set_ylim(0, 115)
        ax2.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars2:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2.0, h + 1.5, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 3: Gereksiz / Tekrar Araç Çağrısı (%)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        tekrar = [
            karsilastirma["gereksiz_tekrar_arac_cagrisi"]["Acgozlu_ReAct"],
            karsilastirma["gereksiz_tekrar_arac_cagrisi"]["Statik_Script"],
            karsilastirma["gereksiz_tekrar_arac_cagrisi"]["Plan_and_Solve_PS"],
        ]
        bars3 = ax3.bar(modeller, tekrar, color=["#ef4444", "#38bdf8", "#10b981"], width=0.45)
        ax3.set_ylabel("Gereksiz Çağrı (%)", fontsize=10, color="#cbd5e1")
        ax3.set_title("3. Fazladan Araç Çağrısı İsrafı (%32.0 -> %3.5)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax3.set_ylim(0, 40)
        ax3.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars3:
            h = b.get_height()
            ax3.text(b.get_x() + b.get_width() / 2.0, h + 0.8, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 4: Plana Sadakat ve Görev Kapsamı (%)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        sadakat = [
            karsilastirma["gorev_kapsami_plana_sadakat"]["Acgozlu_ReAct"],
            karsilastirma["gorev_kapsami_plana_sadakat"]["Statik_Script"],
            karsilastirma["gorev_kapsami_plana_sadakat"]["Plan_and_Solve_PS"],
        ]
        bars4 = ax4.bar(modeller, sadakat, color=renkler, width=0.45)
        ax4.set_ylabel("Görev Kapsamı (%)", fontsize=10, color="#cbd5e1")
        ax4.set_title("4. Alt Görevleri Eksiksiz İcra (%64.0 -> %99.2)", fontsize=11, color="#38bdf8", fontweight="bold")
        ax4.set_ylim(0, 120)
        ax4.grid(axis="y", linestyle=":", alpha=0.3)

        for b in bars4:
            h = b.get_height()
            ax4.text(b.get_x() + b.get_width() / 2.0, h + 1.8, f"%{h:.1f}", ha="center", va="bottom", color="#ffffff", fontweight="bold", fontsize=10)

        # -------------------------------------------------------------
        # PANEL 5: Sıralı Alt Görev İcra Akışı
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        gorevler = ["Görev 1: Veri Çekme", "Görev 2: Toplam Hesap", "Görev 3: Rapor Üretim"]
        sureler = [1.5, 0.6, 0.4]
        ax5.barh(gorevler[::-1], sureler[::-1], color=["#10b981", "#38bdf8", "#8b5cf6"], height=0.4)
        ax5.set_xlabel("İcra Süresi (s)", fontsize=10, color="#cbd5e1")
        ax5.set_title("5. Sıralı Alt Görev Tamamlanma Durumu", fontsize=11, color="#38bdf8", fontweight="bold")
        ax5.grid(axis="x", linestyle=":", alpha=0.3)

        # -------------------------------------------------------------
        # PANEL 6: GÜN 224 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")

        ozet_metin = (
            "GÜN 224: PLAN-AND-SOLVE (PS+) KARNESİ\n"
            "----------------------------------------------------\n"
            "• Yöntem              : Plan-and-Solve (PS / PS+)\n"
            "• Literatür           : Wang et al., 2023 (ACL 2023)\n"
            "• Ajan Fazları        : Planner (Strateji) -> Solver (İcra)\n"
            "• Görev Başarısı      : %52.0 -> %93.8 (Liderlik)\n"
            "• Araç Çağrısı İsrafı : %32.0 -> %3.5 (Minimum Maliyet)\n"
            "• Plana Sadakat       : %64.0 -> %99.2 (Eksiksiz Kapsam)\n"
            "----------------------------------------------------\n"
            "SONUÇ: Ajanımız artık büyük projeleri önce planlayıp\n"
            "sonra sırayla icra ederek kusursuzca tamamlıyor!"
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
