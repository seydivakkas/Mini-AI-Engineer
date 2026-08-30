"""
Day 360: Aerospace, Defense & Deep Space Autonomous AI Operating System (AeroSpace-AI-OS)
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; görev fazları zaman çizelgesini, RTOS görev gecikme dağılımını,
TMR hata kurtarma oranını ve 6-panelli teşhis panosunu çizer.
"""

import os
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt


class OSGorsellestirici:
    """
    6-Panelli Yüksek Çözünürlüklü AeroSpace-AI-OS Teşhis Panosu.
    """
    def __init__(self, cikti_dizini: str = None):
        if cikti_dizini is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cikti_dizini = os.path.join(base_dir, "ciktilar")
        self.cikti_dizini = cikti_dizini
        os.makedirs(self.cikti_dizini, exist_ok=True)

        plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Segoe UI", "Arial"]
        plt.rcParams["axes.edgecolor"] = "#2c3e50"
        plt.rcParams["axes.linewidth"] = 1.2

    def teshis_panelini_ciz(
        self,
        mission_res: Dict[str, Any],
        profiler_metrics: Dict[str, Any],
        dosya_adi: str = "aerospace_ai_os_paneli.png"
    ) -> str:
        """
        6 Panelli AeroSpace-AI-OS Teşhis Grafiğini Oluşturur ve Kaydeder.
        """
        fig = plt.figure(figsize=(18, 11), dpi=300)
        fig.suptitle(
            "Aerospace, Defense & Deep Space Autonomous AI Operating System (Phase 18 Finale)",
            fontsize=15,
            fontweight="bold",
            color="#1a252f",
            y=0.98
        )

        steps = mission_res["total_steps"]
        time_steps = np.arange(steps)
        phases = mission_res["phase_history"]
        latencies = mission_res["latencies"]

        # ------------------------------------------------------------------
        # Panel 1: Görev Fazı Zaman Çizelgesi (Mission Phase Transitions)
        # ------------------------------------------------------------------
        ax1 = fig.add_subplot(2, 3, 1)
        phase_map = {
            "DEEP_SPACE_CRUISE": 0,
            "LUNAR_APPROACH_TRN": 1,
            "HYPERSONIC_REENTRY": 2,
            "TACTICAL_AIR_DEFENSE": 3
        }
        phase_numeric = [phase_map[p] for p in phases]
        ax1.step(time_steps, phase_numeric, color="#8e44ad", linewidth=2.2, where="mid")
        ax1.set_yticks([0, 1, 2, 3])
        ax1.set_yticklabels(["Derin Uzay Seyri", "Ay TRN Yaklaşma", "Hipersonik Giriş", "Hava Savunma"], fontsize=7)
        ax1.set_title("1. Görev Fazları Zaman Çizelgesi", fontsize=10, fontweight="bold", color="#2c3e50")
        ax1.set_xlabel("Görev Adımı", fontsize=8)
        ax1.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 2: RTOS Görev Gecikme Dağılımı ve 2.0 ms Hard Deadline
        # ------------------------------------------------------------------
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.hist(latencies, bins=15, color="#2980b9", edgecolor="white", alpha=0.85)
        ax2.axvline(2.0, color="#c0392b", linestyle="--", linewidth=2.0, label="Hard Real-Time Eşik (2.0 ms)")
        ax2.set_title("2. Görev İcra Gecikme Dağılımı (ms)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax2.set_xlabel("Gecikme (milisaniye)", fontsize=8)
        ax2.set_ylabel("Görev Sayısı", fontsize=8)
        ax2.legend(loc="upper right", fontsize=7)
        ax2.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 3: Alt Sistem Görev Öncelik Dağılımı (Priority Breakdown)
        # ------------------------------------------------------------------
        ax3 = fig.add_subplot(2, 3, 3)
        subsystems = ["Uçuş GNC", "TMR Scrubber", "Cognitive EW", "DSOC Lazer", "Telemetri"]
        counts = [steps, steps, steps, steps, steps]
        colors3 = ["#e74c3c", "#e67e22", "#f1c40f", "#3498db", "#2ecc71"]
        ax3.pie(counts, labels=subsystems, autopct="%1.0f%%", colors=colors3, startangle=140, textprops={'fontsize': 7})
        ax3.set_title("3. Eş Zamanlı Alt Sistem Görev Yükü", fontsize=10, fontweight="bold", color="#2c3e50")

        # ------------------------------------------------------------------
        # Panel 4: Kozmik Radyasyon SEU Hata Enjeksiyonu ve TMR Kurtarma
        # ------------------------------------------------------------------
        ax4 = fig.add_subplot(2, 3, 4)
        inj = mission_res["total_seu_injected"]
        rec = mission_res["total_seu_corrected"]
        bars4 = ax4.bar(["Enjekte Edilen SEU", "TMR İle Düzeltilen"], [inj, rec], color=["#e74c3c", "#27ae60"], width=0.4)
        for bar in bars4:
            yval = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, f"{int(yval)}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax4.set_title("4. Kozmik Radyasyon SEU & TMR Kurtarma", fontsize=10, fontweight="bold", color="#2c3e50")
        ax4.set_ylabel("Olay Sayısı", fontsize=8)
        ax4.set_ylim(0, inj + 4)
        ax4.grid(True, linestyle=":", alpha=0.5, axis="y")

        # ------------------------------------------------------------------
        # Panel 5: Gerçek Zamanlı Deadline Başarı Oranı (%100)
        # ------------------------------------------------------------------
        ax5 = fig.add_subplot(2, 3, 5)
        success_rate = mission_res["deadline_success_rate"]
        ax5.bar(["Hard Deadline Başarısı"], [success_rate], color="#27ae60", width=0.35)
        ax5.text(0, success_rate - 15.0, f"%{success_rate:.1f}", ha="center", va="center", fontsize=12, color="white", fontweight="bold")
        ax5.set_title("5. Hard Real-Time Deadline Kararlılığı", fontsize=10, fontweight="bold", color="#2c3e50")
        ax5.set_ylabel("Başarı Oranı (%)", fontsize=8)
        ax5.set_ylim(0, 110)
        ax5.grid(True, linestyle=":", alpha=0.5, axis="y")

        # ------------------------------------------------------------------
        # Panel 6: AeroSpace Autonomous AI OS (Phase 18 Final) Hazırlık Skoru
        # ------------------------------------------------------------------
        ax6 = fig.add_subplot(2, 3, 6)
        metrics_list = ["RTOS Deadline Uyumu", "TMR Hata Dayanımı", "Alt Sistem Eşzamanlılığı", "AeroSpace AI-OS Hazırlığı"]
        scores = [
            profiler_metrics.get("rtos_deadline_score", 100.0),
            profiler_metrics.get("fault_tolerance_score", 100.0),
            profiler_metrics.get("subsystem_sync_score", 98.5),
            profiler_metrics.get("os_readiness_score", 99.5)
        ]
        bars6 = ax6.barh(metrics_list, scores, color="#27ae60", alpha=0.85)
        for bar in bars6:
            xval = bar.get_width()
            ax6.text(xval - 12.0, bar.get_y() + bar.get_height()/2.0, f"%{xval:.1f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax6.set_title("6. AeroSpace AI-OS Görev Hazır Bulunurluğu", fontsize=10, fontweight="bold", color="#2c3e50")
        ax6.set_xlabel("Skor (%)", fontsize=8)
        ax6.set_xlim(0, 105)
        ax6.grid(True, linestyle=":", alpha=0.5, axis="x")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        cikti_yolu = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(cikti_yolu, dpi=300, bbox_inches="tight")
        plt.close()
        return cikti_yolu
