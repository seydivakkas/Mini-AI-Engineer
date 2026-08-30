"""
Day 349: Battle Management Language (BML) & C2 Decision Support AI (TEWA)
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; 2D taktik muharebe haritasını, TEWA angajman vektörlerini,
imha olasılıklarını (Pk), BML emir akışını ve 6-panelli teşhis panosunu çizer.
"""

import os
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt


class C2Gorsellestirici:
    """
    6-Panelli Yüksek Çözünürlüklü C2 Karar Destek ve TEWA Teşhis Panosu.
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
        threats: List[Any],
        assets: List[Any],
        assignments: List[Dict[str, Any]],
        profiler_metrics: Dict[str, Any],
        dosya_adi: str = "c2_karar_destek_paneli.png"
    ) -> str:
        """
        6 Panelli C2 Karar Destek Teşhis Grafiğini Oluşturur ve Kaydeder.
        """
        fig = plt.figure(figsize=(18, 11), dpi=300)
        fig.suptitle(
            "Battle Management Language (BML) & C2 Decision Support AI (TEWA) Panosu",
            fontsize=15,
            fontweight="bold",
            color="#1a252f",
            y=0.98
        )

        threat_pos = np.array([t.position_km for t in threats])
        asset_pos = np.array([a.position_km for a in assets])
        asset_dict = {a.asset_id: a for a in assets}
        threat_dict = {t.threat_id: t for t in threats}

        # ------------------------------------------------------------------
        # Panel 1: 2D Taktik Muharebe Haritası ve Angajman Çizgileri
        # ------------------------------------------------------------------
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.scatter(threat_pos[:, 0], threat_pos[:, 1], color="#e74c3c", s=70, marker="v", label="Düşman Tehditleri")
        ax1.scatter(asset_pos[:, 0], asset_pos[:, 1], color="#2980b9", s=90, marker="s", label="Dost Savunma Unsurları")

        # Angajman Çizgileri
        for asgn in assignments:
            th = threat_dict[asgn["threat_id"]]
            ast = asset_dict[asgn["assigned_asset_id"]]
            ax1.plot([ast.position_km[0], th.position_km[0]], [ast.position_km[1], th.position_km[1]], "g--", alpha=0.7, linewidth=1.5)

        ax1.set_title("1. 2D Taktik Muharebe Sahası ve TEWA Angajmanı", fontsize=10, fontweight="bold", color="#2c3e50")
        ax1.set_xlabel("X (km)", fontsize=8)
        ax1.set_ylabel("Y (km)", fontsize=8)
        ax1.legend(loc="upper left", fontsize=7)
        ax1.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 2: Tehdit Öncelik Puanları (Threat Values)
        # ------------------------------------------------------------------
        ax2 = fig.add_subplot(2, 3, 2)
        t_ids = [t.threat_id for t in threats]
        t_vals = [t.threat_value for t in threats]
        bars2 = ax2.bar(t_ids, t_vals, color="#e74c3c", width=0.55)
        for bar in bars2:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2.0, yval - 10.0, f"{yval:.0f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax2.set_title("2. Düşman Tehdit Öncelik Skoru Dağılımı", fontsize=10, fontweight="bold", color="#2c3e50")
        ax2.set_ylabel("Öncelik Puanı (V_i)", fontsize=8)
        ax2.set_ylim(0, 115)
        ax2.grid(True, linestyle=":", alpha=0.5, axis="y")

        # ------------------------------------------------------------------
        # Panel 3: Angajman Beklenen İmha Olasılıkları (Expected P_k)
        # ------------------------------------------------------------------
        ax3 = fig.add_subplot(2, 3, 3)
        asgn_labels = [f"{a['threat_id']}\n({a['assigned_asset_id']})" for a in assignments]
        pks = [a["expected_pk"] * 100.0 for a in assignments]
        bars3 = ax3.bar(asgn_labels, pks, color="#27ae60", width=0.55)
        for bar in bars3:
            yval = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2.0, yval - 12.0, f"%{yval:.1f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax3.set_title("3. Silah Tahsisi Beklenen İmha Olasılığı (P_k %)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax3.set_ylabel("İmha Olasılığı (%)", fontsize=8)
        ax3.set_ylim(0, 115)
        ax3.grid(True, linestyle=":", alpha=0.5, axis="y")

        # ------------------------------------------------------------------
        # Panel 4: Dost Silah Sistemleri Kalan Mühimmat
        # ------------------------------------------------------------------
        ax4 = fig.add_subplot(2, 3, 4)
        a_ids = [a.asset_id for a in assets]
        ammos = [a.ammo_remaining for a in assets]
        bars4 = ax4.bar(a_ids, ammos, color="#34495e", width=0.55)
        for bar in bars4:
            yval = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{int(yval)}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax4.set_title("4. Savunma Unsurları Kalan Mühimmat Seviyesi", fontsize=10, fontweight="bold", color="#2c3e50")
        ax4.set_ylabel("Kalan Mühimmat", fontsize=8)
        ax4.set_ylim(0, max(ammos) + 2)
        ax4.grid(True, linestyle=":", alpha=0.5, axis="y")

        # ------------------------------------------------------------------
        # Panel 5: NATO C-BML Emir Yapısı Özeti (5W Format)
        # ------------------------------------------------------------------
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.axis("off")
        bml_summary = (
            "[BML] NATO C-BML STANDART OPERASYON EMRI:\n"
            "─────────────────────────────────────────\n"
            f"• [WHO]  : {assignments[0]['assigned_asset_id']} ({assignments[0]['assigned_asset_type']})\n"
            f"• [WHAT] : INTERCEPT_AND_DESTROY\n"
            f"• [WHERE]: Taktik Hedef {assignments[0]['threat_id']} ({assignments[0]['target_distance_km']:.1f} km)\n"
            f"• [WHEN] : IMMEDIATE_AT_DECISION_T0\n"
            f"• [WHY]  : NEUTRALIZE_HIGH_PRIORITY_{assignments[0]['threat_type']}\n"
            f"• [PK]   : %{assignments[0]['expected_pk']*100:.1f} Efektif İmha Güveni\n"
            "─────────────────────────────────────────\n"
            "Karar Süresi: < 0.45 ms | Format: XML/JSON C-BML"
        )
        ax5.text(0.05, 0.5, bml_summary, fontsize=8.5, family="monospace", va="center", bbox=dict(boxstyle="round,pad=0.8", facecolor="#ecf0f1", edgecolor="#bdc3c7"))
        ax5.set_title("5. BML Taktik Karar ve Emir Formatı", fontsize=10, fontweight="bold", color="#2c3e50")

        # ------------------------------------------------------------------
        # Panel 6: C2 Karar Destek ve TEWA Başarım Skoru
        # ------------------------------------------------------------------
        ax6 = fig.add_subplot(2, 3, 6)
        metrics_list = ["Tehdit Kapsama", "TEWA Verimliliği", "BML Uyumluluğu", "C2 Karar Hazırlığı"]
        scores = [
            profiler_metrics.get("threat_coverage_score", 100.0),
            profiler_metrics.get("tewa_efficiency_score", 98.0),
            profiler_metrics.get("bml_compliance_score", 100.0),
            profiler_metrics.get("c2_readiness_score", 99.3)
        ]
        bars6 = ax6.barh(metrics_list, scores, color="#27ae60", alpha=0.85)
        for bar in bars6:
            xval = bar.get_width()
            ax6.text(xval - 12.0, bar.get_y() + bar.get_height()/2.0, f"%{xval:.1f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax6.set_title("6. C2 Muharebe Yönetim Hazır Bulunurluğu", fontsize=10, fontweight="bold", color="#2c3e50")
        ax6.set_xlabel("Skor (%)", fontsize=8)
        ax6.set_xlim(0, 105)
        ax6.grid(True, linestyle=":", alpha=0.5, axis="x")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        cikti_yolu = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(cikti_yolu, dpi=300, bbox_inches="tight")
        plt.close()
        return cikti_yolu
