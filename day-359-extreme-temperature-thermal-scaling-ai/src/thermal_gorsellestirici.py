"""
Day 359: Extreme-Temperature Adaptive Neural Scaling & Dynamic Voltage/Frequency Scaling (DVFS)
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; çip sıcaklık eğrilerini, dinamik frekans değişimini,
güç tasarrufunu ve 6-panelli teşhis panosunu çizer.
"""

import os
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt


class ThermalGorsellestirici:
    """
    6-Panelli Yüksek Çözünürlüklü Ekstrem Termal Yönetim Teşhis Panosu.
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
        flight_res: Dict[str, Any],
        profiler_metrics: Dict[str, Any],
        dosya_adi: str = "termal_olcekleme_paneli.png"
    ) -> str:
        """
        6 Panelli Termal Ölçekleme Teşhis Grafiğini Oluşturur ve Kaydeder.
        """
        fig = plt.figure(figsize=(18, 11), dpi=300)
        fig.suptitle(
            "Extreme-Temperature Adaptive Neural Scaling & DVFS Panosu",
            fontsize=15,
            fontweight="bold",
            color="#1a252f",
            y=0.98
        )

        time_axis = flight_res["time_axis"]
        t_amb = flight_res["t_ambient_profile"]
        unm_die = flight_res["unmanaged_t_die"]
        ai_die = flight_res["ai_t_die"]
        ai_power = flight_res["ai_power"]
        ai_clocks = flight_res["ai_clocks"]
        ai_acc = flight_res["ai_accuracies"]

        # ------------------------------------------------------------------
        # Panel 1: Dış Ortam ve Çip Sıcaklığı Karşılaştırması (°C)
        # ------------------------------------------------------------------
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.plot(time_axis, t_amb, "k:", linewidth=1.5, label="Dış Ortam Sıcaklığı (°C)")
        ax1.plot(time_axis, unm_die, "r--", linewidth=1.8, label="Sabit Model Çip Isısı (Aşırı Isınma)")
        ax1.plot(time_axis, ai_die, "g-", linewidth=2.2, label="AI Termal Korumalı Çip Isısı")
        ax1.axhline(105.0, color="#c0392b", linestyle="--", label="Kritik Donanım İmha Limiti (105°C)")
        ax1.set_title("1. Hipersonik Isınmada Çip Sıcaklık Evrimi (°C)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax1.set_xlabel("Zaman (saniye)", fontsize=8)
        ax1.set_ylabel("Sıcaklık (°C)", fontsize=8)
        ax1.legend(loc="upper left", fontsize=6)
        ax1.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 2: Dinamik Frekans Yönetimi (DVFS Clock GHz)
        # ------------------------------------------------------------------
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.step(time_axis, ai_clocks, color="#2980b9", linewidth=2.0, where="post", label="İşlemci Saat Frekansı")
        ax2.set_title("2. Sıcaklığa Duyarlı DVFS Frekans Kısma (GHz)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax2.set_xlabel("Zaman (saniye)", fontsize=8)
        ax2.set_ylabel("Saat Frekansı (GHz)", fontsize=8)
        ax2.set_ylim(0.2, 1.4)
        ax2.legend(loc="upper right", fontsize=7)
        ax2.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 3: Toplam Aviyonik Güç Tüketimi (Watt)
        # ------------------------------------------------------------------
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.plot(time_axis, ai_power, color="#d35400", linewidth=2.0, label="Toplam Güç (Dinamik + Kaçak)")
        ax3.set_title("3. Isıl Güç Yayılımı (Watt)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax3.set_xlabel("Zaman (saniye)", fontsize=8)
        ax3.set_ylabel("Güç (Watt)", fontsize=8)
        ax3.legend(loc="upper right", fontsize=7)
        ax3.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 4: Elastik Nöral Ağ Doğruluk ve Görev Kararlılığı
        # ------------------------------------------------------------------
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.plot(time_axis, ai_acc * 100.0, color="#8e44ad", linewidth=2.0, label="AI Çıkarım Doğruluğu (%)")
        ax4.axhline(85.0, color="#f39c12", linestyle=":", label="Minimum Uçuş Güvenlik Eşiği (%85)")
        ax4.set_title("4. Termal Mod Değişiminde Model Doğruluğu (%)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax4.set_xlabel("Zaman (saniye)", fontsize=8)
        ax4.set_ylabel("Doğruluk (%)", fontsize=8)
        ax4.set_ylim(80, 102)
        ax4.legend(loc="lower right", fontsize=7)
        ax4.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 5: Maksimum Çip Sıcaklık Karşılaştırması
        # ------------------------------------------------------------------
        ax5 = fig.add_subplot(2, 3, 5)
        max_unm = float(np.max(unm_die))
        max_ai = float(np.max(ai_die))
        categories = ["Sabit Model (Yönetilmeyen)", "AI Termal Ölçekleme (Bizim)"]
        vals = [max_unm, max_ai]
        bars5 = ax5.bar(categories, vals, color=["#c0392b", "#27ae60"], width=0.45)
        for bar in bars5:
            yval = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f} °C", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax5.set_title("5. Zirve Çip Sıcaklığı Karşılaştırması (°C)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax5.set_ylabel("Maksimum Sıcaklık (°C)", fontsize=8)
        ax5.set_ylim(0, max_unm * 1.2 + 5)
        ax5.grid(True, linestyle=":", alpha=0.5, axis="y")

        # ------------------------------------------------------------------
        # Panel 6: Ekstrem Termal AI Hazır Bulunurluk Skoru
        # ------------------------------------------------------------------
        ax6 = fig.add_subplot(2, 3, 6)
        metrics_list = ["Aşırı Isınma Önleme", "DVFS Güç Tasarrufu", "Model Elastikliği", "Termal Hayatta Kalma"]
        scores = [
            profiler_metrics.get("overheat_prevention_score", 100.0),
            profiler_metrics.get("power_savings_score", 98.0),
            profiler_metrics.get("elastic_scaling_score", 97.5),
            profiler_metrics.get("thermal_survival_readiness", 98.5)
        ]
        bars6 = ax6.barh(metrics_list, scores, color="#27ae60", alpha=0.85)
        for bar in bars6:
            xval = bar.get_width()
            ax6.text(xval - 12.0, bar.get_y() + bar.get_height()/2.0, f"%{xval:.1f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax6.set_title("6. Ekstrem Termal AI Hazır Bulunurluğu", fontsize=10, fontweight="bold", color="#2c3e50")
        ax6.set_xlabel("Skor (%)", fontsize=8)
        ax6.set_xlim(0, 105)
        ax6.grid(True, linestyle=":", alpha=0.5, axis="x")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        cikti_yolu = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(cikti_yolu, dpi=300, bbox_inches="tight")
        plt.close()
        return cikti_yolu
