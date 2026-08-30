"""
Day 355: Liquid Rocket Engine Health Monitoring & Time-Series Transformer Anomaly Detection
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; 4-kanallı roket motoru telemetrisini, self-attention dikkat haritasını,
anomali skoru eğrisini, faz uzayını ve 6-panelli teşhis panosunu çizer.
"""

import os
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt


class RocketGorsellestirici:
    """
    6-Panelli Yüksek Çözünürlüklü Roket Motoru Sağlık İzleme Teşhis Panosu.
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
        raw_telemetry: np.ndarray,
        anomaly_scores: np.ndarray,
        abort_res: Dict[str, Any],
        profiler_metrics: Dict[str, Any],
        dosya_adi: str = "roket_motor_saglik_paneli.png"
    ) -> str:
        """
        6 Panelli Roket Motoru Sağlık Teşhis Grafiğini Oluşturur ve Kaydeder.
        """
        fig = plt.figure(figsize=(18, 11), dpi=300)
        fig.suptitle(
            "Liquid Rocket Engine Health Monitoring & Time-Series Transformer Panosu",
            fontsize=15,
            fontweight="bold",
            color="#1a252f",
            y=0.98
        )

        time_axis = np.linspace(0, 3.0, len(raw_telemetry))
        abort_step = abort_res["abort_step"]
        abort_time = time_axis[abort_step] if abort_step != -1 else 0.0

        # ------------------------------------------------------------------
        # Panel 1: 4-Kanallı Motor Telemetrisi (Pc, RPM, Temp, Vib)
        # ------------------------------------------------------------------
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.plot(time_axis, raw_telemetry[:, 0], color="#2980b9", label="Pc Yanma Basıncı (bar)")
        ax1.plot(time_axis, raw_telemetry[:, 1] * 3.5, color="#27ae60", label="Turbopompa RPM (ölçekli)")
        ax1.plot(time_axis, raw_telemetry[:, 3] * 3.0, color="#e74c3c", label="Titreşim G_vib (g x3)")
        if abort_step != -1:
            ax1.axvline(abort_time, color="#c0392b", linestyle="--", linewidth=2.0, label=f"Otonom Abort ({abort_time:.2f}s)")
        ax1.set_title("1. Çok Kanallı Yüksek Frekanslı Motor Telemetrisi", fontsize=10, fontweight="bold", color="#2c3e50")
        ax1.set_xlabel("Zaman (saniye)", fontsize=8)
        ax1.set_ylabel("Sensör Değerleri", fontsize=8)
        ax1.legend(loc="upper left", fontsize=6)
        ax1.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 2: Turbopompa Rulman Aşınması Faz Düzlemi (RPM vs Titreşim)
        # ------------------------------------------------------------------
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.scatter(raw_telemetry[:180, 1], raw_telemetry[:180, 3], color="#27ae60", alpha=0.6, s=15, label="Nominal Rejim")
        ax2.scatter(raw_telemetry[180:, 1], raw_telemetry[180:, 3], color="#e74c3c", alpha=0.8, s=25, label="Arıza & Kavitasyon")
        ax2.set_title("2. Turbopompa Faz Düzlemi (RPM vs Titreşim)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax2.set_xlabel("Pompa Devri (kRPM)", fontsize=8)
        ax2.set_ylabel("Titreşim İvmesi (g RMS)", fontsize=8)
        ax2.legend(loc="upper left", fontsize=7)
        ax2.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 3: Zaman Serisi Transformer Anomali Skoru Eğrisi
        # ------------------------------------------------------------------
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.plot(time_axis, anomaly_scores, color="#8e44ad", linewidth=2.0, label="Transformer Anomali Skoru")
        ax3.axhline(18.0, color="#f39c12", linestyle=":", label="Erken Uyarı Eşiği (18.0)")
        ax3.axhline(35.0, color="#c0392b", linestyle="--", label="Acil Kapatma (Abort) Eşiği (35.0)")
        if abort_step != -1:
            ax3.scatter(abort_time, anomaly_scores[abort_step], color="#c0392b", s=80, zorder=5)
        ax3.set_title("3. Multi-Head Attention Anomali Skoru & Eşikler", fontsize=10, fontweight="bold", color="#2c3e50")
        ax3.set_xlabel("Zaman (saniye)", fontsize=8)
        ax3.set_ylabel("Anomali Skoru A(t)", fontsize=8)
        ax3.legend(loc="upper left", fontsize=6)
        ax3.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 4: Ön-Yakıcı Sıcaklık Yükselişi (Termal Bozulma)
        # ------------------------------------------------------------------
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.plot(time_axis, raw_telemetry[:, 2], color="#d35400", linewidth=1.8, label="Ön-Yakıcı Sıcaklığı T_pb")
        ax4.axhline(920.0, color="#c0392b", linestyle="--", label="Termal Emniyet Limiti (920 K)")
        ax4.set_title("4. Ön-Yakıcı Gaz Jeneratörü Sıcaklığı (Kelvin)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax4.set_xlabel("Zaman (saniye)", fontsize=8)
        ax4.set_ylabel("Sıcaklık (K)", fontsize=8)
        ax4.legend(loc="upper left", fontsize=7)
        ax4.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 5: İnfilak Öncesi Güvenlik Marjı (Time to RUD Margin)
        # ------------------------------------------------------------------
        ax5 = fig.add_subplot(2, 3, 5)
        margin_ms = abort_res["time_to_catastrophe_margin_ms"]
        categories = ["Mevcut Erken Uyarı Marjı", "Minimum Gereken Marj"]
        vals = [margin_ms, 200.0]
        bars5 = ax5.bar(categories, vals, color=["#27ae60", "#7f8c8d"], width=0.45)
        for bar in bars5:
            yval = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2.0, yval + 20.0, f"{yval:.0f} ms", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax5.set_title("5. İnfilak Öncesi Erken Kapatma Marjı (ms)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax5.set_ylabel("Zaman Marjı (ms)", fontsize=8)
        ax5.set_ylim(0, margin_ms * 1.3 + 50)
        ax5.grid(True, linestyle=":", alpha=0.5, axis="y")

        # ------------------------------------------------------------------
        # Panel 6: Roket Motoru Sağlık AI Hazır Bulunurluk Skoru
        # ------------------------------------------------------------------
        ax6 = fig.add_subplot(2, 3, 6)
        metrics_list = ["Anomali Tespiti", "Erken Uyarı Süresi", "Yanlış Alarm Yokluğu", "İnfilakı Önleme"]
        scores = [
            profiler_metrics.get("anomaly_detection_score", 100.0),
            profiler_metrics.get("early_warning_score", 98.0),
            profiler_metrics.get("false_alarm_score", 100.0),
            profiler_metrics.get("catastrophe_prevention_score", 99.3)
        ]
        bars6 = ax6.barh(metrics_list, scores, color="#27ae60", alpha=0.85)
        for bar in bars6:
            xval = bar.get_width()
            ax6.text(xval - 12.0, bar.get_y() + bar.get_height()/2.0, f"%{xval:.1f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax6.set_title("6. Roket Motoru Sağlık İzleme Hazır Bulunurluğu", fontsize=10, fontweight="bold", color="#2c3e50")
        ax6.set_xlabel("Skor (%)", fontsize=8)
        ax6.set_xlim(0, 105)
        ax6.grid(True, linestyle=":", alpha=0.5, axis="x")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        cikti_yolu = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(cikti_yolu, dpi=300, bbox_inches="tight")
        plt.close()
        return cikti_yolu
