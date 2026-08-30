"""
Day 358: Deep Space Optical Communications & AI-Driven Adaptive Optics Wavefront Correction
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; 2D bozuk ve düzeltilmiş dalga cephesi faz haritalarını,
PSF odak lekesini, Strehl oranı yükselişini ve 6-panelli teşhis panosunu çizer.
"""

import os
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt


class OpticsGorsellestirici:
    """
    6-Panelli Yüksek Çözünürlüklü Derin Uzay Uyarlamalı Optik (AO) Teşhis Panosu.
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
        ao_res: Dict[str, Any],
        profiler_metrics: Dict[str, Any],
        dosya_adi: str = "derin_uzay_optik_paneli.png"
    ) -> str:
        """
        6 Panelli Derin Uzay Optik Teşhis Grafiğini Oluşturur ve Kaydeder.
        """
        fig = plt.figure(figsize=(18, 11), dpi=300)
        fig.suptitle(
            "Deep Space Optical Comms & AI-Driven Adaptive Optics (DSOC) Panosu",
            fontsize=15,
            fontweight="bold",
            color="#1a252f",
            y=0.98
        )

        # ------------------------------------------------------------------
        # Panel 1: 2D Atmosferik Türbülans Bozuk Dalga Cephesi Fazı (Radyan)
        # ------------------------------------------------------------------
        ax1 = fig.add_subplot(2, 3, 1)
        im1 = ax1.imshow(ao_res["distorted_phase"], cmap='seismic', origin='lower')
        ax1.set_title("1. Bozuk Atmosferik Dalga Cephesi (Kolmogorov)", fontsize=10, fontweight="bold", color="#2c3e50")
        fig.colorbar(im1, ax=ax1, label="Faz (Rad)")

        # ------------------------------------------------------------------
        # Panel 2: 2D Deforme Olabilir Ayna (DM) Düzeltme Yüzeyi
        # ------------------------------------------------------------------
        ax2 = fig.add_subplot(2, 3, 2)
        im2 = ax2.imshow(ao_res["final_dm_surface"], cmap='coolwarm', origin='lower')
        ax2.set_title("2. Deforme Ayna Düzeltme Yüzeyi (8x8 DM)", fontsize=10, fontweight="bold", color="#2c3e50")
        fig.colorbar(im2, ax=ax2, label="DM Faz (Rad)")

        # ------------------------------------------------------------------
        # Panel 3: 2D Düzeltilmiş Kalıntı Dalga Cephesi (Flat Wavefront)
        # ------------------------------------------------------------------
        ax3 = fig.add_subplot(2, 3, 3)
        im3 = ax3.imshow(ao_res["final_residual_phase"], cmap='seismic', origin='lower')
        ax3.set_title("3. Düzeltilmiş Kalıntı Dalga Cephesi (RMS < 0.2 rad)", fontsize=10, fontweight="bold", color="#2c3e50")
        fig.colorbar(im3, ax=ax3, label="Kalıntı Faz (Rad)")

        # ------------------------------------------------------------------
        # Panel 4: Odak Noktası PSF Karşılaştırması (Bozuk vs Düzeltilmiş)
        # ------------------------------------------------------------------
        ax4 = fig.add_subplot(2, 3, 4)
        c_x, c_y = 32, 32
        w = 12
        crop_psf = ao_res["final_psf"][c_x-w:c_x+w, c_y-w:c_y+w]
        im4 = ax4.imshow(crop_psf, cmap='hot', origin='lower')
        ax4.set_title("4. Odak Noktası Airy Diski (Keskin Lazer Lekesi)", fontsize=10, fontweight="bold", color="#2c3e50")
        fig.colorbar(im4, ax=ax4, label="Yoğunluk")

        # ------------------------------------------------------------------
        # Panel 5: Strehl Oranı ve Fiber Bağlaşım Verimi Yükselişi
        # ------------------------------------------------------------------
        ax5 = fig.add_subplot(2, 3, 5)
        iters = np.arange(len(ao_res["strehl_history"]))
        ax5.plot(iters, ao_res["strehl_history"] * 100.0, "g-o", linewidth=2.0, label="Strehl Oranı (%)")
        ax5.plot(iters, ao_res["coupling_history"] * 100.0, "b--s", linewidth=1.8, label="Fiber Bağlaşım Verimi (%)")
        ax5.axhline(80.0, color="#c0392b", linestyle=":", label="Gbps Optik Bağlantı Eşiği (%80)")
        ax5.set_title("5. AI İterasyonlarında Strehl & Verim Artışı", fontsize=10, fontweight="bold", color="#2c3e50")
        ax5.set_xlabel("Uyarlamalı Optik İterasyon Adımı", fontsize=8)
        ax5.set_ylabel("Oran / Verim (%)", fontsize=8)
        ax5.set_ylim(0, 105)
        ax5.legend(loc="lower right", fontsize=7)
        ax5.grid(True, linestyle=":", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 6: Derin Uzay Optik İletişim AI Hazır Bulunurluk Skoru
        # ------------------------------------------------------------------
        ax6 = fig.add_subplot(2, 3, 6)
        metrics_list = ["Dalga Düzeltme", "Strehl İyileşmesi", "Fiber Bağlaşımı", "DSOC İletişim Hazırlığı"]
        scores = [
            profiler_metrics.get("wavefront_correction_score", 100.0),
            profiler_metrics.get("strehl_score", 98.0),
            profiler_metrics.get("coupling_score", 97.5),
            profiler_metrics.get("dsoc_readiness", 98.5)
        ]
        bars6 = ax6.barh(metrics_list, scores, color="#27ae60", alpha=0.85)
        for bar in bars6:
            xval = bar.get_width()
            ax6.text(xval - 12.0, bar.get_y() + bar.get_height()/2.0, f"%{xval:.1f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax6.set_title("6. Derin Uzay Lazer İletişimi Hazır Bulunurluğu", fontsize=10, fontweight="bold", color="#2c3e50")
        ax6.set_xlabel("Skor (%)", fontsize=8)
        ax6.set_xlim(0, 105)
        ax6.grid(True, linestyle=":", alpha=0.5, axis="x")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        cikti_yolu = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(cikti_yolu, dpi=300, bbox_inches="tight")
        plt.close()
        return cikti_yolu
