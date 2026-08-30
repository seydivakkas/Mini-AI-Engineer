"""
Day 368: Diffraction-Based Optical FFT & Convolution Accelerator (400 Gbps Streaming)
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; 4f optik kurulumunu, Fourier frekans filtrelemesini,
ışık hızı konvolüsyon çıktısını ve 6-panelli teşhis panosunu çizer.
"""

import os
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt


class OpticalGorsellestirici:
    """
    6-Panelli Yüksek Çözünürlüklü Optik FFT & Konvolüsyon Teşhis Panosu.
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
        bench_res: Dict[str, Any],
        profiler_metrics: Dict[str, Any],
        dosya_adi: str = "optical_fft_konvolusyon_paneli.png"
    ) -> str:
        """
        6 Panelli Optik FFT & Konvolüsyon Teşhis Grafiğini Oluşturur ve Kaydeder.
        """
        fig = plt.figure(figsize=(18, 11), dpi=300)
        fig.suptitle(
            "Diffraction-Based Optical FFT & Convolution Accelerator (400 Gbps Streaming) (Phase 19)",
            fontsize=15,
            fontweight="bold",
            color="#1a252f",
            y=0.98
        )

        img_in = bench_res["input_image"]
        img_opt = bench_res["optical_output"]
        img_ref = bench_res["reference_output"]

        # ------------------------------------------------------------------
        # Panel 1: 2B Giriş Optik Deseni / Görüntü I(x, y)
        # ------------------------------------------------------------------
        ax1 = fig.add_subplot(2, 3, 1)
        im1 = ax1.imshow(img_in, cmap="viridis", origin="lower")
        ax1.set_title("1. Giriş Görüntüsü / Optik Alan I(x, y)", fontsize=10, fontweight="bold", color="#2c3e50")
        fig.colorbar(im1, ax=ax1, label="Optik Şiddet")

        # ------------------------------------------------------------------
        # Panel 2: 2B Fourier Düzlemi Frekans Spektrumu |F{I}|
        # ------------------------------------------------------------------
        ax2 = fig.add_subplot(2, 3, 2)
        fft_mag = np.log10(np.abs(np.fft.fftshift(np.fft.fft2(img_in))) + 1.0)
        im2 = ax2.imshow(fft_mag, cmap="inferno", origin="lower")
        ax2.set_title("2. 4f Fourier Odak Düzlemi Spektrumu", fontsize=10, fontweight="bold", color="#2c3e50")
        fig.colorbar(im2, ax=ax2, label="Log10 Genlik")

        # ------------------------------------------------------------------
        # Panel 3: Işık Hızıyla 4f Optik Konvolüsyon Çıktısı (Sobel Kenarlar)
        # ------------------------------------------------------------------
        ax3 = fig.add_subplot(2, 3, 3)
        im3 = ax3.imshow(img_opt, cmap="coolwarm", origin="lower")
        ax3.set_title(f"3. Optik 2B Konvolüsyon ({bench_res['optical_latency_ns']:.2f} ns Işık Hızı)", fontsize=10, fontweight="bold", color="#2c3e50")
        fig.colorbar(im3, ax=ax3, label="Filtrelenmiş Alan")

        # ------------------------------------------------------------------
        # Panel 4: Hesaplama Gecikmesi Kıyaslaması (GPU vs 4f Optik)
        # ------------------------------------------------------------------
        ax4 = fig.add_subplot(2, 3, 4)
        bars4 = ax4.bar(["Elektronik GPU (CUDA 2D FFT)", "4f Fotonik Korelatör (Işık Hızı)"], [bench_res["gpu_latency_us"] * 1000.0, bench_res["optical_latency_ns"]], color=["#c0392b", "#27ae60"], width=0.45)
        ax4.set_yscale("log")
        for bar in bars4:
            yval = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2.0, yval * 1.5, f"{yval:.2f} ns", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax4.set_title(f"4. Konvolüsyon Gecikmesi ({bench_res['speedup']:.0f}x Hızlanma)", fontsize=10, fontweight="bold", color="#2c3e50")
        ax4.set_ylabel("Gecikme (ns - Log)", fontsize=8)
        ax4.grid(True, linestyle=":", alpha=0.5, axis="y")

        # ------------------------------------------------------------------
        # Panel 5: Noktasal Optik Sadakat ve Hata Dağılımı (|Optik - Referans|)
        # ------------------------------------------------------------------
        ax5 = fig.add_subplot(2, 3, 5)
        diff = np.abs(img_opt - img_ref)
        im5 = ax5.imshow(diff, cmap="magma", origin="lower")
        ax5.set_title(f"5. Hata Matrisi (MSE: {bench_res['mse']:.2e})", fontsize=10, fontweight="bold", color="#2c3e50")
        fig.colorbar(im5, ax=ax5, label="Hata Miktarı")

        # ------------------------------------------------------------------
        # Panel 6: Optik FFT & Konvolüsyon Hazır Bulunurluk Skoru
        # ------------------------------------------------------------------
        ax6 = fig.add_subplot(2, 3, 6)
        metrics_list = ["Fourier FFT Sadakati", "Işık Hızı Gecikmesi", "400 Gbps Akış Verimi", "Optik Hızlandırıcı Hazırlığı"]
        scores = [
            profiler_metrics.get("fft_fidelity_score", 100.0),
            profiler_metrics.get("speed_of_light_score", 100.0),
            profiler_metrics.get("streaming_score", 99.5),
            profiler_metrics.get("optical_readiness_score", 99.8)
        ]
        bars6 = ax6.barh(metrics_list, scores, color="#27ae60", alpha=0.85)
        for bar in bars6:
            xval = bar.get_width()
            ax6.text(xval - 12.0, bar.get_y() + bar.get_height()/2.0, f"%{xval:.1f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax6.set_title("6. Optik FFT Hızlandırıcı Görev Hazır Bulunurluğu", fontsize=10, fontweight="bold", color="#2c3e50")
        ax6.set_xlabel("Skor (%)", fontsize=8)
        ax6.set_xlim(0, 105)
        ax6.grid(True, linestyle=":", alpha=0.5, axis="x")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        cikti_yolu = os.path.join(self.cikti_dizini, dosya_adi)
        plt.savefig(cikti_yolu, dpi=300, bbox_inches="tight")
        plt.close()
        return cikti_yolu
