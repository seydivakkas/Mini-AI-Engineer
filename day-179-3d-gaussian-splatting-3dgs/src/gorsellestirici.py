"""
3D Gaussian Splatting (3DGS) Teşhis Panosu Görselleştirici Modülü (Day 179 - FAZ 9).
6 panelli NeRF vs 3DGS FPS Kıyaslaması, 3D Gauss Elipsoidleri, 2D Kovaryans Projeksiyonu, Diferansiyellenebilir Alfa Renderı, İcra İzi ve Özet Kartı.
"""

import os
import sys
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class GaussianGorsellestirici:
    """3D Gaussian Splatting (3DGS) Teşhis Panosu Üreticisi."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        rapor: Dict[str, Any],
        rendered_data: Optional[Dict[str, Any]] = None,
        mu_3d: Optional[np.ndarray] = None,
        kayit_yolu: str = "ciktilar/gaussian_splatting_3dgs_paneli.png",
    ):
        """6 panelli 3DGS teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(os.path.abspath(kayit_yolu)), exist_ok=True)

        fig = plt.figure(figsize=(23, 13.5))
        fig.suptitle(
            "GÜN 179 (FAZ 9): 3D Gaussian Splatting (3DGS): Gerçek Zamanlı (100+ FPS) Radyan ve Nokta Kümesi Renderı",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        # -------------------------------------------------------------
        # PANEL 1: 3DGS vs NeRF vs Instant-NGP FPS & PSNR Kıyaslaması
        # -------------------------------------------------------------
        ax1 = fig.add_subplot(2, 3, 1)
        yontemler = [item["yontem"].split("(")[0].strip() for item in rapor["karsilastirma"]]
        fpsler = [item["fps"] for item in rapor["karsilastirma"]]
        psnrler = [item["psnr"] for item in rapor["karsilastirma"]]

        ax1_twin = ax1.twinx()
        bar1 = ax1.bar(yontemler, fpsler, color="#4e73df", alpha=0.7, width=0.45, label="Render Hızı (FPS)")
        line1 = ax1_twin.plot(yontemler, psnrler, color="#e74a3b", marker="o", linewidth=2.5, label="Kalite (PSNR dB)")

        ax1.set_title("1. 3DGS vs NeRF: Gerçek Zamanlı Render Hızı (FPS)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Render Hızı (FPS - Log Skala)", color="#4e73df")
        ax1_twin.set_ylabel("PSNR Kalitesi (dB)", color="#e74a3b")
        ax1.set_yscale("log")
        ax1.grid(axis="y", linestyle="--", alpha=0.7)

        for bar in bar1:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, yval * 1.15, f"{yval:.1f} FPS", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        # -------------------------------------------------------------
        # PANEL 2: 3D Gauss Nokta Kümesi ve Elipsoid Saçılımı
        # -------------------------------------------------------------
        ax2 = fig.add_subplot(2, 3, 2, projection="3d")
        if mu_3d is None:
            np.random.seed(42)
            mu_3d = np.random.randn(120, 3) * 0.45

        colors_pts = (mu_3d - mu_3d.min()) / (mu_3d.max() - mu_3d.min() + 1e-8)
        ax2.scatter(mu_3d[:, 0], mu_3d[:, 1], mu_3d[:, 2], c=colors_pts, s=40, alpha=0.85, edgecolors="k", linewidth=0.5)
        ax2.set_title("2. 3D Gauss Nokta Kümesi (mu in R^3)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("X (metre)")
        ax2.set_ylabel("Y (metre)")
        ax2.set_zlabel("Z (Derinlik)")
        ax2.view_init(elev=22, azim=45)

        # -------------------------------------------------------------
        # PANEL 3: 2D Ekran Düzlemine Kovaryans Projeksiyonu (Splatting)
        # -------------------------------------------------------------
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.set_xlim(0, 64)
        ax3.set_ylim(64, 0)

        if rendered_data is not None and "mu_2d" in rendered_data:
            mu_2d_np = rendered_data["mu_2d"].detach().cpu().numpy()
            N_pts = min(len(mu_2d_np), 50)
            for i in range(N_pts):
                u, v = mu_2d_np[i]
                if 0 <= u <= 64 and 0 <= v <= 64:
                    ellipse = patches.Ellipse(
                        (u, v), width=6.0, height=3.5, angle=i * 15,
                        facecolor=plt.cm.viridis(i / max(N_pts, 1)), alpha=0.5, edgecolor="black", linewidth=0.8
                    )
                    ax3.add_patch(ellipse)
                    ax3.plot(u, v, "ro", markersize=2)
        else:
            for i in range(30):
                u = np.random.uniform(10, 54)
                v = np.random.uniform(10, 54)
                ellipse = patches.Ellipse(
                    (u, v), width=np.random.uniform(4, 10), height=np.random.uniform(2, 6),
                    angle=np.random.uniform(0, 180), facecolor=plt.cm.plasma(i / 30.0), alpha=0.45, edgecolor="black", linewidth=0.5
                )
                ax3.add_patch(ellipse)
                ax3.plot(u, v, "ko", markersize=2)

        ax3.set_title("3. 2D Ekran Kovaryans Projeksiyonu (Sigma')", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Piksel U Koordinatı")
        ax3.set_ylabel("Piksel V Koordinatı")
        ax3.grid(True, linestyle="--", alpha=0.6)

        # -------------------------------------------------------------
        # PANEL 4: Diferansiyellenebilir Alfa Render Edilmiş Görüntü
        # -------------------------------------------------------------
        ax4 = fig.add_subplot(2, 3, 4)
        if rendered_data is not None and "image" in rendered_data:
            img_np = rendered_data["image"].detach().cpu().numpy()
            img_np = np.clip(img_np, 0.0, 1.0)
            ax4.imshow(img_np)
        else:
            dummy_img = np.zeros((64, 64, 3))
            dummy_img[:, :, 0] = 0.2
            dummy_img[:, :, 1] = 0.6
            dummy_img[:, :, 2] = 0.8
            ax4.imshow(dummy_img)

        ax4.set_title("4. Diferansiyellenebilir Render (Alpha Blending)", fontsize=12, fontweight="bold")
        ax4.axis("on")
        ax4.set_xlabel("Genişlik (64 px)")
        ax4.set_ylabel("Yükseklik (64 px)")

        # -------------------------------------------------------------
        # PANEL 5: 3DGS Mimarisi ve İcra İzi
        # -------------------------------------------------------------
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.axis("off")
        ax5.set_title("5. 3DGS Matematiksel Mimarisi & İcra İzi", fontsize=12, fontweight="bold", pad=10)

        sema_metni = (
            "====================================================\n"
            "       3D GAUSSIAN SPLATTING RASTERIZER LOG         \n"
            "====================================================\n"
            f"SAHNE ADI        : {rapor.get('sahne', '3DGS Real-time Scene')}\n"
            f"RENDER HIZI      : {rapor.get('karsilastirma', [{}, {}, {'fps': 145.0}])[2].get('fps', 145.0)} FPS (Gerçek Zamanlı!)\n"
            f"PSNR SKORU       : {rapor.get('karsilastirma', [{}, {}, {'psnr': 34.5}])[2].get('psnr', 34.5)} dB (Fotogerçekçi)\n"
            "----------------------------------------------------\n"
            "3DGS MATEMATİKSEL İŞLEM ADIMLARI:\n"
            "1. 3D Gauss Tanımı  : G(x) = exp(-0.5 * x^T * Sigma^{-1} * x)\n"
            "2. Kovaryans Matrisi: Sigma = R * S * S^T * R^T (Simetrik Yarı-Tanımlı)\n"
            "3. EWA Projeksiyonu : Sigma' = J * W * Sigma * W^T * J^T + 0.3*I\n"
            "4. Radix Derinlik   : Sort(depths) (Önden arkaya sıralama)\n"
            "5. Tile Rasterizer  : C(p) = sum c_i * alpha_i * T_i\n"
            "----------------------------------------------------\n"
            f"HIZ ARTIŞI       : {rapor.get('fps_artis_kati', '~414x Hızlanma')}\n"
            "===================================================="
        )

        ax5.text(
            0.02, 0.5, sema_metni,
            fontsize=7.2,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 6: GÜN 179 Özet Kartı
        # -------------------------------------------------------------
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis("off")
        ax6.set_title("6. GÜN 179 ÖZET KARTI", fontsize=12, fontweight="bold", pad=10)

        ozet_metin = (
            "====================================================\n"
            "   DAY 179 SUMMARY: 3D GAUSSIAN SPLATTING (3DGS)    \n"
            "====================================================\n"
            "• Modül              : FAZ 9 (Çok Modlu Modeller)\n"
            "• Temel Mimari       : Kerbl et al. (2023) 3DGS\n"
            "• Temsil Biçimi      : Açık Elipsoidler (mu, s, q, alpha, SH)\n"
            "• Render Motoru      : Diferansiyellenebilir Tile Rasterizer\n"
            "• Hızlanma Oranı     : ~414x (0.35 FPS -> 145 FPS)\n"
            "----------------------------------------------------\n"
            "TEMEL KAZANIMLAR:\n"
            "  1. NeRF'ün ağır ışın takibi yerine açık elipsoid rasterizasyonu\n"
            "  2. EWA Kovaryans İzdüşümü ile perspektif Jacobian dönüşümü\n"
            "  3. GPU tile-tabanlı sıralama ve hızlı alfa birleştirme (Over)\n"
            "  4. Adaptif Yoğunluk Kontrolü (Bölme ve Klonlama) mantığı\n"
            "====================================================\n"
            "   SIRADAKİ GÜN: Gün 180 (Multimodal Omni Benchmark Suite)\n"
            "===================================================="
        )

        ax6.text(
            0.02, 0.5, ozet_metin,
            fontsize=7.5,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#d4edda", edgecolor="#28a745", lw=1.5),
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ 3DGS Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
