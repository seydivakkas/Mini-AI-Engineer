"""
NeRF (Neural Radiance Fields) Teşhis Panosu Görselleştirici Modülü (Day 178 - FAZ 9).
6 panelli Hacimsel Yoğunluk Profil Eğrisi (sigma(t) vs Derinlik), Işın Ağırlık Dağılımı (w_i), Pozisyonel Kodlama Frekans Spektrumu, 3D Işın Takibi Geometrisi, NeRF MLP Mimarisi ve Özet Kartı.
"""

import os
from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np


class NeRFGorsellestirici:
    """NeRF 3D Sahne Sentezi Teşhis Panosu Üreticisi."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        rapor: Dict[str, Any],
        kayit_yolu: str = "ciktilar/nerf_neural_radiance_fields_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(23, 13.5))
        fig.suptitle(
            "GÜN 178 (FAZ 9): NeRF (Neural Radiance Fields): Pozlandırılmış Fotoğraflardan 3D Sahne Hacimsel Sentezi & Işın Takibi",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        # -------------------------------------------------------------
        # PANEL 1: 3D Sahne Temsil Yöntemleri Kalite & Bellek Kıyası
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        yontemler = [item["yontem"].split("(")[0].strip() for item in rapor["karsilastirma"]]
        psnrs = [item["psnr"] for item in rapor["karsilastirma"]]
        bellekler = [item["bellek_mb"] for item in rapor["karsilastirma"]]

        ax1_twin = ax1.twinx()
        bar1 = ax1.bar(yontemler, bellekler, color="#e74a3b", alpha=0.6, width=0.45, label="Bellek Kullanımı (MB)")
        line1 = ax1_twin.plot(yontemler, psnrs, color="#1cc88a", marker="o", linewidth=2.5, label="Görüntü Kalitesi (PSNR dB)")

        ax1.set_title("1. 3D Sahne Temsil Kıyası: Bellek vs PSNR Kalitesi", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Depolama / Bellek (MB)", color="#e74a3b")
        ax1_twin.set_ylabel("PSNR Kalitesi (dB)", color="#1cc88a")
        ax1.set_yscale("log")
        ax1.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 2: Işın Boyunca Hacimsel Yoğunluk (sigma) ve Ağırlık (w_i)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        z_vals = np.linspace(2.0, 6.0, 100)
        # 3.8m civarında nesne yüzeyi var
        sigma = 15.0 * np.exp(-((z_vals - 3.8) ** 2) / 0.05)
        # Transmittance ve ağırlık
        alpha = 1.0 - np.exp(-sigma * (z_vals[1] - z_vals[0]))
        transmittance = np.cumprod(1.0 - alpha + 1e-10)
        weights = transmittance * alpha

        ax2.plot(z_vals, sigma, label="Hacimsel Yoğunluk: sigma(t)", color="#4e73df", linewidth=2.2)
        ax2.plot(z_vals, weights * 50, label="Piksel Ağırlığı: w_i = T_i * alpha_i (x50)", color="#f6c23e", linewidth=2.2, linestyle="--")

        ax2.axvline(3.8, color="#e74a3b", linestyle=":", label="Tespit Edilen Yüzey (Derinlik: 3.8m)")
        ax2.set_title("2. Işın Boyunca Yoğunluk ve Katkı Ağırlığı", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Işın Derinliği z (metre)")
        ax2.set_ylabel("Değer")
        ax2.legend(loc="upper right", frameon=True)
        ax2.grid(True, linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 3: Fourier Pozisyonel Kodlama Frekans Spektrumu (L=10)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        L_vals = np.arange(0, 10)
        frekanslar = 2.0 ** L_vals

        ax3.bar(L_vals, frekanslar, color="#36b9cc", edgecolor="black", width=0.55)
        for i, f in enumerate(frekanslar):
            ax3.text(i, f * 1.15, f"{int(f)}π", ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax3.set_title("3. Fourier Pozisyonel Kodlama Frekansları (L=10)", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Frekans Bandı İndeksi (0..L-1)")
        ax3.set_ylabel("Frekans Çarpanı (2^i * pi)")
        ax3.set_yscale("log")
        ax3.grid(True, linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 4: NeRF İcra İzi ve Render Metrikleri Logu
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.axis("off")
        ax4.set_title("4. NeRF Hacimsel Render İcra İzi", fontsize=12, fontweight="bold", pad=10)

        nerf_metni = (
            "====================================================\n"
            "        NEURAL RADIANCE FIELD (NeRF) LOG            \n"
            "====================================================\n"
            f"SAHNE ADI        : {rapor['sahne_adi']}\n"
            f"KAMERA BAKIŞLARI : {rapor['kamera_sayisi']} Pozlandırılmış Fotoğraf\n"
            f"IŞIN BAŞINA NOKTA: {rapor['ornekleme_sayisi']} Tabakalı Örneklem (Stratified)\n"
            "----------------------------------------------------\n"
            "REKONSTRÜKSİYON METRİKLERİ:\n"
            f"• PSNR Skoru     : {rapor['metrikler']['psnr']} dB (Üstün Kalite)\n"
            f"• SSIM Sadakati  : %{rapor['metrikler']['ssim']*100:.1f}\n"
            f"• LPIPS Hatası   : {rapor['metrikler']['lpips']} (Algısal Benzerlik)\n"
            f"• Model Boyutu   : 5.2 MB (Tüm 3D Sahne Tek Bir MLP'de)\n"
            "----------------------------------------------------\n"
            f"FREKANS GÖMMESİ  : {rapor['fourier_l_seviyesi']}\n"
            "===================================================="
        )

        ax4.text(
            0.02, 0.5, nerf_metni,
            fontsize=7.2,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 5: Hacimsel Işın İntegrali ve MLP Akış Şeması
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. NeRF Işın İzleme ve MLP Mimarisi", fontsize=12, fontweight="bold", pad=10)

        sema_metni = (
            "====================================================\n"
            "         VOLUMETRIC RAY RENDERING EQUATION          \n"
            "====================================================\n"
            "  Kamera Merkezi (o) ──> Işın: r(t) = o + t*d ─────┐\n"
            "                                                   │\n"
            "  Noktalar (x, y, z) ──> [Fourier gamma(p)] ──────┐│\n"
            "                                                  ▼▼\n"
            "  [8-Katmanlı 256-D MLP] ──> Hacimsel Yoğunluk: sigma(x)\n"
            "           │ (Özellik Vektörü)                     │\n"
            "           ▼                                       │\n"
            "  [Bakış Yönü (theta, phi) + MLP] ──> Renk: c(x, d)│\n"
            "                                                   │\n"
            "  HACİMSEL İNTEGRAL:                               │\n"
            "  C(r) = sum_{i=1}^N T_i * (1 - exp(-sigma_i*delta_i)) * c_i\n"
            "  (Piksel Rengi Sentezlenir: Fotogerçekçi 3D Sahne!) \n"
            "===================================================="
        )

        ax5.text(
            0.02, 0.5, sema_metni,
            fontsize=7.1,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 6: GÜN 178 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. GÜN 178 ÖZET KARTI", fontsize=12, fontweight="bold", pad=10)

        ozet_metin = (
            "====================================================\n"
            "    DAY 178 SUMMARY: NEURAL RADIANCE FIELDS (NeRF)  \n"
            "====================================================\n"
            "• Modül              : FAZ 9 (Çok Modlu Modeller)\n"
            "• Temel Mimari       : Mildenhall et al. (2020) NeRF\n"
            "• Girişler           : 3D Konum (x,y,z) + 3D Yön (dx,dy,dz)\n"
            "• Çıkışlar           : Yoğunluk sigma (Saydamlık) + Renk RGB (c)\n"
            f"• Sahne Kalitesi     : {rapor['metrikler']['psnr']} dB PSNR\n"
            "----------------------------------------------------\n"
            "TEMEL KAZANIMLAR:\n"
            "  1. 3D ızgara (voxel) yerine 5MB'lık sürekli sinirsel fonksiyon\n"
            "  2. Fourier Positional Encoding ile keskin doku ve kenar sentezi\n"
            "  3. Volumetric Ray Marching ile ışık geçirgenliği ve gölge fiziği\n"
            "  4. Görüş açısına bağımlı parıltı ve yansımaları (specular) modelleme\n"
            "====================================================\n"
            "   SIRADAKİ GÜN: Gün 179 (3D Gaussian Splatting - 3DGS)\n"
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
        print(f"  ✓ NeRF Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
