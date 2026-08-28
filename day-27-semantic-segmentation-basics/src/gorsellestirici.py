"""Anlamsal Bölütleme 6 Panelli Teşhis ve Değerlendirme Görselleştirme Modülü.

Bu modül; Orijinal Görsel, Ground Truth Maskesi, Model Tahmin Maskesi,
Piksel Hata Haritası (Error Heatmap), Eğitim/Doğrulama Kayıp & mIoU Eğrileri ve
Sınıf Bazında IoU/Dice metriklerini içeren 6 panelli endüstriyel teşhis panosunu üretir.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class BolutlemeGorsellestirici:
    """Anlamsal bölütleme sonuçlarını ve metriklerini görselleştiren sınıf."""

    @staticmethod
    def dashboard_ciz(
        orijinal_gorsel: np.ndarray,
        gt_maske: np.ndarray,
        pred_maske: np.ndarray,
        egitim_tarihcesi: Dict[str, List[float]],
        sinif_raporu: Dict[str, Dict[str, float]],
        sinif_isimleri: List[str],
        hedef_dosya: Union[str, Path] = "ciktilar/bolutleme_teshis_paneli.png",
    ) -> Path:
        """6 panelli görselleştirme panosunu oluşturur ve kaydeder."""
        hedef_path = Path(hedef_dosya)
        hedef_path.parent.mkdir(parents=True, exist_ok=True)

        sns.set_theme(style="whitegrid")
        fig, eksenler = plt.subplots(2, 3, figsize=(18, 11), dpi=140)
        fig.suptitle(
            "U-Net Anlamsal Bölütleme (Semantic Segmentation) Kapsamlı Teşhis Panosu",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )

        # Özel Renk Paleti: 0: Arka Plan (Gri), 1: Hücre Gövdesi (Mavi), 2: Çekirdek (Kırmızı/Mor)
        maske_cmap = ListedColormap(["#2b2d42", "#457b9d", "#e63946"])

        # ----------------------------------------------------
        # PANEL 1: Orijinal Giriş Görseli
        # ----------------------------------------------------
        ax1 = eksenler[0, 0]
        ax1.imshow(orijinal_gorsel)
        ax1.set_title("1. Orijinal Mikroskobik Doku Görseli (RGB)", fontsize=11, fontweight="bold")
        ax1.axis("off")

        # ----------------------------------------------------
        # PANEL 2: Ground Truth Piksel Maskesi
        # ----------------------------------------------------
        ax2 = eksenler[0, 1]
        im2 = ax2.imshow(gt_maske, cmap=maske_cmap, vmin=0, vmax=len(sinif_isimleri) - 1)
        ax2.set_title("2. Gerçek Bölütleme Maskesi (Ground Truth)", fontsize=11, fontweight="bold")
        ax2.axis("off")

        # ----------------------------------------------------
        # PANEL 3: U-Net Tahmin Maskesi
        # ----------------------------------------------------
        ax3 = eksenler[0, 2]
        im3 = ax3.imshow(pred_maske, cmap=maske_cmap, vmin=0, vmax=len(sinif_isimleri) - 1)
        ax3.set_title("3. U-Net Model Tahmin Maskesi (Prediction)", fontsize=11, fontweight="bold")
        ax3.axis("off")

        # Renk Çubuğu (Colorbar with Class Names)
        cbar = fig.colorbar(im3, ax=[ax2, ax3], fraction=0.025, pad=0.03, ticks=list(range(len(sinif_isimleri))))
        cbar.ax.set_yticklabels(sinif_isimleri, fontsize=9, fontweight="bold")

        # ----------------------------------------------------
        # PANEL 4: Piksel Düzeyinde Hata Haritası (Error Heatmap)
        # ----------------------------------------------------
        ax4 = eksenler[1, 0]
        hata_haritasi = (gt_maske != pred_maske).astype(int)
        hata_cmap = ListedColormap(["#f1faee", "#e63946"])  # Beyaz/Açık Yeşil: Doğru, Kırmızı: Yanlış
        im4 = ax4.imshow(hata_haritasi, cmap=hata_cmap, vmin=0, vmax=1)
        hata_orani = np.mean(hata_haritasi) * 100.0
        ax4.set_title(f"4. Piksel Hata Haritası (Hata: %{hata_orani:.2f})", fontsize=11, fontweight="bold")
        ax4.axis("off")

        # ----------------------------------------------------
        # PANEL 5: Eğitim ve Doğrulama Kayıp & mIoU Eğrileri
        # ----------------------------------------------------
        ax5 = eksenler[1, 1]
        epoklar = list(range(1, len(egitim_tarihcesi["train_loss"]) + 1))

        # Sol Eksen: Kayıp (Loss)
        ax5.plot(epoklar, egitim_tarihcesi["train_loss"], "o-", label="Train Combo Loss", color="#1f77b4", linewidth=2)
        ax5.plot(epoklar, egitim_tarihcesi["val_loss"], "s-", label="Val Combo Loss", color="#d62728", linewidth=2)
        ax5.set_xlabel("Epok (Epoch)")
        ax5.set_ylabel("Kayıp (Combo Loss)")
        ax5.set_title("5. Eğitim & Doğrulama Metrik İlerlemesi", fontsize=11, fontweight="bold")
        ax5.grid(True, linestyle="--", alpha=0.5)

        # Sağ Eksen: mIoU
        ax5_sag = ax5.twinx()
        ax5_sag.plot(epoklar, [v * 100 for v in egitim_tarihcesi["val_miou"]], "^--", label="Val mIoU (%)", color="#2ca02c", linewidth=2)
        ax5_sag.set_ylabel("Doğrulama mIoU (%)", color="#2ca02c")
        ax5_sag.tick_params(axis="y", labelcolor="#2ca02c")
        ax5_sag.grid(False)

        # Ortak Legend
        lines1, labels1 = ax5.get_legend_handles_labels()
        lines2, labels2 = ax5_sag.get_legend_handles_labels()
        ax5.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8, frameon=True)

        # ----------------------------------------------------
        # PANEL 6: Sınıf Bazında IoU ve Dice Skorları Çubuk Grafiği
        # ----------------------------------------------------
        ax6 = eksenler[1, 2]
        siniflar = list(sinif_raporu.keys())
        iou_degerleri = [sinif_raporu[s]["iou"] * 100 for s in siniflar]
        dice_degerleri = [sinif_raporu[s]["dice"] * 100 for s in siniflar]

        x = np.arange(len(siniflar))
        width = 0.35

        bars1 = ax6.bar(x - width / 2, iou_degerleri, width, label="IoU (Jaccard)", color="#457b9d")
        bars2 = ax6.bar(x + width / 2, dice_degerleri, width, label="Dice (F1-Score)", color="#2a9d8f")

        ax6.set_title("6. Sınıf Bazında IoU & Dice Dağılımı (%)", fontsize=11, fontweight="bold")
        ax6.set_xticks(x)
        ax6.set_xticklabels(siniflar, fontsize=9, fontweight="bold")
        ax6.set_ylabel("Skor (%)")
        ax6.set_ylim(0, 115)
        ax6.grid(True, linestyle="--", alpha=0.5, axis="y")
        ax6.legend(loc="upper right", fontsize=8, frameon=True)

        for bar in bars1:
            h = bar.get_height()
            ax6.annotate(f"%{h:.1f}", (bar.get_x() + bar.get_width() / 2, h), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7, fontweight="bold")

        for bar in bars2:
            h = bar.get_height()
            ax6.annotate(f"%{h:.1f}", (bar.get_x() + bar.get_width() / 2, h), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7, fontweight="bold")

        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.06, right=0.94, hspace=0.28, wspace=0.28)
        fig.savefig(hedef_path, bbox_inches="tight")
        plt.close(fig)
        return hedef_path
