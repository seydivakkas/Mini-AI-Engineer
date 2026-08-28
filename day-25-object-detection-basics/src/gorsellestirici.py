"""Nesne Tespiti, NMS ve Anchor Box 4 Panelli Görselleştirme Modülü.

Bu modül; NMS öncesi ham çakışan kutuları, NMS sonrası izole edilmiş nihai tespitleri,
çok ölçekli Anchor Box ızgara yerleşimini ve IoU/GIoU/DIoU karşılaştırma analizini
içeren 4 panelli endüstri standardı teşhis panosunu (Diagnostic Dashboard) üretir.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


class TespitGorsellestirici:
    """Nesne tespiti adımlarını görselleştiren sınıf."""

    @staticmethod
    def dashboard_ciz(
        gt_boxes: np.ndarray,
        gt_labels: List[str],
        raw_boxes: np.ndarray,
        raw_scores: np.ndarray,
        raw_labels: List[str],
        nms_boxes: np.ndarray,
        nms_scores: np.ndarray,
        nms_labels: List[str],
        anchors_sample: np.ndarray,
        iou_mat: np.ndarray,
        giou_mat: np.ndarray,
        img_size: Tuple[int, int] = (512, 512),
        hedef_dosya: Union[str, Path] = "ciktilar/nesne_tespiti_paneli.png",
    ) -> Path:
        """4 panelli nesne tespiti görselleştirme panosunu oluşturur ve kaydeder."""
        hedef_path = Path(hedef_dosya)
        hedef_path.parent.mkdir(parents=True, exist_ok=True)

        img_w, img_h = img_size
        fig, eksenler = plt.subplots(2, 2, figsize=(16, 14), dpi=140)
        fig.suptitle(
            "Nesne Tespiti Temelleri: IoU, NMS, Bounding Box Regresyonu ve Anchor Kutuları",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )

        renk_paleti = {
            "Araba": "#1f77b4",
            "Yaya": "#d62728",
            "Bisiklet": "#2ca02c",
            "Trafik Lambası": "#ff7f0e",
        }

        # ----------------------------------------------------
        # PANEL 1: Ham Tahminler & NMS Öncesi Çakışan Kutular
        # ----------------------------------------------------
        ax1 = eksenler[0, 0]
        ax1.set_xlim(0, img_w)
        ax1.set_ylim(img_h, 0)
        ax1.set_facecolor("#f8f9fa")

        # Ground Truth kutuları (Kesikli Yeşil Çizgi)
        for i, (box, lbl) in enumerate(zip(gt_boxes, gt_labels)):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2.5, edgecolor="green", facecolor="none", linestyle="--", label="Ground Truth" if i == 0 else ""
            )
            ax1.add_patch(rect)
            ax1.text(x1, y1 - 6, f"GT: {lbl}", color="green", fontweight="bold", fontsize=8,
                     bbox=dict(boxstyle="square,pad=0.2", facecolor="white", alpha=0.8, edgecolor="green"))

        # Ham aday kutular (Kırmızı/Mavi çakışan kutular)
        for box, score, lbl in zip(raw_boxes, raw_scores, raw_labels):
            x1, y1, x2, y2 = box
            c = renk_paleti.get(lbl, "#7f7f7f")
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.2, edgecolor=c, facecolor="none", alpha=0.6
            )
            ax1.add_patch(rect)
            ax1.text(x1, y2 + 10, f"%{score*100:.0f}", color=c, fontsize=6, alpha=0.8)

        ax1.set_title(f"1. Ham Tahminler & Çakışan Adaylar (NMS Öncesi: {len(raw_boxes)} Kutu)", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Piksel X")
        ax1.set_ylabel("Piksel Y")
        ax1.grid(True, linestyle=":", alpha=0.5)
        ax1.legend(loc="upper right", fontsize=8)

        # ----------------------------------------------------
        # PANEL 2: NMS Sonrası Temizlenmiş Nihai Tespitler
        # ----------------------------------------------------
        ax2 = eksenler[0, 1]
        ax2.set_xlim(0, img_w)
        ax2.set_ylim(img_h, 0)
        ax2.set_facecolor("#f8f9fa")

        # Ground Truth kutuları
        for box, lbl in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2.0, edgecolor="green", facecolor="none", linestyle="--", alpha=0.5
            )
            ax2.add_patch(rect)

        # NMS sonrası seçilen nihai kutular
        for box, score, lbl in zip(nms_boxes, nms_scores, nms_labels):
            x1, y1, x2, y2 = box
            c = renk_paleti.get(lbl, "#1f77b4")
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2.5, edgecolor=c, facecolor=c, alpha=0.2
            )
            ax2.add_patch(rect)
            border_rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2.5, edgecolor=c, facecolor="none"
            )
            ax2.add_patch(border_rect)
            ax2.text(x1, y1 - 8, f"{lbl} %{score*100:.1f}", color="white", fontweight="bold", fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor=c, edgecolor="none"))

        ax2.set_title(f"2. NMS Sonrası İzole Edilmiş Nihai Tespitler ({len(nms_boxes)} Kutu)", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Piksel X")
        ax2.set_ylabel("Piksel Y")
        ax2.grid(True, linestyle=":", alpha=0.5)

        # ----------------------------------------------------
        # PANEL 3: Çok Ölçekli Anchor Box Izgara Dağılımı
        # ----------------------------------------------------
        ax3 = eksenler[1, 0]
        ax3.set_xlim(0, img_w)
        ax3.set_ylim(img_h, 0)
        ax3.set_facecolor("#ffffff")

        # Grid çizgileri
        grid_step = 64
        for gx in range(0, img_w + 1, grid_step):
            ax3.axvline(gx, color="#e0e0e0", linestyle=":")
        for gy in range(0, img_h + 1, grid_step):
            ax3.axhline(gy, color="#e0e0e0", linestyle=":")

        # Anchor örnekleri çiz
        anchor_renkleri = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
        for i, box in enumerate(anchors_sample):
            x1, y1, x2, y2 = box
            c = anchor_renkleri[i % len(anchor_renkleri)]
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.8, edgecolor=c, facecolor="none", linestyle="-", alpha=0.75
            )
            ax3.add_patch(rect)
            ax3.plot((x1 + x2) / 2, (y1 + y2) / 2, "o", color=c, markersize=4)

        ax3.set_title(f"3. Anchor Kutuları Örnek Dağılımı (Ölçekler & En-Boy Oranları)", fontsize=11, fontweight="bold")
        ax3.set_xlabel("Piksel X")
        ax3.set_ylabel("Piksel Y")

        # ----------------------------------------------------
        # PANEL 4: IoU vs GIoU Karşılaştırma Isı Haritası
        # ----------------------------------------------------
        ax4 = eksenler[1, 1]
        im = ax4.imshow(iou_mat, cmap="YlGnBu", vmin=0.0, vmax=1.0)
        fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04, label="IoU Skoru")

        n_rows, n_cols = iou_mat.shape
        for r in range(n_rows):
            for col in range(n_cols):
                val_iou = iou_mat[r, col]
                val_giou = giou_mat[r, col]
                ax4.text(
                    col, r, f"IoU: {val_iou:.2f}\nGIoU: {val_giou:.2f}",
                    ha="center", va="center",
                    color="white" if val_iou > 0.5 else "black",
                    fontweight="bold", fontsize=7
                )

        ax4.set_title("4. Eşleşme Matrisi: IoU vs Generalized IoU (GIoU)", fontsize=11, fontweight="bold")
        ax4.set_xlabel("Ground Truth Kutuları")
        ax4.set_ylabel("Aday / Anchor Kutuları")
        ax4.set_xticks(np.arange(n_cols))
        ax4.set_xticklabels([f"GT-{i+1}" for i in range(n_cols)], fontsize=8)
        ax4.set_yticks(np.arange(n_rows))
        ax4.set_yticklabels([f"Aday-{i+1}" for i in range(n_rows)], fontsize=8)

        plt.tight_layout()
        fig.savefig(hedef_path, bbox_inches="tight")
        plt.close(fig)
        return hedef_path
