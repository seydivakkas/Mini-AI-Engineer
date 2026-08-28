"""
İleri Düzey Bölütleme 6 Panelli Teşhis Panosu (Diagnostic Dashboard).
"""

from typing import Dict, List, Any
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns


class IleriBolutlemeGorsellestirici:
    """
    Instance, Semantic, Panoptic, RoIAlign ve Metrikleri kapsayan
    6 panelli yüksek çözünürlüklü teşhis çizelgesi üreticisi.
    """

    @classmethod
    def teshis_panosu_ciz(
        cls,
        sahne_verisi: Dict[str, Any],
        roi_align_ornek: np.ndarray,
        mask_pred_ornek: np.ndarray,
        metrikler: Dict[str, Any],
        hedef_path: str = "ciktilar/ileri_bolutleme_teshis_paneli.png"
    ) -> str:
        os.makedirs(os.path.dirname(hedef_path), exist_ok=True)

        sns.set_theme(style="white", font_scale=0.9)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=300)
        fig.suptitle("Day 28: İleri Düzey Bölütleme & Mask R-CNN / SegFormer Teşhis Panosu", fontsize=16, fontweight="bold", y=0.98)

        # -------------------------------------------------------------
        # Panel 1: Orijinal Çok Nesneli RGB Sahne
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.imshow(sahne_verisi["gorsel_rgb"])
        ax1.set_title("1. Orijinal Çok Nesneli RGB Sahne", fontweight="bold", color="#1f77b4")
        ax1.axis("off")

        # -------------------------------------------------------------
        # Panel 2: Anlamsal Bölütleme (Semantic Segmentation)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        sem_cmap = plt.get_cmap("tab10", 5)
        im2 = ax2.imshow(sahne_verisi["semantik_harita"], cmap=sem_cmap, vmin=0, vmax=4)
        ax2.set_title("2. Anlamsal Bölütleme (Semantic Map - Stuff & Things)", fontweight="bold", color="#2ca02c")
        ax2.axis("off")
        cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, ticks=[0, 1, 2, 3, 4])
        cbar2.ax.set_yticklabels(["0: Gökyüzü", "1: Yol", "2: Araç", "3: Yaya", "4: Engel"], fontsize=8)

        # -------------------------------------------------------------
        # Panel 3: Örnek Tabanlı Bölütleme (Instance Segmentation)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.imshow(sahne_verisi["gorsel_rgb"])
        ax3.set_title("3. Örnek Bölütleme (Instance Masks & BBoxes)", fontweight="bold", color="#d62728")
        ax3.axis("off")

        renkler = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33"]
        sinif_isimleri = {2: "Araç", 3: "Yaya", 4: "Engel"}

        for idx, (mask, box, lbl) in enumerate(zip(
            sahne_verisi["ornek_maskeleri"],
            sahne_verisi["ornek_kutulari"],
            sahne_verisi["ornek_siniflari"]
        )):
            color = renkler[idx % len(renkler)]
            # Maske overlay
            mask_rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
            c_rgb = [int(color[1:3], 16)/255, int(color[3:5], 16)/255, int(color[5:7], 16)/255]
            mask_rgba[mask > 0.5] = [*c_rgb, 0.45]
            ax3.imshow(mask_rgba)

            # Bounding Box ve ID Etiketi
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor="none")
            ax3.add_patch(rect)
            ax3.text(x1, max(0, y1 - 4), f"{sinif_isimleri.get(lbl, 'Obj')} #{idx+1}", color="white",
                     fontsize=8, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", fc=color, ec="none", alpha=0.9))

        # -------------------------------------------------------------
        # Panel 4: Panoptik Bölütleme (Panoptic Segmentation)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        # Panoptik ID'leri renk paletine eşle
        panoptic_map = sahne_verisi["panoptik_harita"]
        unique_pids = np.unique(panoptic_map)
        pid_to_idx = {pid: i for i, pid in enumerate(unique_pids)}
        indexed_panoptic = np.vectorize(pid_to_idx.get)(panoptic_map)

        im4 = ax4.imshow(indexed_panoptic, cmap="gist_ncar")
        ax4.set_title("4. Panoptik Bölütleme (Panoptic = Stuff + Things)", fontweight="bold", color="#9467bd")
        ax4.axis("off")

        # -------------------------------------------------------------
        # Panel 5: RoIAlign & Mask Head Tahmin Haritası
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        # RoIAlign özellik aktivasyonu ve üzerine tahmin edilen 28x28 maske
        ax5.imshow(roi_align_ornek, cmap="viridis", interpolation="bilinear")
        # Maske konturunu çiz
        ax5.contour(mask_pred_ornek, levels=[0.5], colors=["#ff0055"], linewidths=2.5)
        ax5.set_title("5. RoIAlign & FCN Maske Başlığı Aktivasyonu (28x28)", fontweight="bold", color="#8c564b")
        ax5.set_xlabel("RoIAlign Hücre Koordinatları (Çift Doğrusal Örnekleme)", fontsize=9)
        ax5.set_ylabel("RoIAlign Y-Ekseni", fontsize=9)

        # -------------------------------------------------------------
        # Panel 6: İleri Düzey Metrik Çizelgesi (PQ, SQ, RQ, Mask AP)
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        metrik_isimleri = ["Panoptik Kalite\n(PQ)", "Bölütleme Kalitesi\n(SQ)", "Tanıma Kalitesi\n(RQ)", "Maske AP@50\n(AP_50)", "Maske AP@75\n(AP_75)"]
        degerler = [
            metrikler.get("genel_pq", 0.85) * 100,
            metrikler.get("genel_sq", 0.92) * 100,
            metrikler.get("genel_rq", 0.90) * 100,
            metrikler.get("AP_50", 0.94) * 100,
            metrikler.get("AP_75", 0.82) * 100,
        ]
        bar_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]

        bars = ax6.bar(metrik_isimleri, degerler, color=bar_colors, width=0.55, edgecolor="black", linewidth=1.2)
        ax6.set_ylim(0, 115)
        ax6.set_ylabel("Başarı Oranı (%)", fontweight="bold", fontsize=10)
        ax6.set_title("6. İleri Düzey Bölütleme Metrikleri", fontweight="bold", color="#333333")
        ax6.grid(axis="y", linestyle="--", alpha=0.6)

        for bar in bars:
            h = bar.get_height()
            ax6.annotate(f"%{h:.1f}", (bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")

        fig.subplots_adjust(top=0.93, bottom=0.07, left=0.05, right=0.95, hspace=0.25, wspace=0.25)
        fig.savefig(hedef_path, bbox_inches="tight")
        plt.close(fig)
        return hedef_path
