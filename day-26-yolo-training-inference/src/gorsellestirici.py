"""YOLO Eğitim ve Çıkarım Sonuçları 4 Panelli Görselleştirme Modülü.

Bu modül; Eğitim Kayıp Eğrilerini (Box/Cls/DFL Loss), mAP@0.5 ve mAP@0.5:0.95 metriklerini,
Sınıf bazında Precision-Recall eğrilerini ve Test Görseli üzerindeki nihai tespitleri
içeren 4 panelli endüstri standardı teşhis panosunu (Diagnostic Dashboard) üretir.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


class YOLOGorsellestirici:
    """YOLO eğitim ve çıkarım sonuçlarını görselleştiren sınıf."""

    @staticmethod
    def dashboard_ciz(
        kayip_gecmisi: Dict[str, List[float]],
        map_bilgisi: Dict,
        test_gorseli: np.ndarray,
        tahminler: List[Dict],
        sinif_isimleri: List[str],
        hedef_dosya: Union[str, Path] = "ciktilar/yolo_egitim_ve_cikarim_paneli.png",
    ) -> Path:
        """4 panelli YOLO teşhis panosunu oluşturur ve kaydeder."""
        hedef_path = Path(hedef_dosya)
        hedef_path.parent.mkdir(parents=True, exist_ok=True)

        fig, eksenler = plt.subplots(2, 2, figsize=(16, 13), dpi=140)
        fig.suptitle(
            "Ultralytics YOLO Nesne Tespiti Eğitimi, Metrik Değerlendirmesi ve Çıkarım Panosu",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )

        renk_listesi = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        # ----------------------------------------------------
        # PANEL 1: Eğitim Kayıp Eğrileri (Loss Curves)
        # ----------------------------------------------------
        ax1 = eksenler[0, 0]
        epoklar = list(range(1, len(kayip_gecmisi.get("box_loss", [])) + 1))

        if len(epoklar) > 0:
            ax1.plot(epoklar, kayip_gecmisi.get("box_loss", []), "o-", label="Box Loss (BBox Regresyon)", color="#d62728", linewidth=2)
            ax1.plot(epoklar, kayip_gecmisi.get("cls_loss", []), "s-", label="Class Loss (Sınıflandırma)", color="#1f77b4", linewidth=2)
            ax1.plot(epoklar, kayip_gecmisi.get("dfl_loss", []), "^-", label="DFL Loss (Distribution Focal)", color="#2ca02c", linewidth=2)

        ax1.set_title("1. Eğitim Kayıp Eğrileri (Loss Progression)", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Epok (Epoch)")
        ax1.set_ylabel("Kayıp (Loss)")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper right", fontsize=8, frameon=True)

        # ----------------------------------------------------
        # PANEL 2: Sınıf Bazında AP ve Genel mAP Dağılımı
        # ----------------------------------------------------
        ax2 = eksenler[0, 1]
        sinif_ap_05 = map_bilgisi.get("sinif_ap_05", {})
        sinif_ap_coco = map_bilgisi.get("sinif_ap_coco", {})

        siniflar = list(sinif_ap_05.keys())
        ap50_vals = [sinif_ap_05[s] * 100 for s in siniflar]
        ap_coco_vals = [sinif_ap_coco[s] * 100 for s in siniflar]

        x = np.arange(len(siniflar))
        width = 0.35

        ax2.bar(x - width / 2, ap50_vals, width, label=f"mAP@0.5 (%{map_bilgisi['map_05']*100:.1f})", color="#1f77b4")
        ax2.bar(x + width / 2, ap_coco_vals, width, label=f"mAP@0.5:0.95 (%{map_bilgisi['map_05_95']*100:.1f})", color="#ff7f0e")

        ax2.set_title("2. Doğrulama Metrikleri: AP@0.5 vs COCO AP@0.5:0.95", fontsize=11, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(siniflar, fontsize=9, fontweight="bold")
        ax2.set_ylabel("Ortalama Hassasiyet (AP %)")
        ax2.set_ylim(0, 115)
        ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
        ax2.legend(loc="upper right", fontsize=8, frameon=True)

        # ----------------------------------------------------
        # PANEL 3: Sınıf Bazında Precision-Recall Eğrileri
        # ----------------------------------------------------
        ax3 = eksenler[1, 0]
        pr_egrileri = map_bilgisi.get("pr_egrileri", {})

        for i, (sinif_adi, (prec, rec)) in enumerate(pr_egrileri.items()):
            if len(prec) > 0 and len(rec) > 0:
                ap_val = sinif_ap_05.get(sinif_adi, 0.0)
                ax3.plot(rec, prec, label=f"{sinif_adi} (AP@0.5={ap_val:.2f})", color=renk_listesi[i % len(renk_listesi)], linewidth=2)

        ax3.set_title(f"3. Precision-Recall Eğrisi (Genel mAP@0.5: %{map_bilgisi['map_05']*100:.1f})", fontsize=11, fontweight="bold")
        ax3.set_xlabel("Duyarlılık (Recall)")
        ax3.set_ylabel("Kesinlik (Precision)")
        ax3.set_xlim(0, 1.05)
        ax3.set_ylim(0, 1.05)
        ax3.grid(True, linestyle="--", alpha=0.5)
        ax3.legend(loc="lower left", fontsize=8, frameon=True)

        # ----------------------------------------------------
        # PANEL 4: Gerçek Çıkarım / Tahmin Görseli
        # ----------------------------------------------------
        ax4 = eksenler[1, 1]
        img_rgb = cv2.cvtColor(test_gorseli, cv2.COLOR_BGR2RGB) if len(test_gorseli.shape) == 3 else test_gorseli
        ax4.imshow(img_rgb)

        for t in tahminler:
            box = t["box"]
            score = t["score"]
            c_name = t["class_name"]
            c_id = t["class_id"]

            x1, y1, x2, y2 = box
            color = renk_listesi[c_id % len(renk_listesi)]

            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2.5, edgecolor=color, facecolor="none"
            )
            ax4.add_patch(rect)
            ax4.text(
                x1, y1 - 6, f"{c_name} %{score*100:.0f}",
                color="white", fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.2", facecolor=color, edgecolor="none", alpha=0.85)
            )

        ax4.set_title(f"4. YOLO Model Çıkarım Sonucu ({len(tahminler)} Nesne Tespit Edildi)", fontsize=11, fontweight="bold")
        ax4.axis("off")

        plt.tight_layout()
        fig.savefig(hedef_path, bbox_inches="tight")
        plt.close(fig)
        return hedef_path
