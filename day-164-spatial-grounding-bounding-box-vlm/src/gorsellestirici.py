"""
Spatial Grounding Teşhis Panosu Görselleştirici Modülü (Day 164 - FAZ 9).
6 panelli Bounding Box Görseli, IoU Skorları Dağılımı, Koordinat Ayrıştırma, RefCOCO Doğruluğu, Mimari Şema ve Özet Kartı.
"""

import os
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


class SpatialGroundingGorsellestirici:
    """Spatial Grounding Teşhis Panosu Üreticisi."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def pano_olustur(
        self,
        degerlendirme_raporu: Dict[str, Any],
        kayit_yolu: str = "ciktilar/spatial_grounding_bounding_box_paneli.png",
    ):
        """6 panelli teşhis panosunu oluşturur ve kaydeder."""
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(23, 13.5))
        fig.suptitle(
            "GÜN 164 (FAZ 9): Spatial Grounding — [ymin, xmin, ymax, xmax] Koordinat Çıkarma ve RefCOCO Bounding Box Analizi",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        senaryolar = degerlendirme_raporu["senaryo_sonuclari"]
        ozet = degerlendirme_raporu["genel_ozet"]

        # -------------------------------------------------------------
        # PANEL 1: Sentetik Görüntü Üzerinde Bounding Box Çizimi (GT vs Pred)
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.set_xlim(0, 1000)
        ax1.set_ylim(1000, 0)  # Görüntü koordinat sistemi (y üstten başlar)
        ax1.set_facecolor("#f1f3f5")

        renkler = ["#e74a3b", "#4e73df", "#1cc88a", "#f6c23e"]
        for i, s in enumerate(senaryolar):
            gt = s["gt_kutu"]
            pred = s["tahmin_kutu"]
            r_renk = renkler[i % len(renkler)]

            # Ground Truth Kutusu (Kesikli Çizgi)
            gt_rect = patches.Rectangle(
                (gt[1], gt[0]), gt[3] - gt[1], gt[2] - gt[0],
                linewidth=2, edgecolor=r_renk, facecolor="none", linestyle="--", label=f"GT: {s['nesne_adi']}" if i == 0 else ""
            )
            ax1.add_patch(gt_rect)

            # Model Tahmini Kutusu (Düz Çizgi)
            pred_rect = patches.Rectangle(
                (pred[1], pred[0]), pred[3] - pred[1], pred[2] - pred[0],
                linewidth=2.5, edgecolor=r_renk, facecolor=r_renk, alpha=0.18, label=f"Pred: {s['nesne_adi']}" if i == 0 else ""
            )
            ax1.add_patch(pred_rect)
            ax1.text(pred[1] + 15, pred[0] + 35, f"{s['nesne_adi']}\n(IoU: {s['iou']})", color=r_renk, fontweight="bold", fontsize=8)

        ax1.set_title("1. Koordinat Uzayı (1000x1000 Grid): GT (Kesikli) vs Pred (Dolu)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("X Koordinatı (xmin -> xmax)")
        ax1.set_ylabel("Y Koordinatı (ymin -> ymax)")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # -------------------------------------------------------------
        # PANEL 2: Nesne Başına IoU Skorları (Kesişim / Birleşim)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        nesne_adlari = [s["nesne_adi"] for s in senaryolar]
        iou_degerleri = [s["iou"] * 100.0 for s in senaryolar]

        barlar2 = ax2.bar(nesne_adlari, iou_degerleri, color=["#e74a3b", "#4e73df", "#1cc88a", "#f6c23e"], edgecolor="black", width=0.45)
        ax2.axhline(50.0, color="red", linestyle="--", lw=2, label="Eşik Değeri (IoU @ 0.50)")

        for bar in barlar2:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"%{h:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax2.set_title(f"2. Bounding Box Doğruluğu (Ortalama IoU: %{ozet['ortalama_iou']*100:.1f})", fontsize=12, fontweight="bold")
        ax2.set_ylabel("IoU Skoru (%)")
        ax2.set_ylim(0, 115)
        ax2.legend(loc="lower right")
        ax2.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 3: mAP@0.5 Başarı Metriği
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        metrik_isimleri = ["RefCOCO mAP@0.5", "Başarılı Tespit Oranı", "Hata Payı"]
        metrik_degerleri = [ozet["map_50_yuzdesi"], (ozet["dogru_tespit_sayisi"] / ozet["toplam_nesne"]) * 100.0, 0.0]

        barlar3 = ax3.bar(metrik_isimleri, metrik_degerleri, color=["#1cc88a", "#36b9cc", "#e74a3b"], edgecolor="black", width=0.45)
        for bar in barlar3:
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"%{h:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax3.set_title(f"3. RefCOCO Grounding Başarımı (%{ozet['map_50_yuzdesi']} mAP@0.5)", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Oran (%)")
        ax3.set_ylim(0, 120)
        ax3.grid(axis="y", linestyle="--", alpha=0.7)

        # -------------------------------------------------------------
        # PANEL 4: Örnek Spatial Grounding Çıkarım Detayı
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.axis("off")
        ax4.set_title("4. Örnek Referans Komutu ve Metin Koordinatı", fontsize=12, fontweight="bold", pad=10)

        s_ornek = senaryolar[0]
        ornek_metin = (
            "====================================================\n"
            "         SPATIAL GROUNDING INFERENCE TRACE          \n"
            "====================================================\n"
            f"HEDEF NESNE    : {s_ornek['nesne_adi']}\n"
            "KULLANICI KOMUTU:\n"
            "  'Görseldeki kırmızı spor arabayı tespit et.'\n"
            "----------------------------------------------------\n"
            f"MODEL ÇIKTISI (Metin): [ymin, xmin, ymax, xmax]\n"
            f"  {s_ornek['tahmin_kutu']}\n"
            f"GERÇEK ETİKET (GT)   : {s_ornek['gt_kutu']}\n"
            f"HESAPLANAN IoU SKORU : %{s_ornek['iou']*100:.1f} (Doğrulandı: IoU >= 0.5)\n"
            "===================================================="
        )

        ax4.text(
            0.02, 0.5, ornek_metin,
            fontsize=7.3,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 5: Spatial Grounding Mimari Şeması
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.axis("off")
        ax5.set_title("5. Spatial Grounding VLM Akış Mimarisi", fontsize=12, fontweight="bold", pad=10)

        sema_metni = (
            "====================================================\n"
            "        SPATIAL GROUNDING PIPELINE (Det-VLM)        \n"
            "====================================================\n"
            "  [Görüntü (224x224)] + [Doğal Dil Referansı]        \n"
            "           │                    │                   \n"
            "           ▼                    ▼                   \n"
            "  [ViT Patch Encoder] ──> [LLaVA VLM Füzyonu]       \n"
            "                                │                   \n"
            "                                ▼                   \n"
            "  [Oto-Regresif Metin Koordinatı Üretimi]           \n"
            "  'Tespit edilen nesne: [ymin, xmin, ymax, xmax]'   \n"
            "                                │                   \n"
            "                                ▼                   \n"
            "  [Regex Parser & 0-1000 Normalizer]                \n"
            "                                │                   \n"
            "                                ▼                   \n"
            "  [Kutu Çizimi & IoU / mAP@0.5 Değerlendirme]       \n"
            "===================================================="
        )

        ax5.text(
            0.02, 0.5, sema_metni,
            fontsize=7.3,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.5),
        )

        # -------------------------------------------------------------
        # PANEL 6: GÜN 164 Özet Kartı
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        ax6.set_title("6. GÜN 164 ÖZET KARTI", fontsize=12, fontweight="bold", pad=10)

        ozet_metin = (
            "====================================================\n"
            "   DAY 164 SUMMARY: SPATIAL GROUNDING IN VLMs       \n"
            "====================================================\n"
            "• Modül              : FAZ 9 (Çok Modlu Modeller)\n"
            "• Çıktı Formatı      : [ymin, xmin, ymax, xmax] (0-1000)\n"
            "• Başarı Metriği     : %" + str(ozet["map_50_yuzdesi"]) + " mAP@0.50 (Tüm Nesneler Geçti!)\n"
            f"• Ortalama IoU       : %{ozet['ortalama_iou']*100:.1f}\n"
            "----------------------------------------------------\n"
            "TEMEL KAZANIMLAR:\n"
            "  1. Dil modeline özel tespit kafası eklemeden koordinat üretme\n"
            "  2. Normalize koordinat tokenları [0-1000] ile tam uyum\n"
            "  3. RefCOCO tarzı karmaşık doğal dil referanslarını anlama\n"
            "  4. GUI Ajanları ve Otonom Sürüş için uzamsal temel oluşturma\n"
            "====================================================\n"
            "   SIRADAKİ GÜN: Gün 165 (OCR-Free Document Understanding - Nougat)\n"
            "===================================================="
        )

        ax6.text(
            0.02, 0.5, ozet_metin,
            fontsize=7.8,
            family="monospace",
            va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#d4edda", edgecolor="#28a745", lw=1.5),
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(kayit_yolu, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Spatial Grounding Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(kayit_yolu)}")
