"""
Day 30: Büyük Final Çoklu Görev Görsel Analiz Platformu Teşhis Panosu.
"""

from typing import Dict, List, Any
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns


class BuyukFinalGorsellestirici:
    """
    Sınıflandırma, Nesne Tespiti, Anlamsal Bölütleme, Çoklu Takip,
    Model Kuantizasyonu ve 30 Günlük Yetkinlik Radarını kapsayan nihai teşhis panosu.
    """

    @classmethod
    def buyuk_final_panosu_ciz(
        cls,
        ornek_kare_rgb: np.ndarray,
        telemetri: Dict[str, Any],
        optimizasyon_sonuclari: Dict[str, Any],
        belirsizlik_gecmisi: Dict[str, List[float]],
        radar_metrikleri: Dict[str, float],
        hedef_path: str = "ciktilar/multitask_analiz_paneli.png"
    ) -> str:
        os.makedirs(os.path.dirname(hedef_path), exist_ok=True)

        sns.set_theme(style="white", font_scale=0.9)
        fig, axes = plt.subplots(2, 3, figsize=(19, 13), dpi=300)
        fig.suptitle("Day 30: Büyük Final — Uçtan Uca Çoklu Görev Görsel Analiz Platformu", fontsize=16, fontweight="bold", y=0.98)

        # -------------------------------------------------------------
        # Panel 1: Nihai Çok Görevli Telemetri Görseli
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.imshow(ornek_kare_rgb)

        # Bölütleme saydam overlay
        seg_mask = telemetri.get("seg_maskesi", np.zeros((256, 256)))
        seg_rgba = np.zeros((*seg_mask.shape, 4), dtype=np.float32)
        # Yol pikselleri mor/yeşil saydam
        seg_rgba[seg_mask == 1] = [0.2, 0.8, 0.4, 0.35]
        ax1.imshow(seg_rgba)

        # Sahne Rozeti (Global Classification Badge)
        scene_txt = f"Sahne: {telemetri.get('sahne_etiketi', 'Otoyol')} (%{telemetri.get('sahne_guveni', 0.96)*100:.1f})"
        ax1.text(12, 24, scene_txt, color="white", fontsize=9, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#1f77b4", ec="white", alpha=0.9))

        # Takipçiler, Kutular ve Yörüngeler
        renkler = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
        for idx, t in enumerate(telemetri.get("aktif_takipciler", [])):
            c = renkler[(t.track_id - 1) % len(renkler)]
            x1, y1, x2, y2 = t.kutu
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2.2, edgecolor=c, facecolor="none")
            ax1.add_patch(rect)

            avg_spd = np.mean(t.hiz_gecmisi) if t.hiz_gecmisi else 5.0
            lbl = f"ID #{t.track_id} | {avg_spd:.1f} px/f"
            ax1.text(x1, max(0, y1 - 4), lbl, color="white", fontsize=7.5, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2", fc=c, ec="none", alpha=0.9))

            if len(t.yörünge) > 1:
                pts = np.array(t.yörünge)
                ax1.plot(pts[:, 0], pts[:, 1], color=c, linewidth=1.8, alpha=0.7)

        ax1.set_title("1. Çoklu Görev Görsel Telemetrisi (BBox + Track + Seg)", fontweight="bold", color="#1f77b4")
        ax1.axis("off")

        # -------------------------------------------------------------
        # Panel 2: Paylaşımlı Omurga Aktivasyon Haritası
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        feat_map = telemetri.get("omurga_aktivasyonu", np.random.rand(32, 32))
        im2 = ax2.imshow(feat_map, cmap="magma", interpolation="bilinear")
        ax2.set_title("2. Paylaşımlı Omurga & FPN Özellik Aktivasyonu", fontweight="bold", color="#d62728")
        ax2.axis("off")
        cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label("Özellik Yoğunluğu", fontsize=8)

        # -------------------------------------------------------------
        # Panel 3: Anlamsal Bölütleme Çıktı Haritası
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        im3 = ax3.imshow(seg_mask, cmap="tab10", vmin=0, vmax=4)
        ax3.set_title("3. Anlamsal Bölütleme Maskesi (Dense Segmentation)", fontweight="bold", color="#2ca02c")
        ax3.axis("off")
        cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, ticks=[0, 1, 2, 3, 4])
        cbar3.ax.set_yticklabels(["0: Gökyüzü", "1: Yol", "2: Bina", "3: Araç", "4: Yaya"], fontsize=8)

        # -------------------------------------------------------------
        # Panel 4: Homoscedastic Belirsizlik Ağırlıkları
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        epoklar = np.arange(1, len(belirsizlik_gecmisi.get("cls_sigma", [1, 2, 3])) + 1)
        ax4.plot(epoklar, belirsizlik_gecmisi.get("cls_sigma", [1.0]*len(epoklar)), label=r"$\sigma_{\text{cls}}$ (Sınıflandırma Belirsizliği)", color="#1f77b4", linewidth=2)
        ax4.plot(epoklar, belirsizlik_gecmisi.get("det_sigma", [1.2]*len(epoklar)), label=r"$\sigma_{\text{det}}$ (Tespit Belirsizliği)", color="#d62728", linewidth=2)
        ax4.plot(epoklar, belirsizlik_gecmisi.get("seg_sigma", [1.5]*len(epoklar)), label=r"$\sigma_{\text{seg}}$ (Bölütleme Belirsizliği)", color="#2ca02c", linewidth=2)
        ax4.set_title("4. Belirsizlik Ağırlıklı Kayıp Yakınsaması", fontweight="bold", color="#9467bd")
        ax4.set_xlabel("Eğitim İterasyonu", fontsize=9)
        ax4.set_ylabel(r"Öğrenilen Belirsizlik Değeri ($\sigma$)", fontsize=9)
        ax4.legend(fontsize=8)
        ax4.grid(True, linestyle=":", alpha=0.6)

        # -------------------------------------------------------------
        # Panel 5: Model Kuantizasyon & Hız Kıyaslaması (FPS vs Boyut)
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        modlar = ["FP32\n(Tam Hassasiyet)", "FP16\n(Yarı Hassasiyet)", "INT8\n(Dinamik Kuantize)"]
        fps_vals = [
            optimizasyon_sonuclari.get("FP32", {}).get("fps", 42.0),
            optimizasyon_sonuclari.get("FP16", {}).get("fps", 78.0),
            optimizasyon_sonuclari.get("INT8", {}).get("fps", 125.0)
        ]
        boyut_vals = [
            optimizasyon_sonuclari.get("FP32", {}).get("boyut_mb", 18.5),
            optimizasyon_sonuclari.get("FP16", {}).get("boyut_mb", 9.2),
            optimizasyon_sonuclari.get("INT8", {}).get("boyut_mb", 4.8)
        ]

        x_indices = np.arange(len(modlar))
        width = 0.35

        bars1 = ax5.bar(x_indices - width/2, fps_vals, width, label="Çıkarım Hızı (FPS)", color="#ff7f0e", edgecolor="black", linewidth=1.1)
        ax5_twin = ax5.twinx()
        bars2 = ax5_twin.bar(x_indices + width/2, boyut_vals, width, label="Model Boyutu (MB)", color="#17becf", edgecolor="black", linewidth=1.1)

        ax5.set_xticks(x_indices)
        ax5.set_xticklabels(modlar, fontsize=8)
        ax5.set_ylabel("Hız (FPS)", color="#ff7f0e", fontweight="bold", fontsize=9)
        ax5_twin.set_ylabel("Boyut (MB)", color="#17becf", fontweight="bold", fontsize=9)
        ax5.set_title("5. Model Optimizasyonu & Kuantizasyon Kıyaslaması", fontweight="bold", color="#ff7f0e")

        for bar in bars1:
            h = bar.get_height()
            ax5.annotate(f"{h:.1f} FPS", (bar.get_x() + bar.get_width() / 2, h), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7.5, fontweight="bold")
        for bar in bars2:
            h = bar.get_height()
            ax5_twin.annotate(f"{h:.1f} MB", (bar.get_x() + bar.get_width() / 2, h), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7.5, fontweight="bold")

        # -------------------------------------------------------------
        # Panel 6: 30 Günlük Büyük Final Yetkinlik Çizelgesi
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        gorev_adlari = [
            "Sınıflandırma\n(Top-1 Acc)",
            "Nesne Tespiti\n(mAP@0.5)",
            "Anlamsal Bölütleme\n(mIoU)",
            "Çoklu Takip\n(MOTA)",
            "Kimlik Koruma\n(IDF1)"
        ]
        gorev_skorlari = [
            radar_metrikleri.get("siniflandirma_acc", 0.96) * 100,
            radar_metrikleri.get("tespit_map", 0.94) * 100,
            radar_metrikleri.get("bolutleme_miou", 0.88) * 100,
            radar_metrikleri.get("takip_mota", 0.95) * 100,
            radar_metrikleri.get("takip_idf1", 0.97) * 100
        ]
        gorev_renkleri = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b"]

        bars6 = ax6.bar(gorev_adlari, gorev_skorlari, color=gorev_renkleri, width=0.55, edgecolor="black", linewidth=1.2)
        ax6.set_ylim(0, 115)
        ax6.set_ylabel("Başarı Skoru (%)", fontweight="bold", fontsize=10)
        ax6.set_title("6. 30 Günlük Çoklu Görev Yetkinlik Paneli", fontweight="bold", color="#333333")
        ax6.grid(axis="y", linestyle="--", alpha=0.6)

        for bar in bars6:
            h = bar.get_height()
            ax6.annotate(f"%{h:.1f}", (bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")

        fig.subplots_adjust(top=0.93, bottom=0.07, left=0.05, right=0.95, hspace=0.26, wspace=0.26)
        fig.savefig(hedef_path, bbox_inches="tight")
        plt.close(fig)
        return hedef_path
