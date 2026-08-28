"""Model Değerlendirme ve Hata Analizi 6 Panelli Görselleştirme Modülü.

Bu modül; Karışıklık Matrisi, Çok Sınıflı ROC-AUC, PR-AUC, Olasılık Kalibrasyon Eğrisi (ECE),
Top-k Doğruluğu ve Aşırı Güvenli Hata Analizini içeren 6 panelli endüstri standardı
teşhis panosunu (Diagnostic Dashboard) üretir.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class DegerlendirmeGorsellestirici:
    """Kapsamlı model değerlendirme ve hata analizi görselleştiricisi."""

    @staticmethod
    def dashboard_ciz(
        metrik_raporu: Dict,
        kalibrasyon_ham: Dict,
        kalibrasyon_kalibre: Dict,
        asiri_guvenli_hatalar: List[Dict],
        sinif_isimleri: List[str],
        hedef_dosya: Union[str, Path] = "ciktilar/model_degerlendirme_paneli.png",
    ) -> Path:
        """6 panelli görselleştirme panosunu oluşturur ve kaydeder."""
        hedef_path = Path(hedef_dosya)
        hedef_path.parent.mkdir(parents=True, exist_ok=True)

        fig, eksenler = plt.subplots(2, 3, figsize=(18, 11), dpi=140)
        fig.suptitle(
            "Kapsamlı Model Değerlendirme ve Hata Denetimi Panosu (Diagnostic Dashboard)",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )

        n_classes = len(sinif_isimleri)
        renkler = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        # ----------------------------------------------------
        # PANEL 1: Normalleştirilmiş Karışıklık Matrisi
        # ----------------------------------------------------
        ax1 = eksenler[0, 0]
        cm = metrik_raporu["karisiklik_matrisi"]
        cm_norm = cm.astype('float') / np.maximum(1, cm.sum(axis=1)[:, np.newaxis])
        im1 = ax1.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ticks = np.arange(n_classes)
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(sinif_isimleri, rotation=30, ha="right", fontsize=8)
        ax1.set_yticks(ticks)
        ax1.set_yticklabels(sinif_isimleri, fontsize=8)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                yuzde = cm_norm[i, j] * 100.0
                adet = cm[i, j]
                ax1.text(
                    j, i, f"%{yuzde:.1f}\n({adet})",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                    fontweight="bold", fontsize=7,
                )

        ax1.set_title(f"1. Karışıklık Matrisi (Acc: %{metrik_raporu['dogruluk']*100:.1f})", fontsize=10, fontweight="bold")
        ax1.set_ylabel("Gerçek Sınıf")
        ax1.set_xlabel("Tahmin Edilen Sınıf")

        # ----------------------------------------------------
        # PANEL 2: Çok Sınıflı One-vs-Rest ROC Eğrileri
        # ----------------------------------------------------
        ax2 = eksenler[0, 1]
        roc_bilgi = metrik_raporu["roc_bilgi"]
        for c in range(n_classes):
            fpr, tpr, auc_val = roc_bilgi["sinif_roclari"][c]
            ax2.plot(fpr, tpr, label=f"{sinif_isimleri[c]} (AUC={auc_val:.2f})", color=renkler[c % len(renkler)], linewidth=1.5)

        fpr_m, tpr_m, auc_m = roc_bilgi["micro"]
        ax2.plot(fpr_m, tpr_m, label=f"Micro Avg (AUC={auc_m:.2f})", color="black", linestyle=":", linewidth=2)
        ax2.plot([0, 1], [0, 1], "k--", alpha=0.5)

        ax2.set_title(f"2. Çok Sınıflı ROC-AUC (Macro AUC: {roc_bilgi['macro_auc']:.3f})", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Yanlış Pozitif Oranı (FPR)")
        ax2.set_ylabel("Doğru Pozitif Oranı (TPR / Recall)")
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.legend(loc="lower right", fontsize=7, frameon=True)

        # ----------------------------------------------------
        # PANEL 3: Precision-Recall (PR) Eğrileri & AP
        # ----------------------------------------------------
        ax3 = eksenler[0, 2]
        pr_bilgi = metrik_raporu["pr_bilgi"]
        for c in range(n_classes):
            prec, rec, ap_val = pr_bilgi["sinif_prleri"][c]
            ax3.plot(rec, prec, label=f"{sinif_isimleri[c]} (AP={ap_val:.2f})", color=renkler[c % len(renkler)], linewidth=1.5)

        ax3.set_title(f"3. Precision-Recall Eğrisi (Macro AP: {pr_bilgi['macro_ap']:.3f})", fontsize=10, fontweight="bold")
        ax3.set_xlabel("Duyarlılık (Recall)")
        ax3.set_ylabel("Kesinlik (Precision)")
        ax3.grid(True, linestyle="--", alpha=0.4)
        ax3.legend(loc="lower left", fontsize=7, frameon=True)

        # ----------------------------------------------------
        # PANEL 4: Güvenilirlik Çizelgesi (Reliability Diagram & ECE)
        # ----------------------------------------------------
        ax4 = eksenler[1, 0]
        # İdeal çizgi
        ax4.plot([0, 1], [0, 1], "k--", label="İdeal Kalibrasyon", alpha=0.7)

        # Ham Model ECE
        ax4.plot(
            kalibrasyon_ham["bin_confs"], kalibrasyon_ham["bin_accs"],
            marker="o", color="#d62728", label=f"Ham Model (ECE={kalibrasyon_ham['ece']:.3f})", linewidth=1.5
        )

        # Sıcaklık Ölçekli ECE
        ax4.plot(
            kalibrasyon_kalibre["bin_confs"], kalibrasyon_kalibre["bin_accs"],
            marker="s", color="#2ca02c", label=f"Kalibre Model (ECE={kalibrasyon_kalibre['ece']:.3f})", linewidth=1.5
        )

        ax4.set_title("4. Olasılık Kalibrasyonu (Reliability Diagram)", fontsize=10, fontweight="bold")
        ax4.set_xlabel("Model Güveni (Confidence)")
        ax4.set_ylabel("Gerçek Doğruluk (Accuracy)")
        ax4.grid(True, linestyle="--", alpha=0.4)
        ax4.legend(loc="upper left", fontsize=7, frameon=True)

        # ----------------------------------------------------
        # PANEL 5: Top-k Doğruluk Dağılımı
        # ----------------------------------------------------
        ax5 = eksenler[1, 1]
        top_k = metrik_raporu["top_k"]
        k_etiketler = [f"Top-{k}" for k in top_k.keys()]
        k_degerler = [v * 100 for v in top_k.values()]

        bars = ax5.bar(k_etiketler, k_degerler, color="#1f77b4", width=0.4)
        ax5.set_title("5. Top-k Doğruluk Metrikleri (%)", fontsize=10, fontweight="bold")
        ax5.set_ylabel("Doğruluk Oranı (%)")
        ax5.set_ylim(0, 115)
        ax5.grid(True, linestyle="--", alpha=0.4, axis="y")

        for bar in bars:
            h = bar.get_height()
            ax5.annotate(
                f"%{h:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=9, fontweight="bold"
            )

        # ----------------------------------------------------
        # PANEL 6: Hata Denetimi (Overconfident Failures)
        # ----------------------------------------------------
        ax6 = eksenler[1, 2]
        if asiri_guvenli_hatalar:
            hata_metinleri = [
                f"#{h['ornek_indeks']} G:{sinif_isimleri[h['gercek_sinif']]} -> T:{sinif_isimleri[h['tahmin_sinif']]}"
                for h in asiri_guvenli_hatalar[:5]
            ]
            hata_guvenleri = [h["guven"] * 100 for h in asiri_guvenli_hatalar[:5]]

            y_pos = np.arange(len(hata_metinleri))
            ax6.barh(y_pos, hata_guvenleri, color="#d62728", alpha=0.85, height=0.5)
            ax6.set_yticks(y_pos)
            ax6.set_yticklabels(hata_metinleri, fontsize=8)
            ax6.set_xlabel("Model Güven Skoru (%)")
            ax6.set_xlim(0, 110)
            ax6.set_title("6. En Yüksek Güvenli Yanlış Tahminler (Audit)", fontsize=10, fontweight="bold")
            ax6.grid(True, linestyle="--", alpha=0.4, axis="x")

            for i, v in enumerate(hata_guvenleri):
                ax6.text(v + 1.5, i, f"%{v:.1f}", va="center", fontsize=8, fontweight="bold")
        else:
            ax6.text(0.5, 0.5, "Aşırı Güvenli Hata Bulunmadı!\n(Model Mükemmel Sınıflandırma Yaptı)",
                     ha="center", va="center", fontsize=10, color="green", fontweight="bold")
            ax6.axis("off")

        plt.tight_layout()
        fig.savefig(hedef_path, bbox_inches="tight")
        plt.close(fig)
        return hedef_path
