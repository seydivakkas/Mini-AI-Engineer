"""
Modern Regülerizasyon Görselleştiricisi
=======================================
Mixup, CutMix ve Label Smoothing deneylerinin görsel örneklerini,
eğitim kaybını, doğrulama başarımını, model kalibrasyonunu ve SWOT matrisini 6 panelli endüstriyel panoda birleştirir.
"""

from typing import Dict, Any
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class RegulerizasyonGorsellestirici:
    """
    Day 70 6-Panelli Modern Regülerizasyon Teşhis Panosu üreticisi.
    """

    @staticmethod
    def panoyu_ciz_ve_kaydet(
        laboratuvar_sonuclari: Dict[str, Any],
        cikti_yolu: str = "ciktilar/modern_regulerizasyon_paneli.png"
    ) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(cikti_yolu)), exist_ok=True)

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(2, 3, figsize=(21, 13))
        fig.suptitle(
            "Day 70: Mixup, CutMix ve Label Smoothing Modern Düzenlileştirme Laboratuvar Paneli",
            fontsize=17,
            fontweight="bold",
            color="#1a252f",
            y=0.98
        )

        d1 = laboratuvar_sonuclari["deney_1"]
        d2 = laboratuvar_sonuclari["deney_2"]
        d3 = laboratuvar_sonuclari["deney_3"]
        toplam_epoch = laboratuvar_sonuclari["toplam_epoch"]
        epochs = list(range(1, toplam_epoch + 1))
        ornekler = laboratuvar_sonuclari["ornekler"]

        # -------------------------------------------------------------
        # 1. Panel: Yönetici & Kalibrasyon Özet Kartı
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.axis("off")
        ozet_metin = (
            "      REGULERIZASYON LABORATUVARI OZETI\n"
            "═══════════════════════════════════════════════════════\n"
            f" 1. {d1['deney_adi']:<23} : Acc: %{d1['son_val_accuracy']:.1f} | Guven: {d1['son_ort_guven']:.3f}\n"
            f" 2. {d2['deney_adi']:<23} : Acc: %{d2['son_val_accuracy']:.1f} | Guven: {d2['son_ort_guven']:.3f}\n"
            f" 3. {d3['deney_adi']:<23} : Acc: %{d3['son_val_accuracy']:.1f} | Guven: {d3['son_ort_guven']:.3f}\n"
            "───────────────────────────────────────────────────────\n"
            " * Mixup İnterpolasyonu  : Karar sınırlarını lineerleştirdi\n"
            " * CutMix Yama Aktarımı  : Yerel özelliklere odaklanmayı önledi\n"
            " * Label Smoothing (e=0.1): Aşırı güveni (Overconfidence) kırdı\n"
            "═══════════════════════════════════════════════════════\n"
            " * Uretim Standardi: ViT & CNN Genellestirme Artirildi"
        )
        ax1.text(
            0.5, 0.5, ozet_metin,
            transform=ax1.transAxes,
            fontsize=10.0,
            family="monospace",
            verticalalignment="center",
            horizontalalignment="center",
            bbox=dict(boxstyle="round,pad=1.2", facecolor="#e8f8f5", edgecolor="#1abc9c", linewidth=2.0)
        )
        ax1.set_title("1. Regulerizasyon Laboratuvar Ozeti", fontweight="bold", color="#16a085")

        # -------------------------------------------------------------
        # 2. Panel: Mixup & CutMix Görsel Örnekleri
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.axis("off")

        # Normalize edip yan yana göster
        def norm_img(img_chw: np.ndarray) -> np.ndarray:
            img = np.transpose(img_chw, (1, 2, 0))
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)
            return img

        orig_img = norm_img(ornekler["orijinal"])
        mix_img = norm_img(ornekler["mixup"])
        cut_img = norm_img(ornekler["cutmix"])

        # Yan yana 3 görseli birleştir
        birlestirilmis = np.hstack([orig_img, np.ones((orig_img.shape[0], 2, 3)), mix_img, np.ones((orig_img.shape[0], 2, 3)), cut_img])
        ax2.imshow(birlestirilmis)
        ax2.set_title(f"2. Gorsel Ornekler (Orijinal | Mixup λ={ornekler['mix_lam']} | CutMix λ={ornekler['cut_lam']})", fontweight="bold", color="#2980b9", fontsize=10)

        # -------------------------------------------------------------
        # 3. Panel: Eğitim Kaybı (Train Loss) Yakınsaması
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.plot(epochs, d1["gecmis"]["train_loss"], label=f"{d1['deney_adi']}", color="#e74c3c", linestyle="--", linewidth=2.0)
        ax3.plot(epochs, d2["gecmis"]["train_loss"], label=f"{d2['deney_adi']}", color="#3498db", linewidth=2.5)
        ax3.plot(epochs, d3["gecmis"]["train_loss"], label=f"{d3['deney_adi']}", color="#2ecc71", linewidth=2.5)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Egitim Kaybi (Loss)")
        ax3.set_title("3. Egitim Kaybi Yakinsama Dinamigi", fontweight="bold", color="#8e44ad")
        ax3.legend(loc="upper right")

        # -------------------------------------------------------------
        # 4. Panel: Doğrulama Başarımı (Validation Accuracy %)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.plot(epochs, d1["gecmis"]["val_accuracy"], label=f"{d1['deney_adi']}", color="#e74c3c", marker="x", linestyle="--", linewidth=1.8)
        ax4.plot(epochs, d2["gecmis"]["val_accuracy"], label=f"{d2['deney_adi']}", color="#3498db", marker="o", linewidth=2.2)
        ax4.plot(epochs, d3["gecmis"]["val_accuracy"], label=f"{d3['deney_adi']}", color="#2ecc71", marker="s", linewidth=2.2)
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Dogruluk / Accuracy (%)")
        ax4.set_title("4. Dogrulama Basarimi Karsilastirmasi", fontweight="bold", color="#d35400")
        ax4.legend(loc="lower right")

        # -------------------------------------------------------------
        # 5. Panel: Model Kalibrasyonu ve Güven Skoru (Confidence)
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.plot(epochs, d1["gecmis"]["mean_confidence"], label=f"{d1['deney_adi']} (Overconfident)", color="#e74c3c", linestyle="--", linewidth=2.0)
        ax5.plot(epochs, d2["gecmis"]["mean_confidence"], label=f"{d2['deney_adi']} (Calibrated)", color="#3498db", linewidth=2.2)
        ax5.plot(epochs, d3["gecmis"]["mean_confidence"], label=f"{d3['deney_adi']} (Calibrated)", color="#2ecc71", linewidth=2.2)
        ax5.axhline(0.85, color="gray", linestyle=":", label="Ideal Kalibrasyon Esigi (0.85)")
        ax5.set_xlabel("Epoch")
        ax5.set_ylabel("Ortalama En Yuksek Olasilik (Guven)")
        ax5.set_title("5. Asiri Guven (Overconfidence) Analizi", fontweight="bold", color="#27ae60")
        ax5.legend(loc="lower right")

        # -------------------------------------------------------------
        # 6. Panel: SWOT Karar Matrisi
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        swot_metin = (
            " MODERN REGULERIZASYON SWOT KARAR MATRISI\n"
            "─────────────────────────────────────────────────\n"
            " [S] GUCLU YONLER (Strengths):\n"
            " • Mixup: Lineer davranisi tesvik eder, karar sinirlarini genisletir\n"
            " • CutMix: Modeli yerel ipuclari yerine butunsel nesneye odaklar\n"
            " • Label Smoothing: Asiri logit patlamasini ve overconfidence'i kirar\n\n"
            " [W] ZAYIF YONLER (Weaknesses):\n"
            " • Egitim baslangicinda kayip degerlerinin daha yuksek kalmasi\n"
            " • Daha uzun epoch egitimi gerektirmesi\n\n"
            " [O] FIRSATLAR (Opportunities):\n"
            " • Vision Transformer (ViT) egitiminde asiri uyumu (overfitting) sifirlama\n"
            " • OOD (Dagitim Disi) ve advers gorsellere karsi direnc artisi\n\n"
            " [T] TEHDITLER (Threats):\n"
            " • Cok kucuk modellerde asiri duzenlileştirme (underfitting) riski"
        )
        ax6.text(
            0.5, 0.5, swot_metin,
            transform=ax6.transAxes,
            fontsize=9.2,
            family="monospace",
            verticalalignment="center",
            horizontalalignment="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#fef9e7", edgecolor="#f39c12", linewidth=1.8)
        )
        ax6.set_title("6. Regulerizasyon SWOT Matrisi", fontweight="bold", color="#d35400")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(cikti_yolu, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return cikti_yolu
