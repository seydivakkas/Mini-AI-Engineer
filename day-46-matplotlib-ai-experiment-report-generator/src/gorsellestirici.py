"""
6-Panelli Kapsamlı AI Deney Raporlama ve Teşhis Panosu (Experiment Report Dashboard).
"""

from typing import Dict, Any, Optional
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class DeneyRaporuGorsellestirici:
    """Model eğitim ve değerlendirme telemetrisini 6 panelli panoda görselleştirir."""

    @classmethod
    def panel_ciz(
        cls,
        egitim_gecmisi: Any,
        egitim_analizi: Dict[str, Any],
        cm_analizi: Dict[str, Any],
        roc_analizi: Dict[str, Any],
        pr_analizi: Dict[str, Any],
        hedef_path: str = "ciktilar/deney_raporu_paneli.png"
    ) -> str:
        os.makedirs(os.path.dirname(hedef_path), exist_ok=True)

        sns.set_theme(style="whitegrid", font_scale=0.88)
        fig, axes = plt.subplots(2, 3, figsize=(20, 13), dpi=300)
        fig.suptitle(
            f"Day 46: Otomatik AI Deney ve Performans Teşhis Raporu ({egitim_analizi.get('model_adi', 'Model')})",
            fontsize=15, fontweight="bold", y=0.98
        )

        epochs = egitim_gecmisi.epochlar
        en_iyi_ep = egitim_analizi["en_iyi_epoch"]

        # -------------------------------------------------------------
        # Panel 1: Loss & Yakınsama Eğrisi
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.plot(epochs, egitim_gecmisi.train_loss, label="Eğitim Kaybı (Train Loss)", color="#2980b9", linewidth=2.0)
        ax1.plot(epochs, egitim_gecmisi.val_loss, label="Doğrulama Kaybı (Val Loss)", color="#e74c3c", linewidth=2.0, linestyle="--")
        ax1.axvline(en_iyi_ep, color="#27ae60", linestyle=":", label=f"En İyi Epoch ({en_iyi_ep})")
        ax1.scatter([en_iyi_ep], [egitim_analizi["en_iyi_val_loss"]], color="#27ae60", s=60, zorder=5)

        ax1.set_title("1. Kayıp (Loss) Yakınsama Eğrisi", fontweight="bold", color="#1f77b4")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Kayıp (Loss)")
        ax1.legend(loc="upper right", fontsize=8)

        # -------------------------------------------------------------
        # Panel 2: Doğruluk (Accuracy) Eğrisi
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.plot(epochs, [a * 100 for a in egitim_gecmisi.train_acc], label="Eğitim Acc (%)", color="#3498db", linewidth=2.0)
        ax2.plot(epochs, [a * 100 for a in egitim_gecmisi.val_acc], label="Doğrulama Acc (%)", color="#2ecc71", linewidth=2.0, linestyle="--")
        ax2.axvline(en_iyi_ep, color="#27ae60", linestyle=":")

        ax2.set_title("2. Doğruluk (Accuracy) Seyri", fontweight="bold", color="#27ae60")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Doğruluk (%)")
        ax2.legend(loc="lower right", fontsize=8)

        # -------------------------------------------------------------
        # Panel 3: ROC Eğrisi ve AUC Alanı
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        fpr = roc_analizi["fpr"]
        tpr = roc_analizi["tpr"]
        auc = roc_analizi["roc_auc"]

        ax3.plot(fpr, tpr, color="#8e44ad", linewidth=2.2, label=f"Model ROC (AUC = {auc:.3f})")
        ax3.plot([0, 1], [0, 1], color="#7f8c8d", linestyle="--", label="Rastgele Sınıflandırıcı (0.50)")
        ax3.fill_between(fpr, tpr, color="#8e44ad", alpha=0.15)

        ax3.set_title("3. Alıcı İşletim Karakteristiği (ROC Eğrisi)", fontweight="bold", color="#8e44ad")
        ax3.set_xlabel("Yanlış Pozitif Oranı (FPR)")
        ax3.set_ylabel("Doğru Pozitif Oranı (TPR)")
        ax3.legend(loc="lower right", fontsize=8)

        # -------------------------------------------------------------
        # Panel 4: Precision-Recall (PR) Eğrisi
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        rec = pr_analizi["recall"]
        prec = pr_analizi["precision"]
        ap = pr_analizi["average_precision_ap"]
        base_rate = pr_analizi["taban_oran"]

        ax4.plot(rec, prec, color="#d35400", linewidth=2.2, label=f"Model PR (AP = {ap:.3f})")
        ax4.axhline(base_rate, color="#7f8c8d", linestyle="--", label=f"Taban Oran ({base_rate:.2f})")
        ax4.fill_between(rec, prec, color="#d35400", alpha=0.15)

        ax4.set_title("4. Kesinlik - Duyarlılık (PR Eğrisi)", fontweight="bold", color="#d35400")
        ax4.set_xlabel("Duyarlılık (Recall)")
        ax4.set_ylabel("Kesinlik (Precision)")
        ax4.legend(loc="lower left", fontsize=8)

        # -------------------------------------------------------------
        # Panel 5: Normalize Karmaşıklık Matrisi (Confusion Matrix)
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        cm = np.array(cm_analizi["matris"])
        cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", cbar=False, ax=ax5,
                    xticklabels=["Negatif (0)", "Pozitif (1)"], yticklabels=["Negatif (0)", "Pozitif (1)"])

        ax5.set_title("5. Karmaşıklık Matrisi (Confusion Matrix)", fontweight="bold", color="#2980b9")
        ax5.set_xlabel("Tahmin Edilen Sınıf")
        ax5.set_ylabel("Gerçek Sınıf")

        # -------------------------------------------------------------
        # Panel 6: Yönetici Deney Karar Kartı ve KPI'lar
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")

        f1 = cm_analizi["f1_skoru"]
        overfitting = egitim_analizi["overfitting_gap"]
        bg_c = "#2ecc71" if auc >= 0.85 and f1 >= 0.80 else "#f39c12" if overfitting < 0.15 else "#e74c3c"
        karar = "ÜRETİME HAZIR (PROD_READY)" if auc >= 0.85 and f1 >= 0.80 else "AŞIRI ÖĞRENME RİSKİ" if overfitting >= 0.15 else "GELİŞTİRME GEREKLİ"

        karar_karti = (
            f"YÖNETİCİ DENEY KARARI\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Karar             : {karar}\n"
            f"• ROC-AUC Skoru     : %{auc*100:.2f}\n"
            f"• F1-Skoru          : %{f1*100:.2f} (MCC: {cm_analizi['mcc_skoru']:.3f})\n"
            f"• Doğruluk (Acc)    : %{cm_analizi['dogruluk_acc']:.2f}\n"
            f"• En İyi Epoch      : {en_iyi_ep} (Val Loss: {egitim_analizi['en_iyi_val_loss']})\n"
            f"• Overfitting Farkı : {overfitting:.4f} ({egitim_analizi['overfitting_riski']})\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• HTML Raporu       : ciktilar/deney_raporu.html"
        )

        ax6.text(
            0.5, 0.5, karar_karti, transform=ax6.transAxes, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.9", facecolor=bg_c, alpha=0.25, edgecolor=bg_c, linewidth=2),
            fontsize=9.2, fontweight="bold", family="monospace"
        )
        ax6.set_title("6. Yönetici Model Değerlendirme Kartı", fontweight="bold", color="#2c3e50")

        fig.subplots_adjust(top=0.93, bottom=0.08, left=0.06, right=0.95, hspace=0.32, wspace=0.28)
        fig.savefig(hedef_path, bbox_inches="tight")
        plt.close(fig)
        return hedef_path
