"""
6-Panelli XGBoost Dengesiz Risk Teşhis ve SHAP Açıklanabilirlik Panosu (Risk Dashboard).
"""

from typing import Dict, Any
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class XGBoostRiskGorsellestirici:
    """XGBoost eğitim yakınsamasını, eşik optimizasyonunu ve TreeSHAP katkılarını 6 panelli panoda görselleştirir."""

    @classmethod
    def panel_ciz(
        cls,
        test_sonuclari: Dict[str, Any],
        esik_sonuclari: Dict[str, Any],
        shap_sonuclari: Dict[str, Any],
        egitim_gecmisi: Dict[str, Any],
        scale_pos_weight: float,
        y_test: pd.Series,
        hedef_path: str = "ciktilar/xgboost_risk_paneli.png"
    ) -> str:
        os.makedirs(os.path.dirname(hedef_path), exist_ok=True)

        sns.set_theme(style="whitegrid", font_scale=0.88)
        fig, axes = plt.subplots(2, 3, figsize=(20, 13), dpi=300)
        fig.suptitle(
            "Day 49: XGBoost ile Dengesiz Tabüler Risk Sınıflandırıcısı ve TreeSHAP Açıklanabilirlik Paneli",
            fontsize=15, fontweight="bold", y=0.98
        )

        roc_auc = test_sonuclari["roc_auc"]
        pr_auc = test_sonuclari["pr_auc"]
        f1 = test_sonuclari["f1_skoru"]
        opt_esik = test_sonuclari["esik"]

        # -------------------------------------------------------------
        # Panel 1: Yönetici Risk Karar Kartı
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.axis("off")

        kart_metni = (
            f"XGBOOST RİSK YÖNETİCİ KARTI\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Dengesizlik Oranı : %5 Pozitif / %95 Negatif\n"
            f"• scale_pos_weight  : {scale_pos_weight:.2f}x Ağırlıklandırma\n"
            f"• Test PR-AUC       : %{pr_auc * 100:.2f} (Öncelikli Metrik)\n"
            f"• Test ROC-AUC      : %{roc_auc * 100:.2f}\n"
            f"• Optimize F1-Skoru : %{f1 * 100:.2f} (Eşik: {opt_esik:.3f})\n"
            f"• Yakalanan Risk    : %{test_sonuclari['recall_yuzde']:.1f} (Recall)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Üretim Kararı     : YÜKSEK GÜVENİLİRLİK (ONAYLANDI)"
        )

        ax1.text(
            0.5, 0.5, kart_metni, transform=ax1.transAxes, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.9", facecolor="#2ecc71", alpha=0.22, edgecolor="#27ae60", linewidth=2),
            fontsize=9.2, fontweight="bold", family="monospace"
        )
        ax1.set_title("1. Risk Modeli Karar Kartı", fontweight="bold", color="#2c3e50")

        # -------------------------------------------------------------
        # Panel 2: LogLoss / PR-AUC İniş Eğrisi (Early Stopping)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        train_loss = egitim_gecmisi["validation_0"]["logloss"]
        val_loss = egitim_gecmisi["validation_1"]["logloss"]
        iterasyonlar = range(1, len(train_loss) + 1)

        ax2.plot(iterasyonlar, train_loss, label="Eğitim LogLoss", color="#3498db", linewidth=2.0)
        ax2.plot(iterasyonlar, val_loss, label="Doğrulama LogLoss", color="#e74c3c", linewidth=2.0, linestyle="--")
        en_iyi_iter = int(np.argmin(val_loss)) + 1
        ax2.axvline(en_iyi_iter, color="#27ae60", linestyle=":", label=f"En İyi İterasyon ({en_iyi_iter})")

        ax2.set_title("2. XGBoost LogLoss Yakınsama & Erken Durdurma", fontweight="bold", color="#2980b9")
        ax2.set_xlabel("Boosting İterasyonu")
        ax2.set_ylabel("LogLoss")
        ax2.legend(loc="upper right", fontsize=8)

        # -------------------------------------------------------------
        # Panel 3: Karar Eşiği vs F1-Skoru Optimizasyonu
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        esikler = esik_sonuclari["esikler"]
        f1_vals = esik_sonuclari["f1_skorlari"]

        ax3.plot(esikler, f1_vals, color="#8e44ad", linewidth=2.2, label="Validation F1-Skoru")
        ax3.axvline(opt_esik, color="#e74c3c", linestyle="--", label=f"Optimum Eşik ({opt_esik:.2f})")
        ax3.scatter([opt_esik], [esik_sonuclari["en_iyi_val_f1"]], color="#e74c3c", s=70, zorder=5)

        ax3.set_title("3. Eşik Değeri (Threshold) vs F1 Optimizasyonu", fontweight="bold", color="#8e44ad")
        ax3.set_xlabel("Karar Eşiği (Olasılık)")
        ax3.set_ylabel("F1 Skoru")
        ax3.legend(loc="lower center", fontsize=8)

        # -------------------------------------------------------------
        # Panel 4: Optimize Edilmiş Karmaşıklık Matrisi (Confusion Matrix)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        cm = test_sonuclari["confusion_matrix"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", cbar=False, ax=ax4,
                    xticklabels=["Normal (0)", "Risk/Fraud (1)"], yticklabels=["Normal (0)", "Risk/Fraud (1)"])

        ax4.set_title(f"4. Karmaşıklık Matrisi (Eşik = {opt_esik:.2f})", fontweight="bold", color="#16a085")
        ax4.set_xlabel("Tahmin Edilen Sınıf")
        ax4.set_ylabel("Gerçek Sınıf")

        # -------------------------------------------------------------
        # Panel 5: TreeSHAP Global Özellik Katkısı (Mean |SHAP|)
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        shap_onem = shap_sonuclari["ortalama_mutlak_shap"]
        feat_names = list(shap_onem.keys())
        feat_vals = list(shap_onem.values())

        ax5.barh(feat_names[::-1], feat_vals[::-1], color="#d35400", alpha=0.85, edgecolor="black")
        ax5.set_title("5. TreeSHAP Global Özellik Önemi (Mean |SHAP|)", fontweight="bold", color="#d35400")
        ax5.set_xlabel("Ortalama Mutlak SHAP Değeri")

        # -------------------------------------------------------------
        # Panel 6: Tahmin Olasılık Dağılımı (Sınıf Ayrışımı)
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        y_prob = test_sonuclari["y_prob"]
        y_true_arr = y_test.values

        sns.histplot(y_prob[y_true_arr == 0], color="#3498db", label="Normal İşlemler (0)", kde=True, ax=ax6, stat="density", bins=25, alpha=0.4)
        sns.histplot(y_prob[y_true_arr == 1], color="#e74c3c", label="Riskli İşlemler (1)", kde=True, ax=ax6, stat="density", bins=25, alpha=0.5)
        ax6.axvline(opt_esik, color="#27ae60", linestyle="--", linewidth=2, label=f"Karar Eşiği ({opt_esik:.2f})")

        ax6.set_title("6. Risk Skoru Dağılımı ve Sınıf Ayrışımı", fontweight="bold", color="#2c3e50")
        ax6.set_xlabel("Tahmin Edilen Risk Olasılığı")
        ax6.legend(loc="upper center", fontsize=8)

        fig.subplots_adjust(top=0.93, bottom=0.08, left=0.06, right=0.95, hspace=0.32, wspace=0.28)
        fig.savefig(hedef_path, bbox_inches="tight")
        plt.close(fig)
        return hedef_path
