"""
Optuna HPO ve TPE Optimizasyon Teşhis Panosu
--------------------------------------------
6 panelli yüksek çözünürlüklü Optuna TPE Mimarisi, Optimizasyon Geçmişi,
Erken Budanan Denemeler, Hiperparametre Önem Dereceleri ve SWOT Karar Matrisi.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, List, Any
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import optuna


class OptunaGorsellestirici:
    """
    Optuna çalışma çıktılarını ve TPE dinamiklerini görselleştiren teşhis panosu.
    """
    def __init__(self, stil: str = "seaborn-v0_8-whitegrid"):
        try:
            plt.style.use(stil)
        except Exception:
            sns.set_theme(style="whitegrid")

    def olustur_hpo_paneli(
        self,
        study: optuna.Study,
        ozet: Dict[str, Any],
        kayit_yolu: str
    ) -> str:
        """
        6 panelli kapsamlı HPO & TPE teşhis panosunu oluşturur.
        """
        fig, axes = plt.subplots(2, 3, figsize=(22, 12), dpi=300)
        fig.suptitle(
            "Day 88: Optuna ile Otomatik Hiperparametre Optimizasyonu (TPE Algoritması & Pruning) Paneli",
            fontsize=18,
            fontweight="bold",
            y=0.98
        )

        df_trials = ozet["df_trials"]

        # -------------------------------------------------------------
        # PANEL 1: TPE ve Budama Mimarisi
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.axis("off")
        
        kavram_metin = (
            "         TPE ALGORİTMASI VE ERKEN BUDAMA MİMARİSİ\n"
            "─────────────────────────────────────────────────────────────\n"
            "  1. TREE-STRUCTURED PARZEN ESTIMATOR (Bergstra et al.):\n"
            "     • P(x|y) olasılığını iki gruba ayırır (γ eşiğiyle):\n"
            "       - l(x): Başarılı hiperparametreler kümesi\n"
            "       - g(x): Başarısız hiperparametreler kümesi\n"
            "     • Amaç: l(x) / g(x) yoğunluk oranını maksimize etmek.\n\n"
            "  2. MEDIAN PRUNER (Hesaplama Gücü Tasarrufu):\n"
            "     • Önceki koşuların medyanından kötü olan denemeler 2. veya\n"
            "       3. epokta otomatik durdurulur (TrialPruned).\n"
            "     • GPU/Hesaplama süresinde %60-%80 net tasarruf sağlar."
        )
        ax1.text(
            0.5, 0.5, kavram_metin,
            fontsize=8.5,
            family="monospace",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor="#ebf8ff", edgecolor="#3182ce", linewidth=1.8)
        )
        ax1.set_title("1. TPE ve Medyan Budama (Pruning) Mimarisi", fontsize=12, fontweight="bold", color="#2b6cb0")

        # -------------------------------------------------------------
        # PANEL 2: Optimizasyon Geçmişi (Optimization History)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        tamamlanan = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        budanan = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

        if tamamlanan:
            t_num = [t.number for t in tamamlanan]
            t_val = [t.value for t in tamamlanan]
            ax2.scatter(t_num, t_val, color="#3182ce", s=80, alpha=0.85, label=f"Tamamlanan ({len(tamamlanan)})")

            # En iyi değer ilerleme çizgisi
            en_iyi_adimlari = []
            cur_best = float("inf")
            for t in study.trials:
                if t.state == optuna.trial.TrialState.COMPLETE:
                    if t.value < cur_best:
                        cur_best = t.value
                en_iyi_adimlari.append(cur_best)

            ax2.plot(range(len(en_iyi_adimlari)), en_iyi_adimlari, "r--", linewidth=2.2, label=f"En İyi Kayıp: {ozet['en_iyi_deger']:.4f}")

        if budanan:
            b_num = [t.number for t in budanan]
            b_val = [t.intermediate_values[max(t.intermediate_values.keys())] for t in budanan if t.intermediate_values]
            if b_val:
                ax2.scatter(b_num[:len(b_val)], b_val, color="#e53e3e", marker="x", s=80, label=f"Budanan ({len(budanan)})")

        ax2.set_title("2. Optimizasyon Geçmişi (Trial vs Validation Loss)", fontsize=12, fontweight="bold", color="#c53030")
        ax2.set_xlabel("Deneme Numarası (Trial #)", fontsize=10)
        ax2.set_ylabel("Validation Loss (Düşük = İyi)", fontsize=10)
        ax2.legend(loc="upper right", frameon=True, fontsize=8.5)

        # -------------------------------------------------------------
        # PANEL 3: Deneme Durumları Dağılımı (Pasta Grafiği)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        durum_sayilari = [len(tamamlanan), len(budanan)]
        durum_etiketleri = [f"Tamamlanan ({len(tamamlanan)})", f"Erken Budanan ({len(budanan)})"]
        renkler = ["#38a169", "#e53e3e"]

        ax3.pie(
            durum_sayilari,
            labels=durum_etiketleri,
            autopct="%1.1f%%",
            startangle=140,
            colors=renkler,
            wedgeprops=dict(edgecolor="white", linewidth=2)
        )
        ax3.set_title(f"3. Deneme Durumları (Toplam: {len(study.trials)})", fontsize=12, fontweight="bold", color="#22543d")

        # -------------------------------------------------------------
        # PANEL 4: Hiperparametre Önem Dereceleri (Parameter Importances)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        onemler = ozet.get("param_importances", {})
        if onemler:
            p_names = list(onemler.keys())
            p_vals = list(onemler.values())
            y_pos = np.arange(len(p_names))

            barlar = ax4.barh(y_pos, p_vals, color="#805ad5", edgecolor="#2d3748", height=0.55)
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(p_names, fontsize=9)
            ax4.invert_yaxis()
            ax4.set_xlabel("Göreli Önem Oranı (Importance)", fontsize=10)
            ax4.set_title("4. Hiperparametre Önem Dereceleri", fontsize=12, fontweight="bold", color="#553c9a")

            for bar in barlar:
                w = bar.get_width()
                ax4.text(w + 0.01, bar.get_y() + bar.get_height()/2.0, f"%{w*100:.1f}", va="center", fontsize=8.5, fontweight="bold")
        else:
            ax4.text(0.5, 0.5, "Yetersiz deneme sayısı nedeniyle önem hesaplanamadı", ha="center", va="center")

        # -------------------------------------------------------------
        # PANEL 5: Öğrenme Oranı ve Kayıp Dağılımı
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        if "params_lr" in df_trials.columns and "value" in df_trials.columns:
            complete_df = df_trials[df_trials["state"] == "COMPLETE"]
            if not complete_df.empty:
                sns.scatterplot(
                    data=complete_df,
                    x="params_lr",
                    y="value",
                    hue="params_optimizer" if "params_optimizer" in complete_df.columns else None,
                    style="params_taban_kanal" if "params_taban_kanal" in complete_df.columns else None,
                    s=120,
                    ax=ax5,
                    palette="tab10"
                )
                ax5.set_xscale("log")
                ax5.set_title("5. Öğrenme Oranı (LR) vs Validation Loss", fontsize=12, fontweight="bold", color="#2c5282")
                ax5.set_xlabel("Öğrenme Oranı (Log Ölçek)", fontsize=10)
                ax5.set_ylabel("Validation Loss", fontsize=10)
                ax5.legend(loc="upper right", fontsize=8, frameon=True)

        # -------------------------------------------------------------
        # PANEL 6: SWOT Karar Matrisi
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        
        swot_metni = (
            "         OPTUNA HPO & TPE ALGORİTMASI SWOT MATRİSİ\n"
            "───────────────────────────────────────────────────────────────────\n"
            "  [S] GÜÇLÜ YÖNLER (Strengths):\n"
            "  • Bayesyen TPE örnekleyici ile hızlı ve akıllı arama.\n"
            "  • MedianPruner ile umutsuz koşuları erken durdurarak %70 tasarruf.\n"
            "  • Çoklu parametre tiplerini (kategorik, float, log) doğal destekler.\n\n"
            "  [W] ZAYIF YÖNLER (Weaknesses):\n"
            "  • İlk 5-10 denemede rastgele örnekleme gerektirir (Isınma fazı).\n"
            "  • Pruner eşiği çok agresifse yavaş öğrenen iyi modeller budanabilir.\n\n"
            "  [O] FIRSATLAR (Opportunities):\n"
            "  • Dağıtık ortamda (PostgreSQL / Redis) çoklu GPU ile paralel HPO.\n"
            "  • Model Registry ve CI/CD ile en iyi checkpoint'i doğrudan üretime alma.\n\n"
            "  [T] TEHDİTLER (Threats):\n"
            "  • Arama uzayı (Search Space) çok geniş seçilirse arama süresi uzar."
        )
        
        ax6.text(
            0.5, 0.5, swot_metni,
            fontsize=8.5,
            family="monospace",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor="#f7fafc", edgecolor="#4a5568", linewidth=1.8)
        )
        ax6.set_title("6. Optuna HPO & TPE SWOT Karar Matrisi", fontsize=12, fontweight="bold", color="#2d3748")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)
        plt.savefig(kayit_yolu, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return kayit_yolu
