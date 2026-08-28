"""
MLOps Deney Takibi ve Karşılaştırma Teşhis Panosu
-------------------------------------------------
6 panelli yüksek çözünürlüklü Deney Mimarisi, Çoklu Koşu Zaman Serileri (Loss & Acc),
Pareto Verimlilik Sınırı, Model Lider Tablosu ve MLOps SWOT Karar Matrisi.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, List, Any
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class MLOpsGorsellestirici:
    """
    MLflow ve W&B deney sonuçlarını görselleştiren teşhis paneli.
    """
    def __init__(self, stil: str = "seaborn-v0_8-whitegrid"):
        try:
            plt.style.use(stil)
        except Exception:
            sns.set_theme(style="whitegrid")

    def olustur_deney_paneli(
        self,
        kosular: List[Dict[str, Any]],
        df_liderlik: pd.DataFrame,
        kayit_yolu: str
    ) -> str:
        """
        6 panelli kapsamlı MLOps Deney Takibi ve Liderlik Panosunu oluşturur.
        """
        fig, axes = plt.subplots(2, 3, figsize=(22, 12), dpi=300)
        fig.suptitle(
            "Day 87: MLflow / Weights & Biases ile Merkezi Deney Takibi ve Artefakt Kayıt Sistemi Paneli",
            fontsize=18,
            fontweight="bold",
            y=0.98
        )

        renk_paleti = sns.color_palette("tab10", len(kosular))

        # -------------------------------------------------------------
        # PANEL 1: Deney Takip Mimarisi
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.axis("off")
        
        kavram_metin = (
            "      MLFLOW & W&B MERKEZİ DENEY TAKİP MİMARİSİ\n"
            "─────────────────────────────────────────────────────────────\n"
            "  1. EXPERIMENT & RUN HİYERARŞİSİ:\n"
            "     • Deney (Experiment): Mantıksal proje havuzu.\n"
            "     • Koşu (Run): Tekil model eğitimi (Benzersiz UUID).\n\n"
            "  2. ÜÇ TEMEL KAYIT BİLEŞENİ:\n"
            "     • Parametreler (Params): lr, optimizer, weight_decay, seed.\n"
            "     • Metrikler (Metrics): Step/Epok bazlı Loss, Accuracy, GPU.\n"
            "     • Artefaktlar (Artifacts): Model ağırlıkları (.pt), JSON.\n\n"
            "  3. REPRODUCIBILITY & GOVERNANCE:\n"
            "     • Git Commit Hash + Seed + Donanım metaverisi kaydı.\n"
            "     • Geçmişe dönük tam yeniden üretilebilirlik güvencesi."
        )
        ax1.text(
            0.5, 0.5, kavram_metin,
            fontsize=8.5,
            family="monospace",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor="#ebf8ff", edgecolor="#3182ce", linewidth=1.8)
        )
        ax1.set_title("1. Merkezi Deney Takip & MLOps Mimarisi", fontsize=12, fontweight="bold", color="#2b6cb0")

        # -------------------------------------------------------------
        # PANEL 2: Çoklu Koşuların Doğrulama Kayıp Zaman Serileri
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        for idx, k in enumerate(kosular):
            r_name = k["tags"].get("run_name", k["run_id"])
            if "val_loss" in k["metric_history"]:
                steps = [m["step"] for m in k["metric_history"]["val_loss"]]
                losses = [m["value"] for m in k["metric_history"]["val_loss"]]
                ax2.plot(steps, losses, "-o", color=renk_paleti[idx], linewidth=2.0, label=r_name)

        ax2.set_title("2. Doğrulama Kayıp (Val Loss) Zaman Serisi", fontsize=12, fontweight="bold", color="#c53030")
        ax2.set_xlabel("Epok (Step)", fontsize=10)
        ax2.set_ylabel("Validation Loss", fontsize=10)
        ax2.legend(loc="upper right", fontsize=8, frameon=True)

        # -------------------------------------------------------------
        # PANEL 3: Çoklu Koşuların Doğrulama Başarım Zaman Serileri
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        for idx, k in enumerate(kosular):
            r_name = k["tags"].get("run_name", k["run_id"])
            if "val_acc" in k["metric_history"]:
                steps = [m["step"] for m in k["metric_history"]["val_acc"]]
                accs = [m["value"] for m in k["metric_history"]["val_acc"]]
                ax3.plot(steps, accs, "-s", color=renk_paleti[idx], linewidth=2.0, label=r_name)

        ax3.set_title("3. Doğrulama Başarım (Val Accuracy %) Zaman Serisi", fontsize=12, fontweight="bold", color="#22543d")
        ax3.set_xlabel("Epok (Step)", fontsize=10)
        ax3.set_ylabel("Validation Accuracy (%)", fontsize=10)
        ax3.set_ylim(0, 105)
        ax3.legend(loc="lower right", fontsize=8, frameon=True)

        # -------------------------------------------------------------
        # PANEL 4: Pareto Sınırı (Accuracy vs Parametre Sayısı)
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        if "p_param_count" in df_liderlik.columns and "m_val_acc" in df_liderlik.columns:
            ax4.scatter(
                df_liderlik["p_param_count"],
                df_liderlik["m_val_acc"],
                c="#3182ce",
                s=120,
                alpha=0.85,
                edgecolors="black",
                zorder=3
            )
            for _, r in df_liderlik.iterrows():
                ax4.annotate(
                    f"{r['run_name']}\n({r['p_optimizer']})",
                    (r["p_param_count"], r["m_val_acc"]),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=7.5,
                    fontweight="bold"
                )

        ax4.set_title("4. Model Verimliliği & Pareto Cephesi", fontsize=12, fontweight="bold", color="#553c9a")
        ax4.set_xlabel("Parametre Sayısı (Count)", fontsize=10)
        ax4.set_ylabel("En İyi Val Accuracy (%)", fontsize=10)
        ax4.set_ylim(0, 110)

        # -------------------------------------------------------------
        # PANEL 5: Liderlik Tablosu (Leaderboard Bar Chart)
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        if not df_liderlik.empty and "m_val_acc" in df_liderlik.columns:
            isimler = df_liderlik["run_name"].tolist()
            basarimlar = df_liderlik["m_val_acc"].tolist()
            y_pos = np.arange(len(isimler))

            barlar = ax5.barh(y_pos, basarimlar, color="#38a169", edgecolor="#2d3748", height=0.55)
            ax5.set_yticks(y_pos)
            ax5.set_yticklabels(isimler, fontsize=8.5)
            ax5.invert_yaxis()
            ax5.set_xlim(0, 110)
            ax5.set_xlabel("Nihai Doğruluk (%)", fontsize=10)
            ax5.set_title("5. Model Liderlik Tablosu (Leaderboard)", fontsize=12, fontweight="bold", color="#2c5282")

            for bar in barlar:
                w = bar.get_width()
                ax5.text(w + 1.5, bar.get_y() + bar.get_height()/2.0, f"%{w:.2f}", va="center", fontsize=8.5, fontweight="bold")

        # -------------------------------------------------------------
        # PANEL 6: SWOT Karar Matrisi
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        
        swot_metni = (
            "      EXPERIMENT TRACKING & REGISTRY SWOT MATRİSİ\n"
            "───────────────────────────────────────────────────────────────────\n"
            "  [S] GÜÇLÜ YÖNLER (Strengths):\n"
            "  • Sıfır kayıp: Tüm hiperparametreler, metrikler ve modeller kayıtlı.\n"
            "  • Takım içi şeffaflık, liderlik tablosu ve hızlı model seçimi.\n"
            "  • Model sürümleme ve artefakt yönetimiyle üretime hazır altyapı.\n\n"
            "  [W] ZAYIF YÖNLER (Weaknesses):\n"
            "  • Disk ve depolama yönetimi (Her model checkpoint'i yer kaplar).\n"
            "  • Manuel logging disiplini gerektirir (Otomatik wrapper şart).\n\n"
            "  [O] FIRSATLAR (Opportunities):\n"
            "  • Optuna HPO ve Model Registry aşamalarıyla tam otomatik pipeline.\n"
            "  • CI/CD süreçlerinde regression test kapıları (Quality Gates).\n\n"
            "  [T] TEHDİTLER (Threats):\n"
            "  • İzole/untracked yerel script çalıştırma alışkanlığının sürmesi."
        )
        
        ax6.text(
            0.5, 0.5, swot_metni,
            fontsize=8.5,
            family="monospace",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=1", facecolor="#f7fafc", edgecolor="#4a5568", linewidth=1.8)
        )
        ax6.set_title("6. MLOps Deney Takibi SWOT Karar Matrisi", fontsize=12, fontweight="bold", color="#2d3748")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(os.path.dirname(kayit_yolu), exist_ok=True)
        plt.savefig(kayit_yolu, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return kayit_yolu
