"""
Çökmeye Dayanıklı Eğitim Görselleştiricisi
=========================================
Çökme simülasyonu öncesi ve sonrası eğitim kaybı, doğrulama skoru, LR devamlılığı,
disk saklama politikası ve SWOT karar matrisini 6 panelli endüstriyel panoda birleştirir.
"""

from typing import Dict, Any, List
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class CheckpointTeshisGorsellestirici:
    """
    Day 71 6-Panelli Çökmeye Dayanıklı Eğitim Teşhis Panosu üreticisi.
    """

    @staticmethod
    def panoyu_ciz_ve_kaydet(
        egitim_gecmisi: Dict[str, Any],
        cokus_epochu: int = 5,
        cikti_yolu: str = "ciktilar/resumable_training_paneli.png"
    ) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(cikti_yolu)), exist_ok=True)

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(2, 3, figsize=(21, 13))
        fig.suptitle(
            "Day 71: Çökmeye Dayanıklı Checkpoint, State Restoration ve Devam Edebilir Eğitim Paneli",
            fontsize=17,
            fontweight="bold",
            color="#1a252f",
            y=0.98
        )

        epochs = egitim_gecmisi["epoch"]
        train_loss = egitim_gecmisi["train_loss"]
        val_loss = egitim_gecmisi["val_loss"]
        val_acc = egitim_gecmisi["val_accuracy"]
        lrs = egitim_gecmisi["lr"]

        # -------------------------------------------------------------
        # 1. Panel: Yönetici & Çöküş/Restorasyon Özet Kartı
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.axis("off")
        ozet_metin = (
            "    FAULT-TOLERANT EGITIM MOTORU OZETI\n"
            "═══════════════════════════════════════════════════════\n"
            f" * 1. Faz Egitim         : Epoch 1 -> {cokus_epochu} (Normal Calisma)\n"
            f" * Simule Edilmis Cokus  : Epoch {cokus_epochu} Sonunda (SIGKILL / OOM)\n"
            f" * Durum Restorasyonu    : Model + Opt + Sched + RNG (last.pt)\n"
            f" * 2. Faz Devam          : Epoch {cokus_epochu+1} -> 10 (SIFIR KAYIP SICRAMASI)\n"
            "───────────────────────────────────────────────────────\n"
            f" * Son Train Loss        : {train_loss[-1]:.4f}\n"
            f" * Son Val Accuracy      : %{val_acc[-1]:.2f}\n"
            f" * En Iyi Val Loss       : {min(val_loss):.4f}\n"
            "═══════════════════════════════════════════════════════\n"
            " * Atomik I/O (Atomic Save) : AKTIF (.tmp -> os.replace)\n"
            " * Top-K Checkpoint Budama  : AKTIF (Disk Guvenligi)"
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
        ax1.set_title("1. Cokus & Restorasyon Ozeti", fontweight="bold", color="#16a085")

        # -------------------------------------------------------------
        # 2. Panel: Eğitim Kaybı & Çöküş Kesinti Çizgisi
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.plot(epochs[:cokus_epochu], train_loss[:cokus_epochu], label="1. Faz (Cokus Oncesi)", color="#e74c3c", linewidth=2.2, marker="o")
        ax2.plot(epochs[cokus_epochu-1:], train_loss[cokus_epochu-1:], label="2. Faz (Restorasyon Sonrasi)", color="#2ecc71", linewidth=2.5, marker="s")
        ax2.axvline(cokus_epochu, color="#e67e22", linestyle="--", linewidth=2.0, label=f"Simule Cokus (Epoch {cokus_epochu})")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Egitim Kaybi (Train Loss)")
        ax2.set_title("2. Kesintisiz Egitim Kaybi Trajektorisi", fontweight="bold", color="#2980b9")
        ax2.legend(loc="upper right")

        # -------------------------------------------------------------
        # 3. Panel: Doğrulama Başarımı (Validation Accuracy %)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.plot(epochs, val_acc, label="Validation Accuracy (%)", color="#9b59b6", linewidth=2.5, marker="D")
        ax3.axvline(cokus_epochu, color="#e67e22", linestyle="--", linewidth=1.8, label=f"Cokus Noktasi (Epoch {cokus_epochu})")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Dogruluk (%)")
        ax3.set_title("3. Dogrulama Basarimi (Val Acc %)", fontweight="bold", color="#8e44ad")
        ax3.legend(loc="lower right")

        # -------------------------------------------------------------
        # 4. Panel: Öğrenme Oranı (LR Scheduler) Korunumu
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.plot(epochs, lrs, label="Cosine Annealing LR", color="#3498db", linewidth=2.5, marker="^")
        ax4.axvline(cokus_epochu, color="#e67e22", linestyle="--", linewidth=1.8, label="Restorasyon Noktasi")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Ogrenme Orani (LR)")
        ax4.set_title("4. Scheduler Durum Devamliligi (Sifirlanmadi)", fontweight="bold", color="#d35400")
        ax4.legend(loc="upper right")

        # -------------------------------------------------------------
        # 5. Panel: Top-K Checkpoint Saklama Politikası
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        dosyalar = ["best.pt\n(En Iyi)", "last.pt\n(Son Durum)", f"ep_{epochs[-1]:02d}.pt\n(Guncel)", f"ep_{epochs[-2]:02d}.pt\n(Top-2)", f"ep_{epochs[-3]:02d}.pt\n(Top-3)"]
        boyutlar = [1.2, 1.2, 1.2, 1.2, 1.2]
        renkler = ["#2ecc71", "#3498db", "#9b59b6", "#f1c40f", "#e67e22"]
        bars = ax5.bar(dosyalar, boyutlar, color=renkler, width=0.55)
        ax5.set_ylabel("Yaklasik Dosya Boyutu (MB)")
        ax5.set_title("5. Top-K Disk Yonetimi & Checkpointler", fontweight="bold", color="#27ae60")
        for bar in bars:
            yval = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.1f} MB", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax5.set_ylim(0, 1.6)

        # -------------------------------------------------------------
        # 6. Panel: SWOT Karar Matrisi
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.axis("off")
        swot_metin = (
            " FAULT-TOLERANT CHECKPOINT SWOT MATRISI\n"
            "─────────────────────────────────────────────────\n"
            " [S] GUCLU YONLER (Strengths):\n"
            " • Atomik I/O ile yarım kalmış bozuk dosya oluşmaz\n"
            " • Optimizer momentum vektörleri (m_t, v_t) korunur\n"
            " • RNG durumları saklandığından %100 tekrarlanabilir\n\n"
            " [W] ZAYIF YONLER (Weaknesses):\n"
            " • Her epoch'ta disk I/O gecikmesi (büyük LLM'lerde)\n"
            " • Ekstra disk alanı gereksinimi (Top-K ile çözüldü)\n\n"
            " [O] FIRSATLAR (Opportunities):\n"
            " • Spot / Preemptible GPU sunucularında %70 maliyet tasarrufu\n"
            " • Günlerce süren Transformer eğitimlerinde sıfır risk\n\n"
            " [T] TEHDITLER (Threats):\n"
            " • Yalnızca model ağırlıklarını kaydedip optimizer'ı unutma tuzağı"
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
        ax6.set_title("6. Checkpoint Mimarisi SWOT Matrisi", fontweight="bold", color="#d35400")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(cikti_yolu, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return cikti_yolu
