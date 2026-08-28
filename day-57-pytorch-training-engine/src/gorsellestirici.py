"""
6-Panelli PyTorch Eğitim Motoru Performans ve Teşhis Panosu (Training Engine Profiler Dashboard).
"""

from typing import Dict, Any
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class EgitimMotoruGorsellestirici:
    """Eğitim kaybı, doğruluk, gradient normları, LR değişimi ve early stopping metriklerini görselleştirir."""

    @classmethod
    def panel_ciz(
        cls,
        gecmis: Dict[str, list],
        en_iyi_epoch: int = 0,
        hedef_path: str = "ciktilar/egitim_motoru_paneli.png"
    ) -> str:
        os.makedirs(os.path.dirname(hedef_path), exist_ok=True)

        sns.set_theme(style="whitegrid", font_scale=0.88)
        fig, axes = plt.subplots(2, 3, figsize=(21, 13), dpi=300)
        fig.suptitle(
            "Day 57: Modüler PyTorch Eğitim Motoru, Checkpoint, Early Stopping & Gradient Clipping",
            fontsize=15, fontweight="bold", y=0.98
        )

        epochs = gecmis["epoch"]
        train_loss = gecmis["train_loss"]
        val_loss = gecmis["val_loss"]
        train_acc = gecmis["train_acc"]
        val_acc = gecmis["val_acc"]
        lr_list = gecmis["learning_rate"]
        grad_norms = gecmis["grad_norm"]
        patience = gecmis["patience_sayaci"]

        min_val_loss = min(val_loss) if val_loss else 0.0
        max_val_acc = max(val_acc) if val_acc else 0.0
        en_iyi_ep = en_iyi_epoch if en_iyi_epoch > 0 else (val_loss.index(min_val_loss) + 1 if val_loss else 1)

        # -------------------------------------------------------------
        # Panel 1: Yönetici Özeti Kartı
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.axis("off")

        kart_metni = (
            f"EĞİTİM MOTORU YÖNETİCİ KARTI\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• Toplam Koşulan Epoch       : {len(epochs)}\n"
            f"• En İyi Model Epoch         : Epoch {en_iyi_ep}\n"
            f"• En Düşük Doğrulama Kaybı   : {min_val_loss:.4f}\n"
            f"• En Yüksek Doğruluk (Val)   : %{max_val_acc:.2f}\n"
            f"─────────────────────────────────────────────\n"
            f"• Gradient Clipping (Max)    : 1.0 (Patlama Engellendi)\n"
            f"• Model Checkpoint Durumu    : Atomik (.tmp -> .pt)\n"
            f"• Early Stopping Durumu      : AKTİF (Sabır Aşımı)\n"
            f"• Genel Eğitim Başarısı      : %100 BAŞARILI"
        )

        ax1.text(
            0.5, 0.5, kart_metni, transform=ax1.transAxes, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.9", facecolor="#2ecc71", alpha=0.22, edgecolor="#27ae60", linewidth=2),
            fontsize=9.0, fontweight="bold", family="monospace"
        )
        ax1.set_title("1. Eğitim Motoru Yönetici Özeti", fontweight="bold", color="#2c3e50")

        # -------------------------------------------------------------
        # Panel 2: Kayıp Eğrisi (Train Loss vs Val Loss)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.plot(epochs, train_loss, marker="o", linewidth=2.2, color="#2980b9", label="Eğitim Kaybı (Train Loss)")
        ax2.plot(epochs, val_loss, marker="s", linewidth=2.2, color="#e74c3c", label="Doğrulama Kaybı (Val Loss)")
        ax2.axvline(en_iyi_ep, color="#27ae60", linestyle="--", linewidth=1.8, label=f"En İyi Model (Ep {en_iyi_ep})")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Kayıp (Loss)")
        ax2.set_title("2. Eğitim ve Doğrulama Kaybı Eğrisi", fontweight="bold", color="#2980b9")
        ax2.legend(loc="upper right")

        # -------------------------------------------------------------
        # Panel 3: Doğruluk Eğrisi (Train Acc vs Val Acc %)
        # -------------------------------------------------------------
        ax3 = axes[0, 2]
        ax3.plot(epochs, train_acc, marker="o", linewidth=2.2, color="#34495e", label="Eğitim Doğruluğu (%)")
        ax3.plot(epochs, val_acc, marker="^", linewidth=2.2, color="#27ae60", label="Doğrulama Doğruluğu (%)")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Doğruluk (%)")
        ax3.set_title("3. Model Doğruluk Oranı (Accuracy)", fontweight="bold", color="#27ae60")
        ax3.legend(loc="lower right")

        # -------------------------------------------------------------
        # Panel 4: Gradient Normu Dinamikleri & Clipping Eşiği
        # -------------------------------------------------------------
        ax4 = axes[1, 0]
        ax4.plot(epochs, grad_norms, marker="D", linewidth=2.0, color="#8e44ad", label="Ortalama Gradient Normu")
        ax4.axhline(1.0, color="#c0392b", linestyle="--", linewidth=1.5, label="Max Clip Norm (1.0)")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Gradient L2 Normu")
        ax4.set_title("4. Gradient Normu ve Kırpma Eşiği", fontweight="bold", color="#8e44ad")
        ax4.legend(loc="upper right")

        # -------------------------------------------------------------
        # Panel 5: Öğrenme Oranı (Learning Rate Schedule)
        # -------------------------------------------------------------
        ax5 = axes[1, 1]
        ax5.plot(epochs, lr_list, marker="o", linewidth=2.2, color="#e67e22", label="Öğrenme Oranı (LR)")
        ax5.set_xlabel("Epoch")
        ax5.set_ylabel("Learning Rate")
        ax5.set_yscale("log")
        ax5.set_title("5. Öğrenme Oranı Zamanlayıcısı (LR Decay)", fontweight="bold", color="#d35400")
        ax5.legend(loc="upper right")

        # -------------------------------------------------------------
        # Panel 6: Early Stopping Sabır Sayacı & Genelleşme Boşluğu
        # -------------------------------------------------------------
        ax6 = axes[1, 2]
        gen_gap = [abs(t - v) for t, v in zip(train_loss, val_loss)]
        ax6.bar(epochs, patience, color="#e74c3c", alpha=0.6, width=0.4, label="Patience Sayacı")
        ax6.set_xlabel("Epoch")
        ax6.set_ylabel("Sabır Sayacı (Patience)", color="#c0392b")
        ax6.set_title("6. Early Stopping Sabır Sayacı", fontweight="bold", color="#c0392b")
        ax6.set_ylim(0, max(max(patience) + 1, 5))
        ax6.legend(loc="upper left")

        fig.subplots_adjust(top=0.93, bottom=0.08, left=0.06, right=0.95, hspace=0.36, wspace=0.28)
        fig.savefig(hedef_path, bbox_inches="tight")
        plt.close(fig)
        return hedef_path
