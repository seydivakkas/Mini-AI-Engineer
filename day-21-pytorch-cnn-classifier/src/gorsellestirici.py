"""PyTorch Görselleştirme ve Teşhis Modülü.

Bu modül; PyTorch CNN modelinin eğitim eğrilerini (Loss & Accuracy),
öğrenme oranı dinamiklerini, karışıklık matrisini (Confusion Matrix) ve
test görsel tahminlerini içeren 4 panelli yüksek çözünürlüklü teşhis çizelgesi üretir.
"""

from pathlib import Path
from typing import List, Optional, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from src.egitici import EgitimSonucu


class PyTorchGorsellestirici:
    """PyTorch eğitim çıktılarını görselleştirici sınıf."""

    @staticmethod
    def egitim_raporu_ciz(
        sonuc: EgitimSonucu,
        sinif_isimleri: List[str],
        X_test: np.ndarray,
        hedef_dosya: Union[str, Path] = "ciktilar/pytorch_cnn_raporu.png",
    ) -> Path:
        """4 panelli kapsamlı eğitim ve teşhis raporunu çizer ve kaydeder."""
        hedef_path = Path(hedef_dosya)
        hedef_path.parent.mkdir(parents=True, exist_ok=True)

        fig, eksenler = plt.subplots(2, 2, figsize=(15, 12), dpi=140)
        fig.suptitle(
            "PyTorch CNN Görsel Sınıflandırıcı: Eğitim Teşhisi ve Test Performansı",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )

        epoch_sayisi = len(sonuc.train_kayiplari)
        epochlar = np.arange(1, epoch_sayisi + 1)

        # ----------------------------------------------------
        # PANEL 1: Kayıp (Loss) Eğrileri
        # ----------------------------------------------------
        ax1 = eksenler[0, 0]
        ax1.plot(epochlar, sonuc.train_kayiplari, label="Train Loss", color="#1f77b4", linewidth=2, marker="o", markersize=4)
        ax1.plot(epochlar, sonuc.val_kayiplari, label="Val Loss", color="#ff7f0e", linewidth=2, marker="s", markersize=4)

        en_iyi_epoch = int(np.argmin(sonuc.val_kayiplari)) + 1
        en_iyi_val_loss = np.min(sonuc.val_kayiplari)
        ax1.axvline(x=en_iyi_epoch, color="#2ca02c", linestyle="--", alpha=0.8, label=f"En İyi Epoch ({en_iyi_epoch})")
        ax1.scatter([en_iyi_epoch], [en_iyi_val_loss], color="#2ca02c", s=80, zorder=5)

        ax1.set_title("Eğitim ve Doğrulama Kayıp Eğrileri (Loss)", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("CrossEntropy Kaybı")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper right", frameon=True)

        # ----------------------------------------------------
        # PANEL 2: Doğruluk (Accuracy) ve LR Eğrileri
        # ----------------------------------------------------
        ax2 = eksenler[0, 1]
        line1 = ax2.plot(epochlar, [a * 100 for a in sonuc.train_dogruluklari], label="Train Acc (%)", color="#1f77b4", linewidth=2, marker="o", markersize=4)
        line2 = ax2.plot(epochlar, [a * 100 for a in sonuc.val_dogruluklari], label="Val Acc (%)", color="#2ca02c", linewidth=2, marker="^", markersize=4)
        ax2.set_ylabel("Doğruluk (%)")
        ax2.set_xlabel("Epoch")
        ax2.set_ylim(0, 105)
        ax2.grid(True, linestyle="--", alpha=0.5)

        # İkincil Eksen: Öğrenme Oranı (LR)
        ax2_lr = ax2.twinx()
        line3 = ax2_lr.plot(epochlar, sonuc.lr_tarihcesi, label="Learning Rate", color="#d62728", linestyle=":", linewidth=1.5)
        ax2_lr.set_ylabel("Öğrenme Oranı (LR)", color="#d62728")
        ax2_lr.tick_params(axis='y', labelcolor="#d62728")

        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc="lower right", frameon=True)
        ax2.set_title("Eğitim / Doğrulama Doğruluğu & LR Zamanlaması", fontsize=11, fontweight="bold")

        # ----------------------------------------------------
        # PANEL 3: Karışıklık Matrisi (Confusion Matrix)
        # ----------------------------------------------------
        ax3 = eksenler[1, 0]
        cm = sonuc.karisiklik_matrisi
        im = ax3.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

        tick_marks = np.arange(len(sinif_isimleri))
        ax3.set_xticks(tick_marks)
        ax3.set_xticklabels(sinif_isimleri, rotation=30, ha="right", fontsize=9)
        ax3.set_yticks(tick_marks)
        ax3.set_yticklabels(sinif_isimleri, fontsize=9)

        # Hücre içi sayıları yazdır
        thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax3.text(
                    j, i, f"{cm[i, j]:d}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold",
                    fontsize=10,
                )

        ax3.set_title(
            f"Test Karışıklık Matrisi (Test Acc: %{sonuc.test_dogruluk * 100:.1f} | F1: {sonuc.f1_macro:.3f})",
            fontsize=11,
            fontweight="bold",
        )
        ax3.set_ylabel("Gerçek Sınıf")
        ax3.set_xlabel("Tahmin Edilen Sınıf")

        # ----------------------------------------------------
        # PANEL 4: Test Görsel Tahminleri ve Güven Skorları
        # ----------------------------------------------------
        ax4 = eksenler[1, 1]
        ax4.axis("off")

        # Test kümesinden 6 örnek görsel seç
        ornek_sayisi = min(6, len(X_test))
        alt_grid = ax4.inset_axes([0, 0, 1, 1])
        alt_grid.axis("off")

        satir, sutun = 2, 3
        for k in range(ornek_sayisi):
            sub_ax = alt_grid.inset_axes([(k % sutun) / sutun, 1.0 - ((k // sutun) + 1) / satir, 1.0 / sutun - 0.04, 1.0 / satir - 0.08])
            img = X_test[k]
            gercek_sinif = sinif_isimleri[sonuc.y_test_gercek[k]]
            tahmin_sinif = sinif_isimleri[sonuc.y_test_tahmin[k]]
            guven = sonuc.y_test_olasiliklar[k][sonuc.y_test_tahmin[k]] * 100.0

            sub_ax.imshow(img)
            sub_ax.axis("off")

            renk = "green" if gercek_sinif == tahmin_sinif else "red"
            sub_ax.set_title(
                f"T:{tahmin_sinif} (%{guven:.0f})\nG:{gercek_sinif}",
                fontsize=8,
                color=renk,
                fontweight="bold",
            )

        ax4.set_title("Test Kümesi Tahmin Örnekleri (T: Tahmin, G: Gerçek)", fontsize=11, fontweight="bold", y=1.02)

        plt.tight_layout()
        fig.savefig(hedef_path, bbox_inches="tight")
        plt.close(fig)
        return hedef_path
