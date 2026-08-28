"""CNN Teşhis ve Görselleştirici Modülü.

Öğrenme eğrileri (Loss ve Accuracy), Karmaşıklık Matrisi (Confusion Matrix) ve
Test görsel tahminlerini içeren 4 panelli yayın kalitesinde teşhis çizelgesi üretir.
"""

from pathlib import Path
from typing import List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from .egitici import EgitimSonucu


class CNNGorsellestirici:
    """CNN eğitim çıktısını görselleştiren ve raporlayan sınıf."""

    @classmethod
    def egitim_raporu_ciz(
        cls,
        sonuc: EgitimSonucu,
        sinif_isimleri: List[str],
        X_test: np.ndarray,
        hedef_dosya: Optional[Path] = None,
    ) -> Path:
        """Kapsamlı 4 panelli eğitim teşhis çizelgesi oluşturur.

        Paneller:
        1. Sol Üst: Eğitim ve Doğrulama Kayıp Eğrileri (Loss Curves)
        2. Sağ Üst: Eğitim ve Doğrulama Doğruluk Eğrileri (Accuracy Curves)
        3. Sol Alt: Normalleştirilmiş Karmaşıklık Matrisi (Confusion Matrix Heatmap)
        4. Sağ Alt: Test Kümesinden Örnek Tahminler ve Güven Skorları (Test Preview)

        Args:
            sonuc: EgitimSonucu veri nesnesi.
            sinif_isimleri: Kategori adları listesi.
            X_test: Test görselleri dizisi.
            hedef_dosya: Kaydedilecek PNG dosya yolu.

        Returns:
            Kaydedilen dosyanın Path yolu.
        """
        if hedef_dosya is None:
            hedef_dosya = Path("ciktilar/cnn_egitim_raporu.png")
        hedef_dosya.parent.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(18, 12), dpi=150)
        fig.patch.set_facecolor("#fafafa")

        history = sonuc.tarihce
        epochs_arr = range(1, len(history["loss"]) + 1)

        # ----------------------------------------------------
        # Panel 1: Loss Eğrileri
        # ----------------------------------------------------
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.set_facecolor("#ffffff")
        ax1.plot(epochs_arr, history["loss"], label="Eğitim Kaybı (Train Loss)", color="#d32f2f", linewidth=2.2)
        ax1.plot(epochs_arr, history["val_loss"], label="Doğrulama Kaybı (Val Loss)", color="#1976d2", linewidth=2.2, linestyle="--")

        min_val_loss_idx = np.argmin(history["val_loss"])
        ax1.scatter(
            [min_val_loss_idx + 1],
            [history["val_loss"][min_val_loss_idx]],
            color="#388e3c",
            s=120,
            zorder=5,
            label=f"En İyi Epoch: {min_val_loss_idx + 1} ({history['val_loss'][min_val_loss_idx]:.4f})",
        )

        ax1.set_title("Eğitim ve Doğrulama Kayıp Eğrileri (Cross-Entropy Loss)", fontsize=12, fontweight="bold", pad=10)
        ax1.set_xlabel("Epoch", fontsize=10)
        ax1.set_ylabel("Kayıp (Loss)", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper right", frameon=True, facecolor="#f5f5f5")

        # ----------------------------------------------------
        # Panel 2: Accuracy Eğrileri
        # ----------------------------------------------------
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.set_facecolor("#ffffff")
        ax2.plot(epochs_arr, history["accuracy"], label="Eğitim Doğruluğu (Train Acc)", color="#d32f2f", linewidth=2.2)
        ax2.plot(epochs_arr, history["val_accuracy"], label="Doğrulama Doğruluğu (Val Acc)", color="#1976d2", linewidth=2.2, linestyle="--")

        ax2.set_title("Eğitim ve Doğrulama Doğruluk Eğrileri (Accuracy)", fontsize=12, fontweight="bold", pad=10)
        ax2.set_xlabel("Epoch", fontsize=10)
        ax2.set_ylabel("Doğruluk Oranı (0.0 - 1.0)", fontsize=10)
        ax2.set_ylim([0.0, 1.05])
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="lower right", frameon=True, facecolor="#f5f5f5")

        # ----------------------------------------------------
        # Panel 3: Karmaşıklık Matrisi (Confusion Matrix)
        # ----------------------------------------------------
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.set_facecolor("#ffffff")
        cm = confusion_matrix(sonuc.y_test_gercek, sonuc.y_test_tahmin)
        im = ax3.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

        ticks = np.arange(len(sinif_isimleri))
        ax3.set_xticks(ticks)
        ax3.set_yticks(ticks)
        ax3.set_xticklabels(sinif_isimleri, rotation=30, ha="right", fontsize=10)
        ax3.set_yticklabels(sinif_isimleri, fontsize=10)

        esik = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                renk = "white" if cm[i, j] > esik else "black"
                ax3.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color=renk, fontsize=12, fontweight="bold")

        ax3.set_title(
            f"Test Karmaşıklık Matrisi\n(Test Doğruluğu: %{sonuc.test_dogruluk * 100:.1f} | F1-Macro: {sonuc.f1_macro:.4f})",
            fontsize=12, fontweight="bold", pad=10
        )
        ax3.set_xlabel("Tahmin Edilen Sınıf", fontsize=10)
        ax3.set_ylabel("Gerçek Sınıf", fontsize=10)

        # ----------------------------------------------------
        # Panel 4: Test Örnekleri Tahmin Önizlemesi
        # ----------------------------------------------------
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis("off")
        ax4.set_title("Test Kümesinden Örnek Tahminler ve Güven Skorları", fontsize=12, fontweight="bold", pad=10)

        # En fazla 6 örnek göster
        n_goster = min(6, len(X_test))
        alt_grid = ax4.inset_axes([0, 0, 1, 1])
        alt_grid.axis("off")

        for idx in range(n_goster):
            sub_ax = alt_grid.inset_axes([
                (idx % 3) * 0.33, 0.52 if idx < 3 else 0.02, 0.30, 0.44
            ])
            sub_ax.imshow(X_test[idx])
            sub_ax.axis("off")

            gercek = sinif_isimleri[sonuc.y_test_gercek[idx]]
            tahmin = sinif_isimleri[sonuc.y_test_tahmin[idx]]
            guven = sonuc.y_test_olasiliklar[idx][sonuc.y_test_tahmin[idx]] * 100.0

            renk_yazi = "#2e7d32" if gercek == tahmin else "#c62828"
            sub_ax.set_title(
                f"Tahmin: {tahmin} (%{guven:.1f})\nGerçek: {gercek}",
                fontsize=8, fontweight="bold", color=renk_yazi, pad=2
            )

        plt.suptitle(
            "DAY 20: TENSORFLOW/KERAS İLE DERİN ÖĞRENME GÖRSEL SINIFLANDIRMA (CNN)\n"
            f"Mimari: 3x [Conv2D + BatchNorm + ReLU + MaxPool] + Flatten + Dense + Dropout | {sonuc.ozet()}",
            fontsize=13, fontweight="bold", color="#212121", y=0.98
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(hedef_dosya, bbox_inches="tight", dpi=150)
        plt.close(fig)

        return hedef_dosya
