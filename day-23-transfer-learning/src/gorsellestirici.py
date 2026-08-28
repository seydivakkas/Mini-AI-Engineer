"""Transfer Öğrenme Görselleştirme ve Teşhis Modülü.

Bu modül; Scratch vs Feature Extraction vs Fine-Tuning stratejilerinin
öğrenme eğrilerini, eğitilebilir parametre tasarrufunu ve test performansı
karşılaştırmasını içeren 4 panelli teşhis raporunu çizer.
"""

from pathlib import Path
from typing import List, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.egitici import TransferEgitimSonucu


class TransferGorsellestirici:
    """Transfer öğrenme görselleştirici sınıfı."""

    @staticmethod
    def karsilastirma_raporu_ciz(
        sonuclar: List[TransferEgitimSonucu],
        sinif_isimleri: List[str],
        hedef_dosya: Union[str, Path] = "ciktilar/transfer_ogrenme_raporu.png",
    ) -> Path:
        """4 panelli kapsamlı transfer öğrenme analiz raporu üretir."""
        hedef_path = Path(hedef_dosya)
        hedef_path.parent.mkdir(parents=True, exist_ok=True)

        fig, eksenler = plt.subplots(2, 2, figsize=(15, 12), dpi=140)
        fig.suptitle(
            "Transfer Öğrenme ve İnce Ayar (Transfer Learning & Fine-Tuning) Analizi",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )

        renkler = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]

        # ----------------------------------------------------
        # PANEL 1: Doğrulama Doğruluğu (Val Accuracy) Eğrileri
        # ----------------------------------------------------
        ax1 = eksenler[0, 0]
        for idx, res in enumerate(sonuclar):
            ep_list = np.arange(1, len(res.val_dogruluklari) + 1)
            ax1.plot(
                ep_list,
                [acc * 100 for acc in res.val_dogruluklari],
                label=res.model_adi,
                color=renkler[idx % len(renkler)],
                linewidth=2,
                marker="o",
                markersize=4,
            )
        ax1.set_title("Öğrenme Hızı ve Yakınsama (Validation Acc %)", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Doğruluk (%)")
        ax1.set_ylim(0, 105)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="lower right", frameon=True)

        # ----------------------------------------------------
        # PANEL 2: Eğitilebilir Parametre Tasarrufu vs Doğruluk
        # ----------------------------------------------------
        ax2 = eksenler[0, 1]
        modeller = [s.model_adi.replace(" (", "\n(") for s in sonuclar]
        egitilebilir_p = [s.egitilebilir_parametre for s in sonuclar]
        toplam_p = [s.toplam_parametre for s in sonuclar]

        x = np.arange(len(modeller))
        width = 0.35

        ax2.bar(x - width/2, [tp / 1e6 for tp in toplam_p], width, label="Toplam Parametre (Milyon)", color="#aec7e8")
        ax2.bar(x + width/2, [ep / 1e6 for ep in egitilebilir_p], width, label="Eğitilebilir Parametre (Milyon)", color="#ff7f0e")

        ax2.set_title("Parametre Verimliliği (Eğitilen vs Toplam Parametre)", fontsize=11, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(modeller, fontsize=8)
        ax2.set_ylabel("Parametre Sayısı ($10^6$ / Milyon)")
        ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
        ax2.legend(loc="upper right", frameon=True)

        # ----------------------------------------------------
        # PANEL 3: Test Kümesi Doğruluğu & F1-Macro
        # ----------------------------------------------------
        ax3 = eksenler[1, 0]
        test_accs = [s.test_dogruluk * 100 for s in sonuclar]
        f1_scores = [s.f1_macro for s in sonuclar]

        rects = ax3.bar(modeller, test_accs, color=renkler, width=0.5)
        ax3.set_title("Test Doğruluğu Karşılaştırması (%)", fontsize=11, fontweight="bold")
        ax3.set_ylabel("Test Doğruluğu (%)")
        ax3.set_ylim(0, 115)
        ax3.grid(True, linestyle="--", alpha=0.5, axis="y")

        for rect, f1 in zip(rects, f1_scores):
            height = rect.get_height()
            ax3.annotate(
                f"%{height:.1f}\n(F1: {f1:.2f})",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

        # ----------------------------------------------------
        # PANEL 4: En Başarılı Modelin Karışıklık Matrisi
        # ----------------------------------------------------
        ax4 = eksenler[1, 1]
        en_iyi_sonuc = max(sonuclar, key=lambda s: s.test_dogruluk)
        cm = en_iyi_sonuc.karisiklik_matrisi

        im = ax4.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

        tick_marks = np.arange(len(sinif_isimleri))
        ax4.set_xticks(tick_marks)
        ax4.set_xticklabels(sinif_isimleri, rotation=30, ha="right", fontsize=9)
        ax4.set_yticks(tick_marks)
        ax4.set_yticklabels(sinif_isimleri, fontsize=9)

        thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax4.text(
                    j, i, f"{cm[i, j]:d}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold",
                )

        ax4.set_title(
            f"En İyi Model Karışıklık Matrisi: {en_iyi_sonuc.model_adi}\n(Test Acc: %{en_iyi_sonuc.test_dogruluk * 100:.1f})",
            fontsize=11,
            fontweight="bold",
        )
        ax4.set_ylabel("Gerçek Sınıf")
        ax4.set_xlabel("Tahmin Edilen Sınıf")

        plt.tight_layout()
        fig.savefig(hedef_path, bbox_inches="tight")
        plt.close(fig)
        return hedef_path
