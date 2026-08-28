"""Veri Çoğaltma Görselleştirme ve Veri Hikayeciliği Modülü.

Bu modül; veri çoğaltma dönüşümlerinin (Albumentations, MixUp, CutMix) görsel galerisini
ve stratejiler arası karşılaştırmalı performans hikayesi grafiklerini üretir.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.albumentations_donusturucu import AlbumentationsDonusturucu
from src.mixup_cutmix import MixUpCutMixUygulayici
from src.karsilastirici import StratejiSonucu


class VeriCogaltmaGorsellestirici:
    """Veri çoğaltma görselleştirici sınıfı."""

    @staticmethod
    def galeri_ciz(
        X_ornekler: np.ndarray,
        y_ornekler: np.ndarray,
        sinif_isimleri: List[str],
        hedef_dosya: Union[str, Path] = "ciktilar/veri_cogaltma_galerisi.png",
    ) -> Path:
        """Her sınıf için Orijinal, Albumentations, MixUp ve CutMix örneklerini içeren galeri çizer."""
        hedef_path = Path(hedef_dosya)
        hedef_path.parent.mkdir(parents=True, exist_ok=True)

        albu = AlbumentationsDonusturucu((64, 64))
        n_sinif = len(sinif_isimleri)

        fig, eksenler = plt.subplots(n_sinif, 4, figsize=(14, 3.2 * n_sinif), dpi=140)
        fig.suptitle(
            "Veri Çoğaltma (Data Augmentation) Stratejileri Karşılaştırma Galerisi",
            fontsize=15,
            fontweight="bold",
            y=0.99,
        )

        sutun_basliklari = [
            "Orijinal Görsel",
            "Albumentations (Geom + Renk + Dropout)",
            "MixUp (Çift Etiket İnterpolasyonu)",
            "CutMix (Bölgesel Kes-Yapıştır)",
        ]

        # Tensör hazırlığı
        X_tensor = torch.from_numpy(np.transpose(X_ornekler, (0, 3, 1, 2))).float()
        y_tensor = torch.from_numpy(y_ornekler).long()

        mix_x, ya_m, yb_m, lam_m = MixUpCutMixUygulayici.uygula_mixup(X_tensor, y_tensor, alpha=0.8)
        cut_x, ya_c, yb_c, lam_c = MixUpCutMixUygulayici.uygula_cutmix(X_tensor, y_tensor, alpha=1.0)

        for i in range(n_sinif):
            # 1. Orijinal
            ax0 = eksenler[i, 0]
            img_orig = X_ornekler[i]
            ax0.imshow(img_orig)
            ax0.set_ylabel(sinif_isimleri[y_ornekler[i]], fontsize=11, fontweight="bold")
            if i == 0:
                ax0.set_title(sutun_basliklari[0], fontsize=10, fontweight="bold")
            ax0.set_xticks([])
            ax0.set_yticks([])

            # 2. Albumentations
            ax1 = eksenler[i, 1]
            img_albu = albu.donustur_tekil(img_orig, mod="agir")
            ax1.imshow(img_albu)
            if i == 0:
                ax1.set_title(sutun_basliklari[1], fontsize=10, fontweight="bold")
            ax1.set_xticks([])
            ax1.set_yticks([])

            # 3. MixUp
            ax2 = eksenler[i, 2]
            img_mix = np.transpose(mix_x[i].cpu().numpy(), (1, 2, 0))
            ax2.imshow(np.clip(img_mix, 0.0, 1.0))
            etiket_a = sinif_isimleri[ya_m[i].item()]
            etiket_b = sinif_isimleri[yb_m[i].item()]
            ax2.set_xlabel(f"{lam_m:.2f}*{etiket_a} + {1.0-lam_m:.2f}*{etiket_b}", fontsize=8)
            if i == 0:
                ax2.set_title(sutun_basliklari[2], fontsize=10, fontweight="bold")
            ax2.set_xticks([])
            ax2.set_yticks([])

            # 4. CutMix
            ax3 = eksenler[i, 3]
            img_cut = np.transpose(cut_x[i].cpu().numpy(), (1, 2, 0))
            ax3.imshow(np.clip(img_cut, 0.0, 1.0))
            etiket_ca = sinif_isimleri[ya_c[i].item()]
            etiket_cb = sinif_isimleri[yb_c[i].item()]
            ax3.set_xlabel(f"{lam_c:.2f}*{etiket_ca} + {1.0-lam_c:.2f}*{etiket_cb}", fontsize=8)
            if i == 0:
                ax3.set_title(sutun_basliklari[3], fontsize=10, fontweight="bold")
            ax3.set_xticks([])
            ax3.set_yticks([])

        plt.tight_layout()
        fig.savefig(hedef_path, bbox_inches="tight")
        plt.close(fig)
        return hedef_path

    @staticmethod
    def karsilastirma_raporu_ciz(
        sonuclar: List[StratejiSonucu],
        hedef_dosya: Union[str, Path] = "ciktilar/veri_cogaltma_karsilastirma_raporu.png",
    ) -> Path:
        """Veri hikayesi niteliğinde stratejiler arası karşılaştırmalı performans grafiğini çizer."""
        hedef_path = Path(hedef_dosya)
        hedef_path.parent.mkdir(parents=True, exist_ok=True)

        stratejiler = [s.strateji_adi for s in sonuclar]
        temiz_acc = [s.test_acc * 100 for s in sonuclar]
        gurultu_acc = [s.gurultulu_test_acc * 100 for s in sonuclar]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=140)
        fig.suptitle(
            "Veri Hikayesi: Veri Çoğaltmanın Model Dayanıklılığı ve Genellemesine Etkisi",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )

        x = np.arange(len(stratejiler))
        width = 0.35

        rects1 = ax1.bar(x - width/2, temiz_acc, width, label="Temiz Test Doğruluğu (%)", color="#1f77b4")
        rects2 = ax1.bar(x + width/2, gurultu_acc, width, label="Gürültülü/Bozulmuş Test Doğruluğu (%)", color="#ff7f0e")

        ax1.set_ylabel("Doğruluk (%)", fontsize=11)
        ax1.set_title("Temiz vs Gürültülü Test Performansı (Robustness)", fontsize=12, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(stratejiler, fontsize=10, fontweight="bold")
        ax1.set_ylim(0, 115)
        ax1.grid(True, linestyle="--", alpha=0.4, axis="y")
        ax1.legend(loc="lower right", frameon=True)

        for rect in rects1 + rects2:
            height = rect.get_height()
            ax1.annotate(f"%{height:.1f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center", va="bottom", fontsize=8, fontweight="bold"
            )

        # Panel 2: Dayanıklılık Kaybı (Degradation / Drop) Analizi
        farklar = [t - g for t, g in zip(temiz_acc, gurultu_acc)]
        renkler = ["#d62728" if f > 20 else "#2ca02c" for f in farklar]

        bars = ax2.bar(stratejiler, farklar, color=renkler, width=0.5)
        ax2.set_ylabel("Gürültü Altında Doğruluk Kaybı (% Puan)", fontsize=11)
        ax2.set_title("Model Kırılganlığı (Düşük Kayıp = Yüksek Dayanıklılık)", fontsize=12, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.4, axis="y")

        for bar in bars:
            h = bar.get_height()
            ax2.annotate(f"-{h:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=9, fontweight="bold"
            )

        plt.tight_layout()
        fig.savefig(hedef_path, bbox_inches="tight")
        plt.close(fig)
        return hedef_path
