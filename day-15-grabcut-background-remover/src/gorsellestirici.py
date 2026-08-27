"""GrabCut Segmentasyon ve Kompozit Görselleştirici (Headless Matplotlib).

Bu modül; orijinal görüntüyü sınırlayıcı kutusuyla, GrabCut 4-durumlu olasılık
maskesini, izole edilmiş ön planı ve yeni stüdyo arka planına giydirilmiş kompozit
çıktıyı 4 panelli karşılaştırma çizelgesi olarak kaydeder.
"""

from pathlib import Path
from typing import Tuple, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2


class GrabCutGorsellestirici:
    """GrabCut segmentasyon çıktılarını görselleştiren araç."""

    @classmethod
    def analiz_paneli_ciz(
        cls,
        orijinal_bgr: np.ndarray,
        dikdortgen: Tuple[int, int, int, int],
        ham_maske: np.ndarray,
        izole_on_plan_bgr: np.ndarray,
        kompozit_bgr: np.ndarray,
        dosya_yolu: Path,
        firca_izleri_bgr: Optional[np.ndarray] = None
    ) -> Path:
        """4 panelli (2x2) GrabCut analiz raporunu diske kaydeder."""
        fig, eksenler = plt.subplots(2, 2, figsize=(13, 11), dpi=150)

        # 1. Panel: Orijinal Görüntü ve Başlangıç Kutusu
        img_kutu = orijinal_bgr.copy()
        x, y, w, h = dikdortgen
        cv2.rectangle(img_kutu, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            img_kutu, f"Kutu ({x},{y},{w},{h})", (x + 5, y - 8 if y > 20 else y + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA
        )

        if firca_izleri_bgr is not None:
            # Fırça izlerini bindir
            mask_iz = (firca_izleri_bgr > 0).any(axis=2)
            img_kutu[mask_iz] = firca_izleri_bgr[mask_iz]

        eksenler[0, 0].imshow(img_kutu[:, :, ::-1])
        eksenler[0, 0].set_title("1. Orijinal Görüntü & Başlangıç Kutusu\n(Yeşil: GC_INIT_WITH_RECT)", fontsize=11, fontweight="bold")
        eksenler[0, 0].axis("off")

        # 2. Panel: GrabCut 4-Durumlu Ham Maske
        # 0: BGD (Siyah), 1: FGD (Beyaz), 2: PR_BGD (Koyu Gri), 3: PR_FGD (Açık Gri)
        renk_haritasi = {
            0: [20, 20, 20],       # Kesin Arka Plan
            1: [255, 255, 255],   # Kesin Ön Plan
            2: [70, 70, 90],      # Olası Arka Plan
            3: [180, 190, 210]    # Olası Ön Plan
        }
        renkli_maske = np.zeros((*ham_maske.shape, 3), dtype=np.uint8)
        for val, renk in renk_haritasi.items():
            renkli_maske[ham_maske == val] = renk

        eksenler[0, 1].imshow(renkli_maske)
        eksenler[0, 1].set_title("2. GrabCut 4-Durumlu Enerji Maskesi\n(BGD, PR_BGD, PR_FGD, FGD)", fontsize=11, fontweight="bold")
        eksenler[0, 1].axis("off")

        # 3. Panel: İzole Edilmiş Ön Plan (Siyah Zemin)
        eksenler[1, 0].imshow(izole_on_plan_bgr[:, :, ::-1])
        eksenler[1, 0].set_title("3. İzole Edilmiş Ön Plan\n(Segmentasyon Çıktısı)", fontsize=11, fontweight="bold")
        eksenler[1, 0].axis("off")

        # 4. Panel: Yeni Arka Plan Kompoziti
        eksenler[1, 1].imshow(kompozit_bgr[:, :, ::-1])
        eksenler[1, 1].set_title("4. Yeni Stüdyo Arka Planı Giydirilmiş Kompozit\n(Feathered Alpha Blending)", fontsize=11, fontweight="bold")
        eksenler[1, 1].axis("off")

        fig.tight_layout()
        dosya_yolu.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dosya_yolu)
        plt.close(fig)
        return dosya_yolu
