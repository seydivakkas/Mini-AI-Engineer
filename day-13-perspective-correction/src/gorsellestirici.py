"""Perspektif Düzeltme ve Homografi Görselleştirici (Headless Matplotlib).

Bu modül; orijinal açılı görüntüyü işaretlenmiş köşe noktalarıyla,
kuşbakışı düzeltilmiş ortogonal çıktıyı ve 3x3 homografi projeksiyon matrisinin
ısı haritasını tek bir karşılaştırma paneli halinde kaydeder.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2


class PerspektifGorsellestirici:
    """Perspektif düzeltme adımlarını görselleştiren modül."""

    @classmethod
    def analiz_paneli_ciz(
        cls,
        orijinal_bgr: np.ndarray,
        sirali_noktalar: np.ndarray,
        duzeltilmis_bgr: np.ndarray,
        homografi_matrisi: np.ndarray,
        dosya_yolu: Path
    ) -> Path:
        """3 panelli perspektif analiz çizelgesini oluşturup diske kaydeder."""
        # 1. Panel için köşe çizgili görsel hazırla
        isaretli_gorsel = orijinal_bgr.copy()
        pts_int = sirali_noktalar.astype(np.int32).reshape((-1, 1, 2))

        # Kenarları yeşil çizgiyle birleştir
        cv2.polylines(isaretli_gorsel, [pts_int], isClosed=True, color=(0, 255, 0), thickness=2)

        etiketler = ["Sol-Üst", "Sağ-Üst", "Sağ-Alt", "Sol-Alt"]
        renkler = [(0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255)]

        for i, (pt, etiket, renk) in enumerate(zip(sirali_noktalar, etiketler, renkler)):
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(isaretli_gorsel, (x, y), 6, renk, -1)
            cv2.putText(
                isaretli_gorsel, f"{etiket} ({x},{y})",
                (x - 30 if x > 150 else x + 10, y - 10 if y > 30 else y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA
            )

        fig, eksenler = plt.subplots(1, 3, figsize=(16, 5.5), dpi=150)

        # 1. Panel: Açılı Çekim ve Köşeler
        eksenler[0].imshow(isaretli_gorsel[:, :, ::-1])
        eksenler[0].set_title("1. Açılı / Bozulmuş Görüntü\n(Tespit Edilen 4 Köşe)", fontsize=11, fontweight="bold")
        eksenler[0].axis("off")

        # 2. Panel: Kuşbakışı Düzeltilmiş Çıktı
        h_duz, w_duz = duzeltilmis_bgr.shape[:2]
        eksenler[1].imshow(duzeltilmis_bgr[:, :, ::-1])
        eksenler[1].set_title(f"2. Kuşbakışı Düzeltilmiş Çıktı\n({w_duz} x {h_duz} px - Ortogonal)", fontsize=11, fontweight="bold")
        eksenler[1].axis("off")

        # 3. Panel: 3x3 Homografi Matrisi Isı Haritası
        im_h = eksenler[2].imshow(homografi_matrisi, cmap="coolwarm", aspect="auto")
        eksenler[2].set_title("3. 3x3 Homografi Matrisi (H)\nProjeksiyon Ağırlıkları", fontsize=11, fontweight="bold")
        fig.colorbar(im_h, ax=eksenler[2], fraction=0.046, pad=0.04)

        for i in range(3):
            for j in range(3):
                deger = homografi_matrisi[i, j]
                eksenler[2].text(
                    j, i, f"{deger:.3e}" if abs(deger) > 1000 or abs(deger) < 0.001 else f"{deger:.3f}",
                    ha="center", va="center", color="black" if abs(deger) < np.max(np.abs(homografi_matrisi)) * 0.7 else "white",
                    fontsize=8, fontweight="bold"
                )

        eksenler[2].set_xticks([0, 1, 2])
        eksenler[2].set_yticks([0, 1, 2])
        eksenler[2].set_xticklabels(["h1", "h2", "h3"])
        eksenler[2].set_yticklabels(["x'", "y'", "w'"])

        fig.tight_layout()
        dosya_yolu.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dosya_yolu)
        plt.close(fig)
        return dosya_yolu
