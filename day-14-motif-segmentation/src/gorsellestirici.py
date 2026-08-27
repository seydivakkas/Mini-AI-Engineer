"""Motif Segmentasyonu ve Sınırlayıcı Kutu Görselleştirici (Headless Matplotlib).

Bu modül; orijinal görüntüyü, Otsu ikili maskesini, sınırlayıcı kutularla
işaretlenmiş tespit görüntüsünü ve ayrıştırılmış münferit motif galerisini
tek bir 4 panelli karşılaştırma çizelgesi olarak kaydeder.
"""

from pathlib import Path
from typing import List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2
from src.motif_ayristirici import MotifBilgisi


class MotifGorsellestirici:
    """Motif segmentasyon analizlerini görselleştiren araç."""

    @classmethod
    def analiz_paneli_ciz(
        cls,
        orijinal_bgr: np.ndarray,
        ikili_maske: np.ndarray,
        motifler: List[MotifBilgisi],
        dosya_yolu: Path
    ) -> Path:
        """4 panelli motif segmentasyon çizelgesini oluşturup diske kaydeder."""
        # 1. Tespit Çizimli Görsel Hazırla
        isaretli_gorsel = orijinal_bgr.copy()

        for m in motifler:
            x, y, w, h = m.sinirlayici_kutu
            # Düz eksenli kutu (Kırmızı)
            cv2.rectangle(isaretli_gorsel, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Döndürülmüş minimum alanlı kutu (Sarı)
            kutu_int = m.dondurulmus_kutu.astype(np.int32)
            cv2.polylines(isaretli_gorsel, [kutu_int], isClosed=True, color=(0, 255, 255), thickness=2)

            # Merkez noktası ve ID etiketi
            cx, cy = int(m.merkez[0]), int(m.merkez[1])
            cv2.circle(isaretli_gorsel, (cx, cy), 4, (0, 255, 0), -1)
            cv2.putText(
                isaretli_gorsel, f"M-{m.motif_id}", (x + 4, y - 6 if y > 15 else y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA
            )

        fig = plt.figure(figsize=(15, 10), dpi=150)
        gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1.0])

        # Panel 1: Orijinal Görüntü
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(orijinal_bgr[:, :, ::-1])
        ax1.set_title("1. Orijinal Görüntü", fontsize=11, fontweight="bold")
        ax1.axis("off")

        # Panel 2: Otsu İkili Maske
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(ikili_maske, cmap="gray")
        ax2.set_title("2. Otsu + Morfolojik Maske", fontsize=11, fontweight="bold")
        ax2.axis("off")

        # Panel 3: Sınırlayıcı Kutular ve Tespitler
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(isaretli_gorsel[:, :, ::-1])
        ax3.set_title("3. Kontur & Sınırlayıcı Kutular\n(Kırmızı: Düz, Sarı: Döndürülmüş)", fontsize=11, fontweight="bold")
        ax3.axis("off")

        # Panel 4: Ayrıştırılmış Münferit Motifler Galerisi (Alt Satır)
        secilen_motifler = motifler[:6]  # En fazla ilk 6 motif
        adet = len(secilen_motifler)

        if adet > 0:
            sub_gs = gs[1, :].subgridspec(1, adet)
            for i, m in enumerate(secilen_motifler):
                ax_sub = fig.add_subplot(sub_gs[0, i])
                # Motifi maskeli şekilde göster (arka plan şeffaf/siyah)
                maske_3k = cv2.merge([m.kirpilmis_maske] * 3)
                izole_motif = cv2.bitwise_and(m.kirpilmis_gorsel, maske_3k)

                ax_sub.imshow(izole_motif[:, :, ::-1])
                baslik = (
                    f"Motif #{m.motif_id}\n"
                    f"Alan: {m.alan:.0f} px\n"
                    f"Dairesellik: {m.dairesellik:.2f}\n"
                    f"Solidity: {m.doluluk_orani:.2f}"
                )
                ax_sub.set_title(baslik, fontsize=8.5, fontweight="bold")
                ax_sub.axis("off")

        fig.tight_layout()
        dosya_yolu.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dosya_yolu)
        plt.close(fig)
        return dosya_yolu
