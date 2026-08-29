"""
Intersection over Union (IoU) ve mAP@0.5 Değerlendirici Modülü (Day 164 - FAZ 9).
Tahmin edilen Bounding Box ile Ground Truth kutusu arasındaki örtüşmeyi hesaplar.
"""

from typing import List, Dict, Any


class IoUDegerlendirici:
    """Bounding Box Kesişim/Birleşim Oranı (IoU) Hesaplayıcı."""

    @classmethod
    def iou_hesapla(cls, kutu1: List[int], kutu2: List[int]) -> float:
        """
        Kutular: [ymin, xmin, ymax, xmax] (Piksel veya Normalize 0-1000)
        """
        ymin1, xmin1, ymax1, xmax1 = kutu1
        ymin2, xmin2, ymax2, xmax2 = kutu2

        # Kesişim (Intersection) koordinatları
        inter_ymin = max(ymin1, ymin2)
        inter_xmin = max(xmin1, xmin2)
        inter_ymax = min(ymax1, ymax2)
        inter_xmax = min(xmax1, xmax2)

        inter_h = max(0, inter_ymax - inter_ymin)
        inter_w = max(0, inter_xmax - inter_xmin)
        inter_alan = inter_h * inter_w

        # Alanlar (Alan = Yükseklik * Genişlik)
        alan1 = max(0, ymax1 - ymin1) * max(0, xmax1 - xmin1)
        alan2 = max(0, ymax2 - ymin2) * max(0, xmax2 - xmin2)

        # Birleşim (Union)
        union_alan = alan1 + alan2 - inter_alan

        if union_alan <= 0:
            return 0.0

        return float(inter_alan / union_alan)

    @classmethod
    def toplu_degerlendir(
        cls,
        tahminler: List[List[int]],
        gercek_etiketler: List[List[int]],
        esik_degeri: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Tahmin kümesini IoU >= 0.5 kriterine göre değerlendirir.
        """
        iou_skorlari = []
        dogru_sayisi = 0

        for t_kutu, g_kutu in zip(tahminler, gercek_etiketler):
            skor = cls.iou_hesapla(t_kutu, g_kutu)
            iou_skorlari.append(skor)
            if skor >= esik_degeri:
                dogru_sayisi += 1

        toplam = len(tahminler)
        ortalama_iou = sum(iou_skorlari) / toplam if toplam > 0 else 0.0
        dogruluk_orani = (dogru_sayisi / toplam) * 100.0 if toplam > 0 else 0.0

        return {
            "toplam_nesne": toplam,
            "dogru_tespit_sayisi": dogru_sayisi,
            "ortalama_iou": round(ortalama_iou, 4),
            "map_50_yuzdesi": round(dogruluk_orani, 2),
            "bireysel_iou_skorlari": [round(s, 3) for s in iou_skorlari],
        }
