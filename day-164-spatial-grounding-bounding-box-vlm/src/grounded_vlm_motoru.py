"""
Spatial Grounding VLM Motoru (Grounded VLM Engine) Modülü (Day 164 - FAZ 9).
Doğal dil referans komutlarını ayrıştırıp Bounding Box koordinatları üretir.
"""

from typing import Dict, Any, List
from .koordinat_ayristirici import KoordinatAyristirici
from .iou_degerlendirici import IoUDegerlendirici


class GroundedVLMMotoru:
    """RefCOCO ve Det-VLM Tarzı Görsel Referanslama Motoru."""

    @classmethod
    def test_senaryolarini_getir(cls) -> List[Dict[str, Any]]:
        return [
            {
                "id": "ground_01",
                "nesne_adi": "Kırmızı Spor Araba",
                "komut": "<image>\nGörseldeki 'kırmızı spor araba'yı tespit et ve koordinatlarını ver.",
                "model_yaniti": "Tespit edilen kırmızı spor araba konumu: [210, 150, 680, 820]",
                "ground_truth_kutu": [200, 140, 690, 830],  # [ymin, xmin, ymax, xmax]
            },
            {
                "id": "ground_02",
                "nesne_adi": "Masanın Üzerindeki Laptop",
                "komut": "<image>\n'Masanın üzerindeki açık laptop' nerede?",
                "model_yaniti": "Laptop koordinatları: [450, 320, 780, 640]",
                "ground_truth_kutu": [440, 310, 790, 650],
            },
            {
                "id": "ground_03",
                "nesne_adi": "Sağdaki Kahve Fincanı",
                "komut": "<image>\n'Sağ köşede duran kahve fincanı'nı kutu içine al.",
                "model_yaniti": "Kahve fincanı konumu: [620, 750, 890, 940]",
                "ground_truth_kutu": [600, 740, 900, 950],
            },
            {
                "id": "ground_04",
                "nesne_adi": "Arka Plandaki Sokak Lambası",
                "komut": "<image>\n'Arka plandaki uzun sokak lambası' nerede?",
                "model_yaniti": "Sokak lambası: [50, 80, 480, 160]",
                "ground_truth_kutu": [40, 70, 490, 170],
            },
        ]

    @classmethod
    def senaryolari_degerlendir(cls) -> Dict[str, Any]:
        """Tüm Grounding senaryolarını çalıştırıp IoU metriklerini derler."""
        senaryolar = cls.test_senaryolarini_getir()
        tahminler = []
        gercekler = []

        sonuclar = []
        for s in senaryolar:
            kutular = KoordinatAyristirici.metinden_koordinat_cikar(s["model_yaniti"])
            tahmin_kutu = kutular[0] if kutular else [0, 0, 0, 0]
            gt_kutu = s["ground_truth_kutu"]

            iou = IoUDegerlendirici.iou_hesapla(tahmin_kutu, gt_kutu)
            tahminler.append(tahmin_kutu)
            gercekler.append(gt_kutu)

            sonuclar.append({
                "id": s["id"],
                "nesne_adi": s["nesne_adi"],
                "tahmin_kutu": tahmin_kutu,
                "gt_kutu": gt_kutu,
                "iou": round(iou, 3),
                "dogru_mu": iou >= 0.5,
            })

        ozet = IoUDegerlendirici.toplu_degerlendir(tahminler, gercekler, esik_degeri=0.5)

        return {
            "senaryo_sonuclari": sonuclar,
            "genel_ozet": ozet,
        }
