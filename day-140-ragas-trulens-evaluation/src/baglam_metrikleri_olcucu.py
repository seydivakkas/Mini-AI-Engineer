"""
Bağlam Metrikleri Ölçücü (Context Recall & Precision) Modülü (Day 140 - Faz 7).
Ragas mimarisinde getirilen bağlamın doğruluğunu ve eksiksizliğini ölçen motor.
"""

from typing import List, Dict, Any
import re


class BaglamMetrikleriOlcucu:
    """Getirilen bağlamın Context Recall ve Context Precision değerlerini hesaplar."""

    @classmethod
    def olc(
        cls,
        getirilen_baglam_parcalari: List[str],
        referans_dogrulari: List[str],
    ) -> Dict[str, Any]:
        """
        Context Recall ve Context Precision@K hesaplaması.
        """
        if not referans_dogrulari or not getirilen_baglam_parcalari:
            return {"context_recall": 0.0, "context_precision": 0.0}

        tum_baglam = " ".join(getirilen_baglam_parcalari).lower()

        # 1. Context Recall Hesaplama
        karsilanan_referans_sayisi = 0
        for ref in referans_dogrulari:
            ref_kelimeleri = set(re.findall(r"\w+", ref.lower()))
            if not ref_kelimeleri:
                continue
            ortak = sum(1 for k in ref_kelimeleri if k in tum_baglam)
            if (ortak / len(ref_kelimeleri)) >= 0.50:
                karsilanan_referans_sayisi += 1

        context_recall = karsilanan_referans_sayisi / len(referans_dogrulari)

        # 2. Context Precision@K Hesaplama
        alakali_bayraklar: List[int] = []
        for parca in getirilen_baglam_parcalari:
            parca_kucuk = parca.lower()
            is_relevant = 0
            for ref in referans_dogrulari:
                ref_kelimeleri = set(re.findall(r"\w+", ref.lower()))
                ortak = sum(1 for k in ref_kelimeleri if k in parca_kucuk)
                if (ortak / max(1, len(ref_kelimeleri))) >= 0.40:
                    is_relevant = 1
                    break
            alakali_bayraklar.append(is_relevant)

        toplam_alakali = sum(alakali_bayraklar)
        if toplam_alakali == 0:
            context_precision = 0.0
        else:
            precision_toplami = 0.0
            kosan_alakali = 0
            for k, bayrak in enumerate(alakali_bayraklar, start=1):
                if bayrak == 1:
                    kosan_alakali += 1
                    precision_toplami += (kosan_alakali / k)
            context_precision = precision_toplami / toplam_alakali

        return {
            "context_recall": round(context_recall, 4),
            "context_precision": round(context_precision, 4),
            "karsilanan_referans_sayisi": karsilanan_referans_sayisi,
            "toplam_referans_sayisi": len(referans_dogrulari),
            "alakali_parca_sayisi": toplam_alakali,
            "toplam_getirilen_parca": len(getirilen_baglam_parcalari),
        }
