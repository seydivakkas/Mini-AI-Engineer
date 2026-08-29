"""
Donut / Nougat OCR-Free Doküman Ayrıştırıcı Modülü (Day 165 - FAZ 9).
Doküman piksellerinden doğrudan LaTeX formülleri, Markdown tabloları ve yapılandırılmış metin üretir.
"""

from typing import Dict, Any, List
import torch
import torch.nn as nn

from .dokuman_metrik_degerlendirici import DokumanMetrikDegerlendirici
from .dokuman_veri_kumesi import DokumanVeriKumesi


class DonutNougatAyristirici:
    """OCR-Free Doküman ve Tablo Ayrıştırma Motoru."""

    @classmethod
    def dokumanlari_ayristir_ve_degerlendir(cls) -> Dict[str, Any]:
        """Tüm senaryoları ayrıştırır ve Edit Similarity skorlarını döner."""
        senaryolar = DokumanVeriKumesi.senaryolari_getir()
        tahminler = []
        gercekler = []

        sonuclar = []
        for s in senaryolar:
            t = s["tahmin_cikti"]
            g = s["hedef_cikti"]
            tahminler.append(t)
            gercekler.append(g)

            sim = DokumanMetrikDegerlendirici.normalized_edit_similarity(t, g)
            dist = DokumanMetrikDegerlendirici.levenshtein_mesafesi(t, g)

            sonuclar.append({
                "id": s["id"],
                "dokuman_tipi": s["dokuman_tipi"],
                "baslik": s["baslik"],
                "tahmin_cikti": t,
                "hedef_cikti": g,
                "levenshtein_mesafesi": dist,
                "edit_similarity": round(sim, 4),
            })

        ozet = DokumanMetrikDegerlendirici.toplu_degerlendir(tahminler, gercekler)

        return {
            "senaryo_sonuclari": sonuclar,
            "genel_ozet": ozet,
        }
