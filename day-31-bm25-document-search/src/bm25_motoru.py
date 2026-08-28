"""
Okapi BM25 Arama ve Puanlama Motoru (Okapi BM25 Scoring Engine).
"""

from typing import Dict, List, Tuple, Any
import math
from .ters_indeks import TersIndeks


class OkapiBM25Motoru:
    """
    Okapi BM25 algoritması ile sorgu-belge uygunluk puanını (relevance score) hesaplayan motor.

    Parametreler:
        k1 (float): Terim frekansı doygunluk parametresi (varsayılan: 1.5).
        b (float): Belge uzunluğu normalizasyon/ceza katsayısı (varsayılan: 0.75).
        epsilon (float): Negatif IDF değerlerini önlemek için minimum taban katsayısı.
    """

    def __init__(self, indeks: TersIndeks, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.indeks = indeks
        self.k1 = float(k1)
        self.b = float(b)
        self.epsilon = float(epsilon)

    def idf_hesapla(self, terim: str) -> float:
        """
        Okapi BM25 IDF formülasyonu:
        IDF(q) = ln( (N - n(q) + 0.5) / (n(q) + 0.5) + 1 )
        """
        N = self.indeks.belge_sayisi
        n_q = self.indeks.terim_gecen_belge_sayisi(terim)

        if n_q == 0:
            return 0.0

        # Pozitifliği garanti altına alan Okapi IDF
        hesap = (N - n_q + 0.5) / (n_q + 0.5) + 1.0
        idf = math.log(max(hesap, 1.0 + self.epsilon))
        return float(idf)

    def belge_puani_hesapla(
        self,
        sorgu_tokenlari: List[str],
        doc_id: str
    ) -> Tuple[float, Dict[str, float]]:
        """
        Belirli bir belge için toplam BM25 skorunu ve terim bazlı katkı ayrışımını hesaplar.
        """
        belge_bilgi = self.indeks.belgeler.get(doc_id)
        if not belge_bilgi:
            return 0.0, {}

        doc_len = belge_bilgi["uzunluk"]
        avgdl = self.indeks.ortalama_belge_uzunlugu
        if avgdl == 0:
            return 0.0, {}

        toplam_skor = 0.0
        terim_katkilari = {}

        # BM25 Normalizasyon terimi: K = k1 * ( (1 - b) + b * (doc_len / avgdl) )
        K = self.k1 * ((1.0 - self.b) + self.b * (doc_len / avgdl))

        for terim in set(sorgu_tokenlari):
            f_q = self.indeks.terim_frekansi(terim, doc_id)
            if f_q == 0:
                continue

            idf = self.idf_hesapla(terim)
            # TF Doygunluk Terimi: (f_q * (k1 + 1)) / (f_q + K)
            tf_terimi = (f_q * (self.k1 + 1.0)) / (f_q + K)

            terim_puani = idf * tf_terimi
            toplam_skor += terim_puani
            terim_katkilari[terim] = float(terim_puani)

        return float(toplam_skor), terim_katkilari

    def sorgula(
        self,
        sorgu_tokenlari: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Sorgu tokenlarına göre en yüksek BM25 puanına sahip top-k belgeyi döndürür.
        """
        aday_belgeler = self.indeks.aday_belgeleri_getir(sorgu_tokenlari)
        sonuclar = []

        for doc_id in aday_belgeler:
            skor, katkilar = self.belge_puani_hesapla(sorgu_tokenlari, doc_id)
            if skor > 0.0:
                sonuclar.append({
                    "doc_id": doc_id,
                    "baslik": self.indeks.belgeler[doc_id]["baslik"],
                    "icerik": self.indeks.belgeler[doc_id]["icerik"],
                    "uzunluk": self.indeks.belgeler[doc_id]["uzunluk"],
                    "skor": skor,
                    "terim_katkilari": katkilar
                })

        # Skora göre azalan sırala
        sonuclar.sort(key=lambda x: x["skor"], reverse=True)
        return sonuclar[:top_k]
