"""
Leksikal Arama Motoru (BM25 Lexical Search Engine).
"""

from typing import List, Dict, Any, Set
import math
import re
from collections import defaultdict, Counter


class LeksikalBM25Motoru:
    """Ters indeks tabanlı Okapi BM25 leksikal arama motoru."""

    STOP_WORDS: Set[str] = {
        "ve", "ile", "veya", "bu", "şu", "o", "bir", "de", "da", "ki", "için", "olan",
        "olarak", "gibi", "en", "daha", "çok", "the", "a", "an", "and", "or", "in", "to", "for"
    }

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.postings: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.belgeler: Dict[str, Dict[str, Any]] = {}
        self.toplam_kelime_sayisi: int = 0

    @property
    def belge_sayisi(self) -> int:
        return len(self.belgeler)

    @property
    def ortalama_belge_uzunlugu(self) -> float:
        if self.belge_sayisi == 0:
            return 0.0
        return self.toplam_kelime_sayisi / float(self.belge_sayisi)

    def _tokenlestir(self, metin: str) -> List[str]:
        temiz = re.sub(r"[^\w\s]", " ", metin.lower()).strip()
        return [t for t in temiz.split() if len(t) >= 2 and t not in self.STOP_WORDS]

    def dokuman_ekle(self, doc_id: str, baslik: str, icerik: str, metaveri: Dict[str, Any] = None):
        """Yeni dokümanı indeksler."""
        birlesik = f"{baslik} {icerik}"
        tokenlar = self._tokenlestir(birlesik)
        uzunluk = len(tokenlar)

        self.belgeler[doc_id] = {
            "baslik": baslik,
            "icerik": icerik,
            "uzunluk": uzunluk,
            "tokenlar": tokenlar,
            "metaveri": metaveri or {}
        }
        self.toplam_kelime_sayisi += uzunluk

        sayac = Counter(tokenlar)
        for terim, frekans in sayac.items():
            self.postings[terim][doc_id] = frekans

    def _idf(self, terim: str) -> float:
        N = self.belge_sayisi
        n_q = len(self.postings.get(terim, {}))
        if n_q == 0:
            return 0.0
        return math.log(((N - n_q + 0.5) / (n_q + 0.5)) + 1.0)

    def ara(self, sorgu_metni: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Sorguya göre en yüksek BM25 puanına sahip belgeleri döndürür."""
        sorgu_tokenlari = self._tokenlestir(sorgu_metni)
        if not sorgu_tokenlari or self.belge_sayisi == 0:
            return []

        avgdl = self.ortalama_belge_uzunlugu
        aday_belgeler = set()
        for t in sorgu_tokenlari:
            if t in self.postings:
                aday_belgeler.update(self.postings[t].keys())

        sonuclar = []
        for doc_id in aday_belgeler:
            doc_len = self.belgeler[doc_id]["uzunluk"]
            K = self.k1 * ((1.0 - self.b) + self.b * (doc_len / avgdl))

            skor = 0.0
            for t in set(sorgu_tokenlari):
                f_q = self.postings.get(t, {}).get(doc_id, 0)
                if f_q > 0:
                    idf = self._idf(t)
                    tf_term = (f_q * (self.k1 + 1.0)) / (f_q + K)
                    skor += idf * tf_term

            if skor > 0:
                sonuclar.append({
                    "doc_id": doc_id,
                    "skor": float(skor),
                    "baslik": self.belgeler[doc_id]["baslik"],
                    "icerik": self.belgeler[doc_id]["icerik"],
                    "metaveri": self.belgeler[doc_id]["metaveri"]
                })

        sonuclar.sort(key=lambda x: x["skor"], reverse=True)
        return sonuclar[:top_k]
