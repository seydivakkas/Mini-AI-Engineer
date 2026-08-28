"""
Ters İndeks (Inverted Index) Veri Yapısı ve Belge Yöneticisi.
"""

from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter


class TersIndeks:
    """
    Belgelerdeki terimlerin hangi belgelerde ve hangi sıklıkla (TF) geçtiğini
    haritalayan ters indeks (inverted index) veri yapısı.
    """

    def __init__(self):
        # postings[terim] = {doc_id: frekans}
        self.postings: Dict[str, Dict[str, int]] = defaultdict(dict)
        # belgeler[doc_id] = {"baslik": str, "icerik": str, "uzunluk": int, "tokenlar": List[str]}
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

    def belge_ekle(self, doc_id: str, baslik: str, icerik: str, tokenlar: List[str]):
        """Yeni bir belgeyi indeksler."""
        if doc_id in self.belgeler:
            # Varsa önceki uzunluğu çıkar
            self.toplam_kelime_sayisi -= self.belgeler[doc_id]["uzunluk"]

        uzunluk = len(tokenlar)
        self.belgeler[doc_id] = {
            "baslik": baslik,
            "icerik": icerik,
            "uzunluk": uzunluk,
            "tokenlar": tokenlar
        }
        self.toplam_kelime_sayisi += uzunluk

        # Terim frekanslarını say
        sayac = Counter(tokenlar)
        for terim, frekans in sayac.items():
            self.postings[terim][doc_id] = frekans

    def terim_gecen_belge_sayisi(self, terim: str) -> int:
        """Bir terimin geçtiği toplam belge sayısı n(q)."""
        return len(self.postings.get(terim, {}))

    def terim_frekansi(self, terim: str, doc_id: str) -> int:
        """Bir terimin belirli bir belgedeki frekansı f(q, D)."""
        return self.postings.get(terim, {}).get(doc_id, 0)

    def aday_belgeleri_getir(self, sorgu_tokenlari: List[str]) -> List[str]:
        """Sorgudaki en az bir terimi içeren aday belgelerin ID listesi."""
        aday_idleri = set()
        for t in sorgu_tokenlari:
            if t in self.postings:
                aday_idleri.update(self.postings[t].keys())
        return list(aday_idleri)
