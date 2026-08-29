"""
Bilgi Grafı Getirici (Knowledge Graph Retrieval) Modülü (Day 139 - Faz 7).
Varlık eşleme, 2-hop komşuluk ve ilişkisel kenar gezintisi yapan graf getirme motoru.
"""

from typing import List, Dict, Any, Set
import re


class GrafGetirici:
    """Sorgudaki varlıkları eşleyip graf bağlantılarına göre ilgili belgeleri puanlayan motor."""

    def __init__(self):
        self.varlik_belge_haritasi: Dict[str, List[str]] = {}
        self.graf_kenarlari: List[Dict[str, Any]] = []

    def indeksle(self, belgeler: List[Dict[str, Any]], graf_kenarlari: List[Dict[str, Any]]):
        """Belgelerdeki varlık eşlemelerini ve graf kenarlarını kaydeder."""
        self.graf_kenarlari = graf_kenarlari
        self.varlik_belge_haritasi = {}
        for b in belgeler:
            for v in b.get("varliklar", []):
                self.varlik_belge_haritasi.setdefault(v.lower(), []).append(b["id"])

    def ara(self, sorgu: str, belgeler_haritasi: Dict[str, Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Sorgudaki varlıkları ve bunların 1-2 hop komşuluklarını bularak belgeleri puanlar."""
        sorgu_kucuk = sorgu.lower()

        # 1. Doğrudan Eşleşen Tohum Varlıklar (Seed Entities)
        eslesen_tohumlar: Set[str] = set()
        for v in self.varlik_belge_haritasi.keys():
            if re.search(r"\b" + re.escape(v) + r"\b", sorgu_kucuk):
                eslesen_tohumlar.add(v)

        # 2. 1-Hop ve 2-Hop Graf Gezintisi
        iliskili_varliklar: Set[str] = set(eslesen_tohumlar)
        for k in self.graf_kenarlari:
            ozne = k["ozne"].lower()
            nesne = k["nesne"].lower()
            if ozne in eslesen_tohumlar:
                iliskili_varliklar.add(nesne)
            if nesne in eslesen_tohumlar:
                iliskili_varliklar.add(ozne)

        # 3. Belge Puanlama
        belge_puanlari: Dict[str, float] = {}
        for v in iliskili_varliklar:
            agirlik = 2.0 if v in eslesen_tohumlar else 1.0
            for doc_id in self.varlik_belge_haritasi.get(v, []):
                belge_puanlari[doc_id] = belge_puanlari.get(doc_id, 0.0) + agirlik

        sirali = sorted(belge_puanlari.items(), key=lambda x: x[1], reverse=True)

        sonuclar = []
        for rank, (doc_id, skor) in enumerate(sirali[:top_k], start=1):
            if doc_id in belgeler_haritasi:
                doc = belgeler_haritasi[doc_id].copy()
                doc["graf_skoru"] = round(skor, 3)
                doc["graf_sirasi"] = rank
                sonuclar.append(doc)

        return sonuclar
