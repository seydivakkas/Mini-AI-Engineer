"""
Belge Arama Sunucusu ve Sorgu Yönetim Motoru.
"""

from typing import List, Dict, Any, Optional
from .tokenlestirici import MetinTokenlestirici
from .ters_indeks import TersIndeks
from .bm25_motoru import OkapiBM25Motoru


class BelgeAramaSunucusu:
    """
    Belge korpusunu yöneten, otomatik tokenizasyon, indeksleme ve
    BM25 sorgulaması sunan üst seviye arama motoru.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.tokenlestirici = MetinTokenlestirici()
        self.indeks = TersIndeks()
        self.bm25 = OkapiBM25Motoru(self.indeks, k1=k1, b=b)

    def belge_ekle(self, doc_id: str, baslik: str, icerik: str):
        """Tek bir belgeyi işleyip indekse kaydeder."""
        birlesik_metin = f"{baslik} {icerik}"
        tokenlar = self.tokenlestirici.tokenlestir(birlesik_metin)
        self.indeks.belge_ekle(doc_id=doc_id, baslik=baslik, icerik=icerik, tokenlar=tokenlar)

    def toplu_belge_ekle(self, belge_listesi: List[Dict[str, str]]):
        """Toplu belge listesini indeksler."""
        for b in belge_listesi:
            self.belge_ekle(
                doc_id=b["id"],
                baslik=b.get("baslik", ""),
                icerik=b.get("icerik", "")
            )

    def ara(self, sorgu_metni: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Doğal dil metin sorgusunu BM25 ile aratır."""
        sorgu_tokenlari = self.tokenlestirici.tokenlestir(sorgu_metni)
        if not sorgu_tokenlari:
            return []
        return self.bm25.sorgula(sorgu_tokenlari, top_k=top_k)

    def parametre_duyarlilik_analizi(
        self,
        sorgu_metni: str,
        k1_listesi: List[float] = [0.5, 1.2, 1.5, 2.0, 3.0],
        b_listesi: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0]
    ) -> Dict[str, Any]:
        """
        Farklı k1 ve b parametre değerlerinde skor değişimlerini hesaplar.
        """
        sorgu_tokenlari = self.tokenlestirici.tokenlestir(sorgu_metni)
        analiz_sonuclari = {}

        for k1 in k1_listesi:
            for b in b_listesi:
                motor = OkapiBM25Motoru(self.indeks, k1=k1, b=b)
                sonuclar = motor.sorgula(sorgu_tokenlari, top_k=3)
                anahtar = f"k1={k1:.1f}_b={b:.2f}"
                analiz_sonuclari[anahtar] = sonuclar

        return analiz_sonuclari
