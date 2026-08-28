"""
Hibrit Arama Yöneticisi (Hybrid Search Manager & Orchestrator).
"""

from typing import List, Dict, Any, Optional
from .leksikal_motor import LeksikalBM25Motoru
from .semantik_motor import SemantikVektorMotoru
from .rrf_fuzor import RRFFuzor, PuanNormalizasyonFuzor


class HibritAramaYoneticisi:
    """
    BM25 Leksikal Arama ve Dense Semantik Aramayı tek bir potada eriten
    uçtan uca Hibrit Arama Yöneticisi.
    """

    def __init__(self, rrf_k: int = 60, embed_dim: int = 128, device: str = "cpu"):
        self.bm25_motoru = LeksikalBM25Motoru(k1=1.5, b=0.75)
        self.semantik_motoru = SemantikVektorMotoru(embed_dim=embed_dim, device=device)
        self.rrf_fuzor = RRFFuzor(k=rrf_k)
        self.norm_fuzor = PuanNormalizasyonFuzor()

    def dokuman_ekle(self, doc_id: str, baslik: str, icerik: str, metaveri: Dict[str, Any] = None):
        """Dokümanı her iki arama motoruna da indeksler."""
        self.bm25_motoru.dokuman_ekle(doc_id, baslik, icerik, metaveri)
        self.semantik_motoru.dokuman_ekle(doc_id, baslik, icerik, metaveri)

    def toplu_dokuman_ekle(self, dokumanlar: List[Dict[str, Any]]):
        """Toplu dokümanları indeksler."""
        for d in dokumanlar:
            self.dokuman_ekle(
                doc_id=d["id"],
                baslik=d["baslik"],
                icerik=d["icerik"],
                metaveri=d.get("metaveri", {"kategori": d.get("kategori", "Genel")})
            )

    def hibrit_ara(
        self,
        sorgu: str,
        agirlik_bm25: float = 0.5,
        agirlik_semantik: float = 0.5,
        top_k: int = 5,
        fuzyon_yontemi: str = "rrf"
    ) -> Dict[str, Any]:
        """
        Sorguyu hem leksikal hem semantik motorlarda koşturup seçilen yöntemle (RRF / MinMax) birleştirir.
        """
        bm25_sonuclar = self.bm25_motoru.ara(sorgu, top_k=top_k * 2)
        semantik_sonuclar = self.semantik_motoru.ara(sorgu, top_k=top_k * 2)

        sonuc_listeleri = {
            "bm25": bm25_sonuclar,
            "semantik": semantik_sonuclar
        }
        agirliklar = {
            "bm25": agirlik_bm25,
            "semantik": agirlik_semantik
        }

        if fuzyon_yontemi.lower() == "rrf":
            final_sonuclar = self.rrf_fuzor.birlestir(sonuc_listeleri, agirliklar, top_k=top_k)
        else:
            final_sonuclar = self.norm_fuzor.normalize_ve_birlestir(sonuc_listeleri, agirliklar, top_k=top_k)

        return {
            "sorgu": sorgu,
            "fuzyon_yontemi": fuzyon_yontemi,
            "final_sonuclar": final_sonuclar,
            "bm25_sonuclari": bm25_sonuclar[:top_k],
            "semantik_sonuclari": semantik_sonuclar[:top_k]
        }
