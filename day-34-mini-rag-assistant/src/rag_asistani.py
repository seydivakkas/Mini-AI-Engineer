"""
Uçtan Uca Mini RAG Asistanı (Mini RAG Assistant Orchestrator).
"""

from typing import List, Dict, Any
from .metin_parcalayici import MetinParcalayici
from .vektor_deposu import VektorDeposu
from .rag_ureteci import RAGUreteci


class MiniRAGAsistani:
    """
    Doküman yükleme, parçalama (chunking), anlamsal indeksleme,
    bağlam enjeksiyonu ve kaynak atıflı soru-cevap asistanı.
    """

    def __init__(
        self,
        chunk_boyutu: int = 40,
        cakisma_miktari: int = 10,
        embed_dim: int = 128,
        guven_esigi: float = 0.20,
        device: str = "cpu"
    ):
        self.parcalayici = MetinParcalayici(chunk_boyutu=chunk_boyutu, cakisma_miktari=cakisma_miktari)
        self.vektor_deposu = VektorDeposu(embed_dim=embed_dim, device=device)
        self.uretec = RAGUreteci(guven_esigi=guven_esigi)
        self.dokuman_sayisi: int = 0
        self.parca_sayisi: int = 0

    def dokuman_ekle(self, doc_id: str, baslik: str, icerik: str, metaveri: Dict[str, Any] = None):
        """Tekil dokümanı parçalayıp vektör deposuna indeksler."""
        parcalar = self.parcalayici.parcala(doc_id, baslik, icerik, metaveri)
        self.vektor_deposu.parcalari_ekle(parcalar)
        self.dokuman_sayisi += 1
        self.parca_sayisi += len(parcalar)

    def toplu_dokuman_ekle(self, dokumanlar: List[Dict[str, Any]]):
        """Toplu dokümanları parçalayıp vektör deposuna indeksler."""
        parcalar = self.parcalayici.toplu_parcala(dokumanlar)
        self.vektor_deposu.parcalari_ekle(parcalar)
        self.dokuman_sayisi += len(dokumanlar)
        self.parca_sayisi += len(parcalar)

    def soru_sor(self, soru: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Kullanıcı sorusuna karşılık en ilgili parçaları getirir, prompt enjekte eder
        ve kaynak atıflı yanıt üretir.
        """
        getirilen_parcalar = self.vektor_deposu.arama(soru, top_k=top_k)
        hazir_prompt = self.uretec.prompt_hazirla(soru, getirilen_parcalar)
        sentez = self.uretec.yanit_sentezle(soru, getirilen_parcalar)

        return {
            "soru": soru,
            "getirilen_parcalar": getirilen_parcalar,
            "enjekte_prompt": hazir_prompt,
            "yanit": sentez["yanit"],
            "kaynaklar": sentez["kaynaklar"],
            "guven_skoru": sentez["guven_skoru"],
            "durum": sentez["durum"]
        }
