"""
Metin Parçalama ve Bölümleme Motoru (Text Chunker & Sliding Window).
"""

from typing import List, Dict, Any
import re


class MetinParcalayici:
    """
    Uzun dokümanları çakışmalı kayan pencere (sliding window overlap)
    stratejisiyle anlamsal parçalara (chunks) böler.
    """

    def __init__(self, chunk_boyutu: int = 40, cakisma_miktari: int = 10):
        if cakisma_miktari >= chunk_boyutu:
            raise ValueError("Çakışma miktarı (overlap) parça boyutundan (chunk_size) küçük olmalıdır.")
        self.chunk_boyutu = chunk_boyutu
        self.cakisma_miktari = cakisma_miktari

    def parcala(self, doc_id: str, baslik: str, icerik: str, metaveri: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Doküman metnini çakışmalı parçalara böler."""
        temiz_icerik = re.sub(r"\s+", " ", icerik).strip()
        kelimeler = temiz_icerik.split()

        if not kelimeler:
            return []

        adim = self.chunk_boyutu - self.cakisma_miktari
        parcalar = []
        chunk_idx = 0

        for i in range(0, len(kelimeler), adim):
            parca_kelimeler = kelimeler[i: i + self.chunk_boyutu]
            if not parca_kelimeler:
                break

            parca_metni = " ".join(parca_kelimeler)
            chunk_id = f"{doc_id}_chunk_{chunk_idx:02d}"

            parcalar.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "baslik": baslik,
                "metin": parca_metni,
                "kelime_sayisi": len(parca_kelimeler),
                "baslangic_idx": i,
                "bitis_idx": i + len(parca_kelimeler),
                "metaveri": metaveri or {}
            })
            chunk_idx += 1

            if i + self.chunk_boyutu >= len(kelimeler):
                break

        return parcalar

    def toplu_parcala(self, dokumanlar: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Toplu doküman listesini parçalar."""
        tum_parcalar = []
        for d in dokumanlar:
            parcalar = self.parcala(
                doc_id=d["id"],
                baslik=d["baslik"],
                icerik=d["icerik"],
                metaveri=d.get("metaveri", {"kategori": d.get("kategori", "Genel")})
            )
            tum_parcalar.extend(parcalar)
        return tum_parcalar
