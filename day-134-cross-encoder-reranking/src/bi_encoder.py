"""
Bi-Encoder Vektör Arama Modülü (Day 134 - Faz 7 - 1. Aşama).
Soru ve belgeyi bağımsız olarak vektörleştiren, hızlı aday kümesi üreten modül.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn.functional as F


class BiEncoderArama:
    """Soru ve belgeyi ayrı embedding tensörlerine dönüştürerek hızlı kosinüs araması yapar."""

    def __init__(self, vektor_boyutu: int = 128):
        self.vektor_boyutu = vektor_boyutu
        self.belgeler: List[Dict[str, Any]] = []
        self.belge_embeddingleri: List[torch.Tensor] = []

    def _metin_vektorlestir(self, metin: str) -> torch.Tensor:
        """Deterministik ve anlamsal L2 normalize embedding üretir."""
        sozcukler = metin.lower().split()
        vektor = np.zeros(self.vektor_boyutu, dtype=np.float32)

        for idx, kelime in enumerate(sozcukler):
            np.random.seed(abs(hash(kelime)) % (2**31))
            agirlik = 1.0 / (idx + 1) ** 0.4
            vektor += np.random.randn(self.vektor_boyutu).astype(np.float32) * agirlik

        tensör = torch.tensor(vektor, dtype=torch.float32).unsqueeze(0)
        return F.normalize(tensör, p=2, dim=1)

    def belge_ekle(self, doc_id: str, metin: str, metadata: Dict[str, Any] = None):
        """Belgeyi ekler ve embedding'ini önceden hesaplar (Offline Pre-indexing)."""
        emb = self._metin_vektorlestir(metin)
        self.belgeler.append({
            "doc_id": doc_id,
            "metin": metin,
            "metadata": metadata or {},
        })
        self.belge_embeddingleri.append(emb)

    def toplu_belge_ekle(self, belgeler: List[Dict[str, Any]]):
        """Toplu belge ekleme."""
        for b in belgeler:
            self.belge_ekle(b["doc_id"], b["metin"], b.get("metadata", {}))

    def aday_getir(self, sorgu: str, top_k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """1. Aşama: Sorgu için en yüksek kosinüs benzerliğine sahip Top-K adayı getirir."""
        if not self.belgeler:
            return []

        sorgu_emb = self._metin_vektorlestir(sorgu)
        korpus_tensör = torch.cat(self.belge_embeddingleri, dim=0)
        benzerlikler = F.cosine_similarity(sorgu_emb, korpus_tensör).tolist()

        k = min(top_k, len(self.belgeler))
        sirali = sorted(zip(self.belgeler, benzerlikler), key=lambda x: x[1], reverse=True)
        return [(doc, float(skor)) for doc, skor in sirali[:k]]
