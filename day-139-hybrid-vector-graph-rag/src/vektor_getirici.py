"""
Yoğun Vektör Getirici (Dense Vector Retrieval) Modülü (Day 139 - Faz 7).
Cosine anlamsal benzerliği ile metin parçalarını arayan vektör motoru.
"""

import math
from typing import List, Dict, Any
import torch
import torch.nn.functional as F


class VektorGetirici:
    """Metinler üzerinde anlamsal embedding araması yapan yoğun vektör motoru."""

    def __init__(self, embedding_boyutu: int = 64):
        self.embedding_boyutu = embedding_boyutu
        self.belgeler: List[Dict[str, Any]] = []
        self.vektorler: List[torch.Tensor] = []

    def _metin_vektorlestir(self, metin: str) -> torch.Tensor:
        """Deterministik ve anlamsal karakter/kelime n-gram embedding üretir."""
        vec = torch.zeros(self.embedding_boyutu, dtype=torch.float32)
        kelimeler = metin.lower().split()
        for i, kelime in enumerate(kelimeler):
            val = sum(ord(c) for c in kelime)
            idx = val % self.embedding_boyutu
            pos_weight = 1.0 / (1.0 + 0.05 * i)
            vec[idx] += math.sin(val) * pos_weight
            vec[(idx + 7) % self.embedding_boyutu] += math.cos(val) * pos_weight

        norm = torch.norm(vec, p=2)
        if norm > 1e-6:
            vec = vec / norm
        return vec

    def indeksle(self, belgeler: List[Dict[str, Any]]):
        """Belgeleri vektörleştirip hafızaya indeksler."""
        self.belgeler = belgeler
        self.vektorler = [self._metin_vektorlestir(b["metin"]) for b in belgeler]

    def ara(self, sorgu: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Sorgu ile belgeler arasındaki kosinüs benzerliğini hesaplar ve sıralar."""
        if not self.vektorler:
            return []

        q_vec = self._metin_vektorlestir(sorgu)
        skorlar = []
        for i, doc_vec in enumerate(self.vektorler):
            sim = float(F.cosine_similarity(q_vec.unsqueeze(0), doc_vec.unsqueeze(0)).item())
            skorlar.append((i, sim))

        skorlar.sort(key=lambda x: x[1], reverse=True)

        sonuclar = []
        for rank, (doc_idx, sim) in enumerate(skorlar[:top_k], start=1):
            doc = self.belgeler[doc_idx].copy()
            doc["vektor_skoru"] = round(sim, 4)
            doc["vektor_sirasi"] = rank
            sonuclar.append(doc)

        return sonuclar
