"""
Soru Uygunluğu Ölçücü (Answer Relevance) Modülü (Day 140 - Faz 7).
Ragas & TruLens mimarisinde yanıtın soruya uygunluğunu ve odaklanma derecesini ölçen motor.
"""

from typing import Dict, Any
import math
import torch
import torch.nn.functional as F


class SoruUygunluguOlcucu:
    """Üretilen yanıtın kullanıcının sorusuna anlamsal uygunluğunu ve sapma derecesini ölçer."""

    def __init__(self, embedding_boyutu: int = 64):
        self.dim = embedding_boyutu

    def _vektorlestir(self, metin: str) -> torch.Tensor:
        """Karakter 3-gram tabanlı normalize anlamsal alt-kelime embedding üretir."""
        vec = torch.zeros(self.dim, dtype=torch.float32)
        temiz = metin.lower()
        # Kelime ve 3-gramlar
        for i in range(len(temiz) - 2):
            trigram = temiz[i : i + 3]
            v = sum(ord(c) * (31 ** j) for j, c in enumerate(trigram))
            idx = v % self.dim
            vec[idx] += 1.0

        norm = torch.norm(vec, p=2)
        if norm > 1e-6:
            vec = vec / norm
        return vec

    def olc(self, soru: str, yanit: str) -> Dict[str, Any]:
        """
        Answer Relevance Formülü: cos(E(soru), E(yanit)) + Anahtar Teknik Terim Uyumu
        """
        q_vec = self._vektorlestir(soru)
        a_vec = self._vektorlestir(yanit)

        kos_benzerlik = float(F.cosine_similarity(q_vec.unsqueeze(0), a_vec.unsqueeze(0)).item())
        uygunluk_skoru = max(0.0, min(1.0, kos_benzerlik))

        import re
        etkisiz_kelimeler = {"nasıl", "nedir", "ne", "ile", "ve", "veya", "için", "bir", "bu", "da", "de"}
        soru_terimleri = set(k for k in re.findall(r"\w+", soru.lower()) if len(k) >= 3 and k not in etkisiz_kelimeler)
        yanit_terimleri = set(k for k in re.findall(r"\w+", yanit.lower()) if len(k) >= 3)

        # Kök eşleşmesi kontrolü (ilk 4 harf)
        soru_kokleri = set(k[:4] for k in soru_terimleri)
        yanit_kokleri = set(k[:4] for k in yanit_terimleri)

        ortak_kokler = soru_kokleri.intersection(yanit_kokleri)
        terim_uyumu = min(1.0, (len(ortak_kokler) / max(1, len(soru_kokleri))) * 1.4)

        # Dengeli Soru Uygunluk Skoru
        nihai_skor = 0.45 * uygunluk_skoru + 0.55 * terim_uyumu
        nihai_skor = max(0.0, min(1.0, nihai_skor))

        return {
            "soru_uygunlugu_skoru": round(nihai_skor, 4),
            "kosinus_benzerligi": round(kos_benzerlik, 4),
            "anahtar_kelime_ortakligi": round(terim_uyumu, 4),
        }
