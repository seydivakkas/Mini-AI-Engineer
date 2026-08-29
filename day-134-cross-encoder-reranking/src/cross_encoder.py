"""
Cross-Encoder Derin Re-ranking Modülü (Day 134 - Faz 7 - 2. Aşama).
Soru ve belgeyi tek dizi olarak birleştirip token düzeyinde tam çapraz dikkat (Cross-Attention) ile puanlayan modül.
"""

from typing import List, Dict, Any, Tuple
import re
import numpy as np
import torch


class CrossEncoderReranker:
    """Soru ve belge arasındaki tüm token etkileşimlerini hesaplayan 2. Aşama Re-ranker."""

    def __init__(self, gizli_boyut: int = 64):
        self.gizli_boyut = gizli_boyut

    def _tokenlestir(self, metin: str) -> List[str]:
        """Basit token ayrıştırıcı."""
        temiz = re.sub(r"[^\w\s]", " ", metin.lower())
        tokens = [t for t in temiz.split() if len(t) > 1]
        return tokens if tokens else ["bos"]

    def puanla(self, sorgu: str, belge_metni: str) -> Tuple[float, np.ndarray]:
        """
        Sorgu ve belgeyi çapraz dikkat matrisi üzerinden analiz eder:
        1. Token gömmeleri üretilir.
        2. Çapraz dikkat matrisi A[i, j] = Softmax(Q_i * K_j / sqrt(d)) hesaplanır.
        3. Nihai anlamsal uygunluk skoru Sigmoid(W * A) olarak döner.
        """
        sorgu_tokenlar = self._tokenlestir(sorgu)
        belge_tokenlar = self._tokenlestir(belge_metni)

        n_q = len(sorgu_tokenlar)
        n_d = len(belge_tokenlar)

        # Deterministik token vektörleri üret
        q_matris = np.zeros((n_q, self.gizli_boyut), dtype=np.float32)
        d_matris = np.zeros((n_d, self.gizli_boyut), dtype=np.float32)

        for i, t in enumerate(sorgu_tokenlar):
            np.random.seed(abs(hash(t)) % (2**31))
            q_matris[i] = np.random.randn(self.gizli_boyut)

        for j, t in enumerate(belge_tokenlar):
            np.random.seed(abs(hash(t)) % (2**31))
            d_matris[j] = np.random.randn(self.gizli_boyut)

        # L2 Normalize
        q_matris = q_matris / (np.linalg.norm(q_matris, axis=1, keepdims=True) + 1e-8)
        d_matris = d_matris / (np.linalg.norm(d_matris, axis=1, keepdims=True) + 1e-8)

        # Çapraz Dikkat (Cross-Attention) Nokta Çarpımı: (n_q, n_d)
        nokta_carpim = np.dot(q_matris, d_matris.T) / np.sqrt(self.gizli_boyut)

        # Softmax Dikkat Ağırlıkları
        exp_skorlar = np.exp(nokta_carpim - np.max(nokta_carpim, axis=1, keepdims=True))
        dikkat_matrisi = exp_skorlar / (np.sum(exp_skorlar, axis=1, keepdims=True) + 1e-8)

        # Birebir Kelime Eşleşme Bonusu
        ortak_tokenlar = set(sorgu_tokenlar).intersection(set(belge_tokenlar))
        eslesme_orani = len(ortak_tokenlar) / max(1, len(set(sorgu_tokenlar)))

        # Ortalama Maksimum Dikkat + Anlamsal Örtüşme
        maks_dikkat = np.mean(np.max(dikkat_matrisi, axis=1))
        anlamsal_skor = 0.45 * maks_dikkat + 0.55 * eslesme_orani

        # 0-1 Sigmoid benzeri ölçekleme
        nihai_skor = float(1.0 / (1.0 + np.exp(-4.5 * (anlamsal_skor - 0.35))))
        return round(nihai_skor, 4), dikkat_matrisi

    def yeniden_sirala(
        self, sorgu: str, adaylar: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Aday listesini Cross-Encoder ile yeniden puanlayıp sıralar."""
        yeniden_puanlanmis = []

        for aday in adaylar:
            skor, _ = self.puanla(sorgu, aday["metin"])
            yeni_aday = dict(aday)
            yeni_aday["cross_encoder_skor"] = skor
            yeniden_puanlanmis.append(yeni_aday)

        # Yüksek skora göre azalan sırala
        yeniden_puanlanmis.sort(key=lambda x: x["cross_encoder_skor"], reverse=True)
        return yeniden_puanlanmis
