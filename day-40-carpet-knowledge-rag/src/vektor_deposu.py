"""
40.2: Metadata Filtreli Hibrit Vektör Deposu ve Arama Motoru (Hybrid Vector Store).
"""

from typing import List, Dict, Any, Optional
import re
import numpy as np


class TekstilVektorDeposu:
    """Tekstil ve halı teknik chunk'larını dense ve sparse TF-IDF uzayında indeksler."""

    def __init__(self, dense_agirligi: float = 0.60, sparse_agirligi: float = 0.40):
        self.w_dense = dense_agirligi
        self.w_sparse = sparse_agirligi
        self.chunk_deposu: List[Dict[str, Any]] = []
        self.kelime_sozlugu: Dict[str, int] = {}
        self.idf_degerleri: np.ndarray = np.array([])
        self.tfidf_matrisi: np.ndarray = np.array([])
        self.dense_matrisi: np.ndarray = np.array([])

    def _tokenize(self, metin: str) -> List[str]:
        """Türkçe karakter ve teknik sembol duyarlı tokenizasyon."""
        metin = metin.lower()
        metin = metin.replace("ı", "i").replace("ğ", "g").replace("ü", "u").replace("ş", "s").replace("ö", "o").replace("ç", "c")
        tokens = re.findall(r"\b[a-z0-9_/%.-]{2,}\b", metin)
        return tokens

    def indeksle(self, chunk_listesi: List[Dict[str, Any]]) -> None:
        """Tüm chunk'ları TF-IDF ve Dense Vektör uzayına indeksler."""
        self.chunk_deposu = chunk_listesi
        N = len(chunk_listesi)
        if N == 0:
            return

        # 1. Sözlük İnşası (Vocabulary)
        kelime_frekans = {}
        dokuman_frekans = {}

        for ch in chunk_listesi:
            tokens = set(self._tokenize(ch["metin"]))
            for t in tokens:
                dokuman_frekans[t] = dokuman_frekans.get(t, 0) + 1
            for t in self._tokenize(ch["metin"]):
                kelime_frekans[t] = kelime_frekans.get(t, 0) + 1

        # En sık geçen 1000 kelime
        secilen_kelimeler = sorted(kelime_frekans.keys(), key=lambda k: kelime_frekans[k], reverse=True)[:1000]
        self.kelime_sozlugu = {k: i for i, k in enumerate(secilen_kelimeler)}
        V = len(self.kelime_sozlugu)

        # 2. IDF Hesabı
        self.idf_degerleri = np.zeros(V, dtype=np.float64)
        for k, idx in self.kelime_sozlugu.items():
            df = dokuman_frekans.get(k, 1)
            self.idf_degerleri[idx] = np.log((N + 1.0) / (df + 1.0)) + 1.0

        # 3. TF-IDF ve Dense Matrisleri
        tfidf_listesi = []
        dense_listesi = []

        for ch in chunk_listesi:
            tokens = self._tokenize(ch["metin"])
            tf_vec = np.zeros(V, dtype=np.float64)
            for t in tokens:
                if t in self.kelime_sozlugu:
                    tf_vec[self.kelime_sozlugu[t]] += 1.0

            # TF log normalization: 1 + log(tf)
            tf_nonzero = tf_vec > 0
            tf_vec[tf_nonzero] = 1.0 + np.log(tf_vec[tf_nonzero])

            tfidf_vec = tf_vec * self.idf_degerleri
            norm = np.linalg.norm(tfidf_vec) + 1e-12
            tfidf_norm = tfidf_vec / norm
            tfidf_listesi.append(tfidf_norm)

            # Pseudo-Dense Embedding (Sub-word Hashing & Semantic Projections)
            dense_vec = np.zeros(64, dtype=np.float64)
            for t in tokens:
                h = abs(hash(t)) % 64
                dense_vec[h] += 1.0
            dense_norm = dense_vec / (np.linalg.norm(dense_vec) + 1e-12)
            dense_listesi.append(dense_norm)

        self.tfidf_matrisi = np.array(tfidf_listesi)
        self.dense_matrisi = np.array(dense_listesi)

    def sorgula(
        self,
        sorgu_metni: str,
        top_k: int = 3,
        kategori_filtresi: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Sorgu metni ile vektör deposunda hibrit benzerlik araması yapar."""
        if len(self.chunk_deposu) == 0:
            return []

        tokens = self._tokenize(sorgu_metni)
        V = len(self.kelime_sozlugu)

        # 1. Sorgu TF-IDF
        q_tf = np.zeros(V, dtype=np.float64)
        for t in tokens:
            if t in self.kelime_sozlugu:
                q_tf[self.kelime_sozlugu[t]] += 1.0
        q_nonzero = q_tf > 0
        q_tf[q_nonzero] = 1.0 + np.log(q_tf[q_nonzero])
        q_tfidf = q_tf * self.idf_degerleri
        q_tfidf = q_tfidf / (np.linalg.norm(q_tfidf) + 1e-12)

        # 2. Sorgu Dense
        q_dense = np.zeros(64, dtype=np.float64)
        for t in tokens:
            h = abs(hash(t)) % 64
            q_dense[h] += 1.0
        q_dense = q_dense / (np.linalg.norm(q_dense) + 1e-12)

        # 3. Benzerlik Skorları
        sparse_skorlar = np.dot(self.tfidf_matrisi, q_tfidf)
        dense_skorlar = np.dot(self.dense_matrisi, q_dense)

        # Hibrit Füzyon
        hibrit_skorlar = (self.w_sparse * sparse_skorlar) + (self.w_dense * dense_skorlar)

        adaylar = []
        for idx, chunk in enumerate(self.chunk_deposu):
            # Metadata filtresi
            if kategori_filtresi and chunk["kategori"] != kategori_filtresi:
                continue

            skor = float(hibrit_skorlar[idx])
            adaylar.append({
                "chunk": chunk,
                "skor": float(round(skor, 4)),
                "sparse_skor": float(round(sparse_skorlar[idx], 4)),
                "dense_skor": float(round(dense_skorlar[idx], 4))
            })

        adaylar.sort(key=lambda x: x["skor"], reverse=True)
        return adaylar[:top_k]
