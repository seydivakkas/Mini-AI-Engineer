"""
Düz Vektör İndeksi (Flat Vector Index) ve Kosinüs Benzerliği Arama Motoru.
"""

from typing import List, Dict, Any, Optional
import numpy as np


class DuzVektorIndeksi:
    """
    Yoğun vektörleri bellekte saklayan ve tam kosinüs benzerliği (Exact Cosine k-NN)
    ile en yakın komşuları arayan vektör indeksi.
    """

    def __init__(self, boyut: int = 128):
        self.boyut = boyut
        self.vektorler: np.ndarray = np.empty((0, boyut), dtype=np.float32)
        self.doc_id_listesi: List[str] = []
        self.dokumanlar: Dict[str, Dict[str, Any]] = {}

    @property
    def toplam_vektor_sayisi(self) -> int:
        return len(self.doc_id_listesi)

    def ekle(self, doc_id: str, vektor: np.ndarray, metaveri: Dict[str, Any] = None):
        """Tek bir vektörü indekse ekler."""
        if doc_id in self.dokumanlar:
            # Mevcut dokümanı güncelle
            idx = self.doc_id_listesi.index(doc_id)
            self.vektorler[idx] = vektor
            self.dokumanlar[doc_id] = metaveri or {}
            return

        vektor = vektor.reshape(1, self.boyut)
        if len(self.vektorler) == 0:
            self.vektorler = vektor
        else:
            self.vektorler = np.vstack([self.vektorler, vektor])

        self.doc_id_listesi.append(doc_id)
        self.dokumanlar[doc_id] = metaveri or {}

    def toplu_ekle(self, doc_idler: List[str], vektorler: np.ndarray, metaveriler: List[Dict[str, Any]] = None):
        """Toplu vektör listesini indekse ekler."""
        for i, doc_id in enumerate(doc_idler):
            meta = metaveriler[i] if metaveriler and i < len(metaveriler) else {}
            self.ekle(doc_id, vektorler[i], meta)

    def en_yakin_komsu_ara(
        self,
        sorgu_vektoru: np.ndarray,
        top_k: int = 5,
        filtre_kategori: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Sorgu vektörünün indeksteki tüm vektörlerle kosinüs benzerliğini hesaplar ve
        en yüksek benzerliğe sahip ilk k dokümanı döndürür.
        """
        if self.toplam_vektor_sayisi == 0:
            return []

        # (1, D) x (N, D)^T -> (N,) Kosinüs Benzerlikleri
        sorgu = sorgu_vektoru.reshape(1, self.boyut)
        # Normalize vektörlerin skaler çarpımı doğrudan kosinüs benzerliğidir
        benzerlikler = np.dot(self.vektorler, sorgu.T).flatten()

        # Sıralama indisleri (azalan benzerlik)
        sirali_indisler = np.argsort(benzerlikler)[::-1]

        sonuclar = []
        for idx in sirali_indisler:
            doc_id = self.doc_id_listesi[idx]
            meta = self.dokumanlar.get(doc_id, {})

            if filtre_kategori and meta.get("kategori") != filtre_kategori:
                continue

            sonuclar.append({
                "doc_id": doc_id,
                "skor": float(benzerlikler[idx]),
                "baslik": meta.get("baslik", ""),
                "icerik": meta.get("icerik", ""),
                "kategori": meta.get("kategori", "Genel")
            })

            if len(sonuclar) >= top_k:
                break

        return sonuclar
