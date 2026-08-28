"""
Semantik Arama Motoru (Semantic Search Engine Coordinator).
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from .vektorlestirici import CumleVektorlestirici
from .vektor_indeksi import DuzVektorIndeksi


class SemantikAramaMotoru:
    """
    Dokümanları yoğun vektörlere dönüştüren, indeksleyen ve
    anlamsal sorgulama (Semantic Search) sağlayan üst seviye motor.
    """

    def __init__(self, embed_dim: int = 128, device: str = "cpu"):
        self.vektorlestirici = CumleVektorlestirici(embed_dim=embed_dim, device=device)
        self.indeks = DuzVektorIndeksi(boyut=embed_dim)

    def dokuman_ekle(self, doc_id: str, baslik: str, icerik: str, kategori: str = "Genel"):
        """Tek bir dokümanı vektörleştirip indekse kaydeder."""
        metin = f"{baslik}. {icerik}"
        emb = self.vektorlestirici.vektorlestir(metin)
        self.indeks.ekle(
            doc_id=doc_id,
            vektor=emb,
            metaveri={"baslik": baslik, "icerik": icerik, "kategori": kategori}
        )

    def toplu_dokuman_ekle(self, dokumanlar: List[Dict[str, Any]]):
        """Toplu doküman listesini vektörleştirip indekse kaydeder."""
        metinler = [f"{d['baslik']}. {d['icerik']}" for d in dokumanlar]
        vektorler = self.vektorlestirici.vektorlestir(metinler)
        doc_idler = [d["id"] for d in dokumanlar]
        metaveriler = [{"baslik": d["baslik"], "icerik": d["icerik"], "kategori": d.get("kategori", "Genel")} for d in dokumanlar]

        self.indeks.toplu_ekle(doc_idler, vektorler, metaveriler)

    def semantik_ara(
        self,
        sorgu: str,
        top_k: int = 5,
        filtre_kategori: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Sorgu metnini vektörleştirip en yakın anlamsal dokümanları getirir."""
        sorgu_vektoru = self.vektorlestirici.vektorlestir(sorgu)
        return self.indeks.en_yakin_komsu_ara(
            sorgu_vektoru=sorgu_vektoru,
            top_k=top_k,
            filtre_kategori=filtre_kategori
        )

    def temsil_uzayi_pca_projeksiyonu(self) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Tüm indekslenmiş vektörleri PCA (Principal Component Analysis) ile
        2D görselleştirme uzayına indirger.
        """
        if self.indeks.toplam_vektor_sayisi < 2:
            return np.zeros((self.indeks.toplam_vektor_sayisi, 2)), self.indeks.doc_id_listesi, []

        X = self.indeks.vektorler - np.mean(self.indeks.vektorler, axis=0)
        # SVD ile PCA
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        pca_2d = np.dot(X, Vt[:2].T)

        doc_idler = self.indeks.doc_id_listesi
        kategoriler = [self.indeks.dokumanlar[d].get("kategori", "Genel") for d in doc_idler]
        return pca_2d, doc_idler, kategoriler
