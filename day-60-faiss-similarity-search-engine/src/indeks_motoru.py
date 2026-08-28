"""
FAISS Vektör İndeksleme ve Hızlı Benzerlik Arama Motoru (IndexFlatIP, IndexIVFFlat, IndexHNSWFlat).
"""

from typing import Tuple, Dict, Any, Optional
from enum import Enum
import os
import time
import numpy as np
import faiss


class IndeksTuru(str, Enum):
    FLAT_IP = "IndexFlatIP"
    IVF_FLAT = "IndexIVFFlat"
    HNSW_FLAT = "IndexHNSWFlat"


class FAISSIndeksMotoru:
    """Milyonluk ölçekli L2-normalize vektörler için yüksek performanslı FAISS indeks yöneticisi."""

    def __init__(
        self,
        dim: int,
        indeks_turu: IndeksTuru = IndeksTuru.FLAT_IP,
        **kwargs
    ):
        self.dim = dim
        self.indeks_turu = indeks_turu
        self.kwargs = kwargs
        self.index: Optional[faiss.Index] = None
        self._indeks_olustur()

    def _indeks_olustur(self) -> None:
        """Belirtilen indeksi ve hiperparametrelerini başlatır."""
        if self.indeks_turu == IndeksTuru.FLAT_IP:
            self.index = faiss.IndexFlatIP(self.dim)
        elif self.indeks_turu == IndeksTuru.IVF_FLAT:
            nlist = self.kwargs.get("nlist", 100)
            quantizer = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
        elif self.indeks_turu == IndeksTuru.HNSW_FLAT:
            M = self.kwargs.get("M", 32)
            ef_construction = self.kwargs.get("efConstruction", 64)
            self.index = faiss.IndexHNSWFlat(self.dim, M, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = ef_construction
        else:
            raise ValueError(f"Desteklenmeyen indeks türü: {self.indeks_turu}")

    def egit_ve_ekle(self, vektorler: np.ndarray) -> float:
        """Vektörleri indekse ekler, gerekirse kümeleme eğitimini (train) yürütür ve geçen süreyi döndürür."""
        vektorler = np.ascontiguousarray(vektorler, dtype=np.float32)
        assert vektorler.shape[1] == self.dim, f"Vektör boyutu ({vektorler.shape[1]}) ile indeks boyutu ({self.dim}) uyuşmuyor!"

        start = time.perf_counter()
        if not self.index.is_trained:
            self.index.train(vektorler)

        self.index.add(vektorler)
        gecen_sure = time.perf_counter() - start
        return gecen_sure

    def ara(
        self,
        sorgu_vektorleri: np.ndarray,
        top_k: int = 10,
        nprobe: Optional[int] = None,
        ef_search: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Toplu sorgu vektörleri için top-k en benzer indeksleri ve kosinüs skorlarını döndürür."""
        sorgu_vektorleri = np.ascontiguousarray(sorgu_vektorleri, dtype=np.float32)

        # Çalışma zamanı hiperparametre ayarı
        if self.indeks_turu == IndeksTuru.IVF_FLAT and nprobe is not None:
            self.index.nprobe = nprobe
        elif self.indeks_turu == IndeksTuru.HNSW_FLAT and ef_search is not None:
            self.index.hnsw.efSearch = ef_search

        start = time.perf_counter()
        skorlar, indeksler = self.index.search(sorgu_vektorleri, top_k)
        arama_suresi_ms = (time.perf_counter() - start) * 1000.0

        return skorlar, indeksler, arama_suresi_ms

    @property
    def toplam_vektor(self) -> int:
        return self.index.ntotal if self.index is not None else 0

    def indeksi_kaydet(self, dosya_yolu: str) -> None:
        """FAISS indeksini diske serileştirir."""
        os.makedirs(os.path.dirname(os.path.abspath(dosya_yolu)), exist_ok=True)
        faiss.write_index(self.index, dosya_yolu)

    @classmethod
    def indeksi_yukle(cls, dosya_yolu: str) -> "FAISSIndeksMotoru":
        """Diskteki serileştirilmiş indeksi belleğe geri yükler."""
        if not os.path.exists(dosya_yolu):
            raise FileNotFoundError(f"İndeks dosyası bulunamadı: {dosya_yolu}")
        loaded_index = faiss.read_index(dosya_yolu)
        motor = cls(dim=loaded_index.d, indeks_turu=IndeksTuru.FLAT_IP)
        motor.index = loaded_index
        return motor
