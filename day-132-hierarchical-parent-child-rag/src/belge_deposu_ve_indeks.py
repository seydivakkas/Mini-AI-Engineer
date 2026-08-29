"""
Belge Deposu (DocStore) ve Vektör İndeksleyici Modülü (Day 132 - Faz 7).
Ebeveyn parçaları Key-Value deposunda saklayan ve yalnızca çocuk parçaları vektörleştiren altyapı.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F

from .hiyerarsik_parcalayici import EbeveynParca, CocukParca


class BelgeDeposu:
    """Ebeveyn parçaları parent_id anahtarıyla saklayan Key-Value hafıza deposu."""

    def __init__(self):
        self.depo: Dict[str, EbeveynParca] = {}

    def ekle(self, ebeveyn: EbeveynParca):
        self.depo[ebeveyn.parent_id] = ebeveyn

    def toplu_ekle(self, ebeveynler: List[EbeveynParca]):
        for e in ebeveynler:
            self.ekle(e)

    def getir(self, parent_id: str) -> Optional[EbeveynParca]:
        return self.depo.get(parent_id)

    def toplu_getir(self, parent_id_listesi: List[str]) -> List[EbeveynParca]:
        return [self.depo[pid] for pid in parent_id_listesi if pid in self.depo]

    def boyut(self) -> int:
        return len(self.depo)


class VektorIndeksleyici:
    """Yalnızca küçük çocuk parçaları embedding'e dönüştürüp arayan vektör motoru."""

    def __init__(self, vektor_boyutu: int = 128):
        self.vektor_boyutu = vektor_boyutu
        self.cocuk_parcalar: List[CocukParca] = []
        self.embeddingler: Optional[torch.Tensor] = None

    def _metin_vektorlestir(self, metin: str) -> torch.Tensor:
        """Deterministik ve anlamsal L2 normalize embedding üretir."""
        sozcukler = metin.lower().split()
        vektor = np.zeros(self.vektor_boyutu, dtype=np.float32)

        for idx, kelime in enumerate(sozcukler):
            np.random.seed(abs(hash(kelime)) % (2**31))
            agirlik = 1.0 / (idx + 1) ** 0.5
            vektor += np.random.randn(self.vektor_boyutu).astype(np.float32) * agirlik

        tensör = torch.tensor(vektor, dtype=torch.float32).unsqueeze(0)
        return F.normalize(tensör, p=2, dim=1)

    def indeksle(self, cocuklar: List[CocukParca]):
        """Tüm çocuk parçaları indeksler."""
        self.cocuk_parcalar = cocuklar
        if not cocuklar:
            self.embeddingler = None
            return

        vektor_listesi = [self._metin_vektorlestir(c.metin) for c in cocuklar]
        self.embeddingler = torch.cat(vektor_listesi, dim=0)

    def en_yakin_cocuklari_getir(
        self, sorgu: str, top_k: int = 3
    ) -> List[Tuple[CocukParca, float]]:
        """Sorgu ile en yüksek benzerliğe sahip çocuk parçaları döndürür."""
        if self.embeddingler is None or len(self.cocuk_parcalar) == 0:
            return []

        sorgu_vektoru = self._metin_vektorlestir(sorgu)
        benzerlikler = F.cosine_similarity(sorgu_vektoru, self.embeddingler).squeeze(0)

        k = min(top_k, len(self.cocuk_parcalar))
        en_iyi_skorlar, en_iyi_indeksler = torch.topk(benzerlikler, k=k)

        sonuclar = []
        for idx, skor in zip(en_iyi_indeksler.tolist(), en_iyi_skorlar.tolist()):
            sonuclar.append((self.cocuk_parcalar[idx], float(skor)))

        return sonuclar
