"""
Day 92: Hash Tabanlı Veri Sızıntısı (Data Leakage & Contamination) Dedektörü
---------------------------------------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from dataclasses import dataclass
from typing import List, Set, Tuple
import hashlib
import numpy as np
import torch


@dataclass
class SizintiRaporu:
    train_toplam: int
    val_toplam: int
    kesisen_ornek_sayisi: int
    sizinti_orani_val: float
    sizinti_var_mi: bool
    kesisen_indeksler: List[Tuple[int, int]]  # (train_idx, val_idx)


class VeriSizintiDedektoru:
    """
    Eğitim (Train) ve Doğrulama/Test (Val/Test) veri setleri arasındaki
    birebir yinelenen veya sızan örnekleri kriptografik hash parmak iziyle yakalayan motor.
    """

    def __init__(self, hassasiyet_ondalik: int = 4):
        self.hassasiyet_ondalik = hassasiyet_ondalik

    def _tensör_hash_hesapla(self, tensör: torch.Tensor) -> str:
        """Tensörü sabit hassasiyette yuvarlayıp SHA-256 özetini çıkarır."""
        if isinstance(tensör, torch.Tensor):
            dizi = tensör.detach().cpu().numpy()
        else:
            dizi = np.array(tensör)

        yuvarlanmis = np.round(dizi, decimals=self.hassasiyet_ondalik)
        baytlar = yuvarlanmis.tobytes()
        return hashlib.sha256(baytlar).hexdigest()

    def sizinti_tara(self, train_veriler: torch.Tensor, val_veriler: torch.Tensor) -> SizintiRaporu:
        """Train ve Val tensörleri arasındaki veri kirlenmesini (contamination) analiz eder."""
        train_n = len(train_veriler)
        val_n = len(val_veriler)

        train_hash_haritasi = {}
        for idx in range(train_n):
            h = self._tensör_hash_hesapla(train_veriler[idx])
            train_hash_haritasi[h] = idx

        kesisenler: List[Tuple[int, int]] = []
        for val_idx in range(val_n):
            vh = self._tensör_hash_hesapla(val_veriler[val_idx])
            if vh in train_hash_haritasi:
                train_idx = train_hash_haritasi[vh]
                kesisenler.append((train_idx, val_idx))

        kesisen_sayisi = len(kesisenler)
        sizinti_orani = float(kesisen_sayisi / val_n) if val_n > 0 else 0.0

        return SizintiRaporu(
            train_toplam=train_n,
            val_toplam=val_n,
            kesisen_ornek_sayisi=kesisen_sayisi,
            sizinti_orani_val=sizinti_orani,
            sizinti_var_mi=kesisen_sayisi > 0,
            kesisen_indeksler=kesisenler,
        )
