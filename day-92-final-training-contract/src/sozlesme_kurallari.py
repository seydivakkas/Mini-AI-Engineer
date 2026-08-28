"""
Day 92: Eğitim Öncesi Veri Sözleşmesi Kuralları ve İhlal Modelleri
-----------------------------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class IhlalSeviyesi(Enum):
    BILGI = "BILGI"
    UYARI = "UYARI"
    BLOKE_EDICI = "BLOKE_EDICI"


@dataclass
class KuralIhlali:
    kural_adi: str
    seviye: IhlalSeviyesi
    mesaj: str
    etkilenen_ornek_sayisi: int = 0
    metrik_degeri: Optional[float] = None


@dataclass
class VeriSozlesmesi:
    """
    Eğitim başlamadan önce veri setinin sağlaması gereken katı sözleşme kuralları.
    """
    beklenen_kanal: int = 3
    beklenen_yukseklik: int = 32
    beklenen_genislik: int = 32
    beklenen_dtype: str = "float32"

    # Sayısal sınırlar
    min_deger_limiti: float = -5.0
    maks_deger_limiti: float = 5.0
    nan_inf_yasak: bool = True

    # Veri seti hacmi ve etiket kuralları
    min_ornek_sayisi: int = 50
    beklenen_sinif_sayisi: int = 10
    maks_sinif_dengesizlik_orani: float = 10.0  # max(class_counts) / min(class_counts)
    nadir_sinif_min_ornek: int = 5

    # Veri sızıntısı (Data Leakage) toleransı
    maks_sizinti_toleransi: float = 0.0  # Train-Val arasında 0 sızıntı

    @property
    def beklenen_sekil(self) -> Tuple[int, int, int]:
        return (self.beklenen_kanal, self.beklenen_yukseklik, self.beklenen_genislik)
