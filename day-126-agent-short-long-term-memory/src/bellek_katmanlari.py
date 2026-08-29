"""
Çok Katmanlı Ajan Bellek Hiyerarşisi Modülü (Day 126 - Faz 7).
Working (Kısa Süreli), Episodic (Olay/Deneyim), Semantic (Uzun Süreli Vektör) ve Procedural (İşlemsel) bellek katmanları.
"""

from enum import Enum
import time
import uuid
from typing import Dict, Any, List, Optional
import numpy as np


class BellekTipi(Enum):
    CALISMA = "working"        # Anlık konuşma / kayan pencere
    EPISODIK = "episodic"      # Geçmiş olay ve görev geçmişi
    SEMANTIK = "semantic"      # Kullanıcı tercihleri, olgular ve uzun süreli vektör hafızası
    ISLEMSEL = "procedural"    # İşlem adımları ve kurallar


class BellekKaydi:
    """Tek bir bellek atomunu temsil eden veri sınıfı."""

    def __init__(
        self,
        metin: str,
        vektor: np.ndarray,
        tip: BellekTipi,
        onem_puani: float = 5.0,
        etiketler: Optional[List[str]] = None,
        bellek_id: Optional[str] = None,
    ):
        self.id = bellek_id or f"mem_{uuid.uuid4().hex[:8]}"
        self.metin = metin
        self.vektor = vektor / (np.linalg.norm(vektor) + 1e-9)  # L2 Normalize
        self.tip = tip
        self.onem_puani = onem_puani  # 1.0 - 10.0
        self.etiketler = etiketler or []
        self.olusturulma_zamani = time.time()
        self.son_erisim_zamani = self.olusturulma_zamani
        self.erisim_sayisi = 1
        self.gecerli_mi = True

    def erisim_kaydet(self):
        """Belleğe erişildiğinde zaman damgasını ve sayacı günceller."""
        self.son_erisim_zamani = time.time()
        self.erisim_sayisi += 1


class CalismaBellegi:
    """Anlık konuşma geçmişini tutan Kayan Pencere (Sliding Window) tamponu."""

    def __init__(self, kapasite: int = 5):
        self.kapasite = kapasite
        self.mesajlar: List[Dict[str, str]] = []

    def ekle(self, rol: str, icerik: str):
        self.mesajlar.append({"rol": rol, "icerik": icerik})
        if len(self.mesajlar) > self.kapasite:
            self.mesajlar.pop(0)

    def baglam_metni(self) -> str:
        return "\n".join([f"{m['rol'].upper()}: {m['icerik']}" for m in self.mesajlar])


class EpisodikBellek:
    """Geçmiş görev, oturum ve eylem geçmişini kronolojik olarak saklayan bellek."""

    def __init__(self):
        self.kayitlar: List[BellekKaydi] = []

    def ekle(self, kayit: BellekKaydi):
        self.kayitlar.append(kayit)

    def tumunu_listele(self) -> List[BellekKaydi]:
        return [k for k in self.kayitlar if k.gecerli_mi]


class SemantikBellek:
    """Kullanıcı tercihleri ve olgusal bilgileri vektör uzayında saklayan uzun süreli bellek."""

    def __init__(self):
        self.kayitlar: Dict[str, BellekKaydi] = {}

    def ekle_veya_guncelle(self, kayit: BellekKaydi):
        self.kayitlar[kayit.id] = kayit

    def gecersiz_kil(self, bellek_id: str):
        if bellek_id in self.kayitlar:
            self.kayitlar[bellek_id].gecerli_mi = False

    def aktif_kayitlar(self) -> List[BellekKaydi]:
        return [k for k in self.kayitlar.values() if k.gecerli_mi]
