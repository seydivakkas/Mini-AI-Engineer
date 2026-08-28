"""
Anlamsal (Semantic), Örnek Tabanlı (Instance) ve Panoptik (Panoptic) Bölütleme Yapıları ve Dönüştürücüleri.
"""

from enum import Enum
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


class BolutlemeTipi(Enum):
    SEMANTIC = "semantic"   # Yalnızca sınıf etiketleri (Örnek ayrımı yok, stuff+things)
    INSTANCE = "instance"   # Sayılabilir nesneler (Things) için tekil maskeler ve kutular
    PANOPTIC = "panoptic"   # Hem Stuff (arkaplan) hem Things (nesneler) tekil ID'ler ile birleşik


class PanoptikDonusturucu:
    """
    Anlamsal ve Örnek tabanlı bölütleme verilerini birleştirerek
    Panoptik Harita ve Segment formatına dönüştürür.
    """

    STUFF_CLASSES = {0: "Arka Plan / Gökyüzü", 1: "Yol / Zemin"}
    THING_CLASSES = {2: "Araç", 3: "Yaya", 4: "Engel"}

    @classmethod
    def birlestir_panoptik(
        cls,
        semantik_harita: np.ndarray,
        ornek_maskeleri: List[np.ndarray],
        ornek_siniflari: List[int],
        ornek_skorlari: Optional[List[float]] = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Anlamsal harita ile tekil nesne maskelerini çakışma kurallarına göre birleştirir.

        Panoptik ID Kodlaması:
            panoptic_id = sinif_id * 1000 + ornek_id
            Stuff sınıfları için: sinif_id * 1000 + 0

        Dönüş:
            panoptik_harita: (H, W) boyutunda tam sayı haritası
            segment_bilgileri: Her segment için sınıf_id, ornek_id, alan vb. meta bilgiler
        """
        h, w = semantik_harita.shape
        panoptik_harita = np.zeros((h, w), dtype=np.int32)
        segment_bilgileri = []

        # 1. Aşama: Stuff (Arka Plan) sınıflarını yerleştir
        for stuff_id in cls.STUFF_CLASSES.keys():
            maske = (semantik_harita == stuff_id)
            if np.any(maske):
                pid = stuff_id * 1000
                panoptik_harita[maske] = pid
                segment_bilgileri.append({
                    "id": pid,
                    "kategori_id": stuff_id,
                    "kategori_tipi": "stuff",
                    "kategori_adi": cls.STUFF_CLASSES[stuff_id],
                    "alan": int(np.sum(maske)),
                    "ornek_id": 0
                })

        # 2. Aşama: Thing (Nesne Örnekleri) sınıflarını skor/sıra önceliğine göre yerleştir
        # Eğer skorlar varsa, yüksek skorlu olanlar öne geçer
        if ornek_skorlari is not None:
            sirali_indeksler = np.argsort(ornek_skorlari)[::-1]
        else:
            sirali_indeksler = list(range(len(ornek_maskeleri)))

        for inst_idx in sirali_indeksler:
            maske = ornek_maskeleri[inst_idx] > 0.5
            sinif_id = ornek_siniflari[inst_idx]
            ornek_id = inst_idx + 1
            pid = sinif_id * 1000 + ornek_id

            if np.any(maske):
                # Çakışan pikselleri ezerek nesneyi öne yerleştir
                panoptik_harita[maske] = pid
                kategori_adi = cls.THING_CLASSES.get(sinif_id, f"Nesne-{sinif_id}")
                segment_bilgileri.append({
                    "id": pid,
                    "kategori_id": sinif_id,
                    "kategori_tipi": "thing",
                    "kategori_adi": kategori_adi,
                    "alan": int(np.sum(maske)),
                    "ornek_id": ornek_id
                })

        return panoptik_harita, segment_bilgileri

    @classmethod
    def id_coz(cls, panoptik_id: int) -> Tuple[int, int]:
        """Panoptik ID'den (sinif_id, ornek_id) çiftini ayrıştırır."""
        sinif_id = panoptik_id // 1000
        ornek_id = panoptik_id % 1000
        return sinif_id, ornek_id
