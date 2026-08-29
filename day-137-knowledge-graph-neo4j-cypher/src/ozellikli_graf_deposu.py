"""
Özellikli Graf Deposu (Property Graph Storage Engine) Modülü (Day 137 - Faz 7 - GraphRAG-2).
Neo4j LPG (Labeled Property Graph) mantığında düğüm, kenar ve nitelik yönetim motoru.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class Dugum:
    """Graf Düğümü (Node / Entity)."""
    id: str
    etiket: str = "Entity"
    ozellikler: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Kenar:
    """Yönlü Graf Kenarı (Directed Relationship / Edge)."""
    id: str
    kaynak_id: str
    hedef_id: str
    iliski_tipi: str
    agirlik: float = 1.0
    ozellikler: Dict[str, Any] = field(default_factory=dict)


class OzellikliGrafDeposu:
    """Bellek İçi Labeled Property Graph (LPG) Depolama Motoru."""

    def __init__(self):
        self.dugumler: Dict[str, Dugum] = {}
        self.kenarlar: Dict[str, Kenar] = {}
        self.cikis_komsulugu: Dict[str, List[str]] = {}  # dugum_id -> [kenar_id, ...]
        self.giris_komsulugu: Dict[str, List[str]] = {}  # dugum_id -> [kenar_id, ...]

    def dugum_ekle(self, id: str, etiket: str = "Entity", **ozellikler) -> Dugum:
        """Yeni bir düğüm ekler veya mevcut olanın özelliklerini günceller."""
        if id in self.dugumler:
            self.dugumler[id].etiket = etiket
            self.dugumler[id].ozellikler.update(ozellikler)
            return self.dugumler[id]

        dugum = Dugum(id=id, etiket=etiket, ozellikler=ozellikler)
        self.dugumler[id] = dugum
        self.cikis_komsulugu[id] = []
        self.giris_komsulugu[id] = []
        return dugum

    def kenar_ekle(
        self,
        kaynak_id: str,
        hedef_id: str,
        iliski_tipi: str,
        agirlik: float = 1.0,
        **ozellikler,
    ) -> Kenar:
        """İki düğüm arasına yönlü ilişki kenarı ekler."""
        if kaynak_id not in self.dugumler:
            self.dugum_ekle(kaynak_id)
        if hedef_id not in self.dugumler:
            self.dugum_ekle(hedef_id)

        kenar_id = f"{kaynak_id}__[{iliski_tipi}]__>{hedef_id}"
        if kenar_id in self.kenarlar:
            self.kenarlar[kenar_id].agirlik += agirlik
            self.kenarlar[kenar_id].ozellikler.update(ozellikler)
            return self.kenarlar[kenar_id]

        kenar = Kenar(
            id=kenar_id,
            kaynak_id=kaynak_id,
            hedef_id=hedef_id,
            iliski_tipi=iliski_tipi,
            agirlik=agirlik,
            ozellikler=ozellikler,
        )
        self.kenarlar[kenar_id] = kenar
        self.cikis_komsulugu[kaynak_id].append(kenar_id)
        self.giris_komsulugu[hedef_id].append(kenar_id)
        return kenar

    def dugum_getir(self, id: str) -> Optional[Dugum]:
        """Düğüm ID'sine göre düğümü döndürür."""
        return self.dugumler.get(id)

    def komsulari_getir(self, dugum_id: str, yon: str = "OUT") -> List[Tuple[Kenar, Dugum]]:
        """
        Düğümün komşularını ve bağlayan kenarları döndürür.
        yon: 'OUT' (çıkan), 'IN' (giren), 'BOTH' (her ikisi)
        """
        sonuclar: List[Tuple[Kenar, Dugum]] = []

        if yon in ("OUT", "BOTH") and dugum_id in self.cikis_komsulugu:
            for k_id in self.cikis_komsulugu[dugum_id]:
                k = self.kenarlar[k_id]
                hedef = self.dugumler[k.hedef_id]
                sonuclar.append((k, hedef))

        if yon in ("IN", "BOTH") and dugum_id in self.giris_komsulugu:
            for k_id in self.giris_komsulugu[dugum_id]:
                k = self.kenarlar[k_id]
                kaynak = self.dugumler[k.kaynak_id]
                sonuclar.append((k, kaynak))

        return sonuclar

    def tum_dugumler(self) -> List[Dugum]:
        return list(self.dugumler.values())

    def tum_kenarlar(self) -> List[Kenar]:
        return list(self.kenarlar.values())
