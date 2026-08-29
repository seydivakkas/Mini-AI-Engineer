"""
Bellek Kontrol Noktası (Checkpointing & Time Travel) Modülü (Day 127 - Faz 7).
Her düğüm geçişinde durum anlık görüntüsünü (snapshot) kaydeden ve geri sarma (Rollback) imkanı sunan yönetici.
"""

import copy
import time
from typing import Dict, Any, List, Optional


class CheckpointKaydi:
    """Tek bir kontrol noktası anlık görüntüsü."""

    def __init__(self, adim_no: int, dugum_adi: str, durum: Dict[str, Any]):
        self.adim_no = adim_no
        self.dugum_adi = dugum_adi
        self.durum = copy.deepcopy(durum)
        self.zaman = time.time()


class CheckpointYoneticisi:
    """Çizge adımlarını kaydeden ve zaman yolculuğu (Time Travel) sağlayan yönetici."""

    def __init__(self):
        self.gecmis: List[CheckpointKaydi] = []

    def kaydet(self, adim_no: int, dugum_adi: str, durum: Dict[str, Any]):
        """Durumun derin kopyasını kontrol noktası olarak kaydeder."""
        kayit = CheckpointKaydi(adim_no=adim_no, dugum_adi=dugum_adi, durum=durum)
        self.gecmis.append(kayit)

    def geri_sar(self, hedef_adim_no: int) -> Optional[Dict[str, Any]]:
        """Belirtilen adımdaki durum kopyasını döndürür (Rollback)."""
        for k in self.gecmis:
            if k.adim_no == hedef_adim_no:
                return copy.deepcopy(k.durum)
        return None

    def gecmis_ozeti(self) -> List[Dict[str, Any]]:
        """Kontrol noktalarının özet listesini döner."""
        return [
            {
                "adim": k.adim_no,
                "dugum": k.dugum_adi,
                "risk": k.durum.get("risk_skoru", 0.0),
                "durum": k.durum.get("nihai_durum", "ISLENIYOR"),
            }
            for k in self.gecmis
        ]
