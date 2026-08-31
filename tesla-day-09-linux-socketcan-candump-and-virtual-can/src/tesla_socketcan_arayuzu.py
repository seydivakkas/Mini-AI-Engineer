"""
Tesla Linux SocketCAN ve Sanal CAN (vcan0) Modulu
=================================================
Bu modul; Linux kernel ag yigini tabanli SocketCAN mimarisini (`PF_CAN`, `CAN_RAW`),
`struct can_frame` ikili serilestirmesini ve kernel seviyesinde donanimsal maskeleme
filtrelerini (`struct can_filter`) gercekler.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import struct
import time


@dataclass
class TeslaCANFrame:
    """
    Linux `struct can_frame` esdegeri.
    typedef __u32 canid_t;
    struct can_frame {
        canid_t can_id;  /* 32 bit CAN_ID + EFF/RTR/ERR flags */
        __u8    can_dlc; /* frame payload length in byte (0 .. 8) */
        __u8    __pad;   /* padding */
        __u8    __res0;  /* reserved / padding */
        __u8    __res1;  /* reserved / padding */
        __u8    data[8] __attribute__((aligned(8)));
    };
    """
    can_id: int
    can_dlc: int
    data: bytes

    def to_bytes(self) -> bytes:
        """16 baytlık standart Linux SocketCAN struct can_frame serileştirmesi."""
        pad = 8 - len(self.data)
        dolu_data = self.data + (b'\x00' * pad) if pad > 0 else self.data[:8]
        return struct.pack("=IB3x8s", self.can_id, self.can_dlc, dolu_data)

    @classmethod
    def from_bytes(cls, ham_baytlar: bytes) -> 'TeslaCANFrame':
        can_id, dlc, data_padded = struct.unpack("=IB3x8s", ham_baytlar[:16])
        return cls(can_id=can_id, can_dlc=dlc, data=data_padded[:dlc])


@dataclass
class TeslaCANFiltresi:
    """
    Linux `struct can_filter` esdegeri.
    Kural: (received_can_id & can_mask) == (can_id & can_mask)
    """
    can_id: int
    can_mask: int = 0x7FF  # 11-bit standard CAN mask

    def eslesiyor_mu(self, gelen_id: int) -> bool:
        return (gelen_id & self.can_mask) == (self.can_id & self.can_mask)


class TeslaSocketCANArayuzu:
    """
    Linux `socket(PF_CAN, SOCK_RAW, CAN_RAW)` soyutlamasi.
    """
    def __init__(self, arayuz_adi: str = "vcan0"):
        self.arayuz_adi = arayuz_adi
        self.filtreler: List[TeslaCANFiltresi] = []
        self.gelen_kuyruk: List[TeslaCANFrame] = []
        self.giden_kuyruk: List[TeslaCANFrame] = []
        self.bagli_mi = False

    def baglan(self) -> bool:
        self.bagli_mi = True
        return True

    def filtre_ekle(self, can_id: int, can_mask: int = 0x7FF):
        """`setsockopt(s, SOL_CAN_RAW, CAN_RAW_FILTER, ...)` donanım filtresi ekler."""
        self.filtreler.append(TeslaCANFiltresi(can_id=can_id, can_mask=can_mask))

    def frame_gonder(self, frame: TeslaCANFrame) -> bool:
        if not self.bagli_mi:
            return False
        self.giden_kuyruk.append(frame)
        return True

    def frame_al(self) -> Optional[TeslaCANFrame]:
        if not self.bagli_mi or not self.gelen_kuyruk:
            return None
        return self.gelen_kuyruk.pop(0)

    def kernel_filtresinden_gecir_ve_kabul_et(self, frame: TeslaCANFrame) -> bool:
        """Kernel seviyesi soket filtresi simülasyonu."""
        if not self.filtreler:
            self.gelen_kuyruk.append(frame)
            return True
            
        for f in self.filtreler:
            if f.eslesiyor_mu(frame.can_id):
                self.gelen_kuyruk.append(frame)
                return True
        return False  # Filtreye takıldı -> Kernel seviyesinde düşürüldü (Drop)


class TeslaVCanAgSimulatoru:
    """
    Çok düğümlü (BMS, Inverter, FSD) Sanal CAN (vcan0) Veri Yolu Simülatörü.
    """
    def __init__(self):
        self.dugumler: List[TeslaSocketCANArayuzu] = []

    def dugum_ekle(self, dugum: TeslaSocketCANArayuzu):
        dugum.baglan()
        self.dugumler.append(dugum)

    def yayinla(self, kaynak: TeslaSocketCANArayuzu, frame: TeslaCANFrame):
        """CAN veri yolundaki tüm düğümlere broadcast yayın yapar."""
        kaynak.frame_gonder(frame)
        for d in self.dugumler:
            if d is not kaynak:
                d.kernel_filtresinden_gecir_ve_kabul_et(frame)
