"""
Tesla Klasik CAN vs CAN-FD Protokol ve Arbitrasyon Modulu
=========================================================
Bu modul; Klasik CAN (CAN 2.0A/B - 8 byte) ile CAN-FD (Flexible Data-Rate - 64 byte)
cerceve yapilarini, BRS (Bit Rate Switch) cift fazli iletimini ve
Wired-AND bit duzeyinde donanim arbitrasyonunu (tahkimat) gercekler.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import time


DLC_CAN_FD_BYTE_HARITASI = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
    9: 12, 10: 16, 11: 20, 12: 24, 13: 32, 14: 48, 15: 64
}


@dataclass
class TeslaKlasikCANFrame:
    can_id: int  # 11-bit (0x000 - 0x7FF)
    veri: bytes  # Max 8 bytes
    nominal_bitrate_bps: int = 500000

    @property
    def dlc(self) -> int:
        return len(self.veri)

    def iletim_suresi_us_hesapla(self) -> float:
        # Standart CAN 2.0A Çerçevesi: ~108 bit (Stuff bitler dahil) + 8*8 bit = 108 + 64 = 172 bit
        toplam_bit = 47 + (self.dlc * 8) + 15  # Stuff bits yaklaşık
        bit_suresi_us = (1.0 / self.nominal_bitrate_bps) * 1e6
        return toplam_bit * bit_suresi_us


@dataclass
class TeslaCANFDFrame:
    can_id: int  # 11-bit veya 29-bit
    veri: bytes  # Max 64 bytes
    brs_aktif_mi: bool = True  # Bit Rate Switch
    nominal_bitrate_bps: int = 500000  # 500 kbps (Arbitrasyon)
    veri_bitrate_bps: int = 5000000     # 5 Mbps (Veri Fazı)

    @property
    def dlc(self) -> int:
        uzunluk = len(self.veri)
        for dlc_val, byte_sayisi in sorted(DLC_CAN_FD_BYTE_HARITASI.items()):
            if uzunluk <= byte_sayisi:
                return dlc_val
        return 15

    def iletim_suresi_us_hesapla(self) -> float:
        # Arbitrasyon Fazı (500 kbps): ~30 bit
        # Veri Fazı (5 Mbps - BRS aktifse): (Veri Byte * 8) + 21 bit CRC + ACK
        nominal_bit_sayisi = 32
        veri_bit_sayisi = (len(self.veri) * 8) + 28

        t_nominal_us = nominal_bit_sayisi * ((1.0 / self.nominal_bitrate_bps) * 1e6)
        if self.brs_aktif_mi:
            t_veri_us = veri_bit_sayisi * ((1.0 / self.veri_bitrate_bps) * 1e6)
        else:
            t_veri_us = veri_bit_sayisi * ((1.0 / self.nominal_bitrate_bps) * 1e6)

        return t_nominal_us + t_veri_us


class TeslaCANArbitrasyonSimulasyonu:
    """
    Wired-AND (Baskın '0' / Çekinik '1') Donanımsal Arbitrasyon Simülatörü.
    Aynı anda veri yoluna çerçeve gönderen düğümlerden en küçük CAN ID'ye sahip olan kazanır.
    """
    def __init__(self):
        self.mesaj_kuyrugu: List[Dict[str, Any]] = []

    def mesaj_ekle(self, dugum_adi: str, can_id: int, veri: bytes, oncelik_aciklamasi: str):
        self.mesaj_kuyrugu.append({
            "dugum_adi": dugum_adi,
            "can_id": can_id,
            "veri": veri,
            "aciklama": oncelik_aciklamasi
        })

    def arbitrasyon_yaristir(self) -> Dict[str, Any]:
        """
        Bit düzeyinde yarış simülasyonu: 11-bit ID ikili dizilimini karşılaştırır.
        """
        if not self.mesaj_kuyrugu:
            return {"kazanan": None, "elenenler": []}

        # İkili gösterim (11 bit)
        yarismacilar = []
        for m in self.mesaj_kuyrugu:
            ikili_id = format(m["can_id"], '011b')
            yarismacilar.append({
                "dugum_adi": m["dugum_adi"],
                "can_id": m["can_id"],
                "ikili_id": ikili_id,
                "aciklama": m["aciklama"]
            })

        aktifler = list(yarismacilar)
        elenenler = []

        for bit_index in range(11):
            if len(aktifler) <= 1:
                break

            # Veri yolundaki baskın bit (0) kontrolü
            baskim_bit_var_mi = any(y["ikili_id"][bit_index] == '0' for y in aktifler)
            
            yeni_aktifler = []
            for y in aktifler:
                mevcut_bit = y["ikili_id"][bit_index]
                if baskim_bit_var_mi and mevcut_bit == '1':
                    # Çekinik bit gönderdi ama hatta baskın 0 gördü -> ÇEKİL
                    elenenler.append({
                        "dugum_adi": y["dugum_adi"],
                        "can_id": y["can_id"],
                        "elendigi_bit": bit_index,
                        "sebep": f"Bit {bit_index}'te çekinik '1' gönderdi, baskın '0' gördü"
                    })
                else:
                    yeni_aktifler.append(y)
            aktifler = yeni_aktifler

        kazanan = aktifler[0] if aktifler else None
        return {
            "kazanan": kazanan,
            "elenenler": elenenler,
            "toplam_yarismaci": len(yarismacilar)
        }
