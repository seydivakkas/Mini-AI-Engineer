"""
Tesla LIN (Local Interconnect Network) & BCM Modulu
===================================================
Bu modul; LIN 2.x Master-Slave mimarisini, Master Schedule Table cizelgelemesini,
Break/Sync (0x55), PID (Protected ID) parite hesaplamasini ve
Govde Kontrol Modulu (BCM - Pencere, Koltuk, Silecek, Ambians) suruculerini gercekler.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import time


def pid_hesapla(frame_id: int) -> int:
    """
    LIN 6-bit Frame ID'den 8-bit PID (Protected Identifier) üretir:
    P0 = ID0 ^ ID1 ^ ID2 ^ ID4
    P1 = ~(ID1 ^ ID3 ^ ID4 ^ ID5) & 1
    """
    id0 = (frame_id >> 0) & 1
    id1 = (frame_id >> 1) & 1
    id2 = (frame_id >> 2) & 1
    id3 = (frame_id >> 3) & 1
    id4 = (frame_id >> 4) & 1
    id5 = (frame_id >> 5) & 1

    p0 = id0 ^ id1 ^ id2 ^ id4
    p1 = (id1 ^ id3 ^ id4 ^ id5) ^ 1  # Not

    return (frame_id & 0x3F) | (p0 << 6) | (p1 << 7)


def pid_dogrula(pid: int) -> bool:
    """Gelen PID baytının parite bitlerini doğrular."""
    frame_id = pid & 0x3F
    beklenen_pid = pid_hesapla(frame_id)
    return pid == beklenen_pid


def gelismis_checksum_hesapla(pid: int, veri: bytes) -> int:
    """
    LIN 2.x Enhanced Checksum (PID + Veri baytlarının terslenmiş elde toplamı).
    """
    toplam = pid
    for b in veri:
        toplam += b
        if toplam > 0xFF:
            toplam = (toplam & 0xFF) + 1  # Elde ekleme (Carry fold)
    return (~toplam) & 0xFF


@dataclass
class TeslaLINMesaj:
    pid: int
    veri: bytes
    checksum: int
    gecerli_mi: bool


class TeslaLINSlaveBCM:
    """
    Gövde Kontrol Modülü (BCM) LIN Slave Aygıtı.
    Pencere motoru, koltuk ayarı, silecek ve ambiyans aydınlatma durumlarını yönetir.
    """
    def __init__(self):
        self.pencere_seviyesi_yuzde = 0.0      # 0% (Kapalı) - 100% (Açık)
        self.koltuk_pozisyonu_mm = 150.0       # 0 - 300 mm
        self.silecek_kademesi = 0              # 0: Kapalı, 1: Düşük, 2: Orta, 3: Yüksek
        self.ambiyans_rgb = (255, 255, 255)    # Beyaz

    def lin_mesaj_isle(self, mesaj: TeslaLINMesaj) -> Dict[str, Any]:
        if not mesaj.gecerli_mi:
            return {"durum": "HATA", "sebep": "GECERSIZ_PID_VEYA_CHECKSUM"}

        frame_id = mesaj.pid & 0x3F

        # 0x32 (50) -> Pencere Kontrolü
        if frame_id == 0x32:
            self.pencere_seviyesi_yuzde = float(mesaj.veri[0])
            return {"aygit": "Pencere", "yeni_deger": f"%{self.pencere_seviyesi_yuzde:.0f}"}

        # 0x14 (20) -> Koltuk Pozisyonu
        elif frame_id == 0x14:
            self.koltuk_pozisyonu_mm = float(mesaj.veri[0] * 2)
            return {"aygit": "Koltuk", "yeni_deger": f"{self.koltuk_pozisyonu_mm:.0f} mm"}

        # 0x0A (10) -> Silecek Hızı
        elif frame_id == 0x0A:
            self.silecek_kademesi = int(mesaj.veri[0])
            return {"aygit": "Silecek", "yeni_deger": f"Kademe {self.silecek_kademesi}"}

        # 0x28 (40) -> Ambiyans Aydınlatma RGB
        elif frame_id == 0x28:
            r, g, b = mesaj.veri[0], mesaj.veri[1], mesaj.veri[2]
            self.ambiyans_rgb = (r, g, b)
            return {"aygit": "Ambiyans_RGB", "yeni_deger": f"RGB({r},{g},{b})"}

        return {"durum": "BILINMEYEN_FRAME_ID"}


class TeslaLINMasterCizelgeleyici:
    """
    LIN Master Düğümü: Çizelgeleme Tablosunu (Schedule Table) döngüsel çalıştırır.
    Break (13-bit) + Sync (0x55) + PID başlıklarını veri yoluna basar.
    """
    def __init__(self, slave: TeslaLINSlaveBCM):
        self.slave = slave
        self.cizelge_tablosu: List[Dict[str, Any]] = [
            {"isim": "Pencere_Kaldırma", "frame_id": 0x32, "aralik_ms": 10.0, "veri": b'\x50'},   # %80 Açık
            {"isim": "Silecek_Kontrol",  "frame_id": 0x0A, "aralik_ms": 20.0, "veri": b'\x02'},   # Kademe 2
            {"isim": "Ambiyans_RGB",     "frame_id": 0x28, "aralik_ms": 50.0, "veri": b'\xFF\x00\x00'}, # Kırmızı
            {"isim": "Koltuk_Ayar",      "frame_id": 0x14, "aralik_ms": 100.0, "veri": b'\x64'}  # 200 mm
        ]

    def cerceve_gonder(self, frame_id: int, veri: bytes) -> TeslaLINMesaj:
        pid = pid_hesapla(frame_id)
        pid_ok = pid_dogrula(pid)
        csum = gelismis_checksum_hesapla(pid, veri)
        
        mesaj = TeslaLINMesaj(
            pid=pid,
            veri=veri,
            checksum=csum,
            gecerli_mi=pid_ok
        )
        return mesaj
