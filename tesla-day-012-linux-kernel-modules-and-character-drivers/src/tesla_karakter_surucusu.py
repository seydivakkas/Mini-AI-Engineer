"""
Tesla Linux Cekirdek Modulu (LKM) ve Karakter Surucusu (cdev)
============================================================
Bu modul; Linux kernel `cdev_add`, `struct file_operations` (`open`, `read`, `write`, `ioctl`),
`copy_from_user` ve `copy_to_user` guvenlik kontrollerini gercekler.
`/dev/tesla_tork_kontrol` karakter aygiti uzerinden `0xAA55` ASIL-D guvenlik anahtarli
motor tork komutlarini dogrular.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import struct
import time


# Linux ioctl Kodları: _IOW('T', 1, struct TeslaTorkPaketi)
IOCTL_TESLA_TORK_YAZ  = 0x40085401
IOCTL_TESLA_DURUM_OKU = 0x80085402

ASIL_D_GUVENLIK_ANAHTARI = 0xAA55


@dataclass
class TeslaTorkPaketi:
    guvenlik_anahtari: int  # 0xAA55 olmak zorunda
    hedef_tork_nm: float    # -500.0 ile +1000.0 Nm arasi
    rejenerasyon_etkin_mi: bool

    def to_bytes(self) -> bytes:
        return struct.pack("=Hfd", self.guvenlik_anahtari, self.hedef_tork_nm, 1.0 if self.rejenerasyon_etkin_mi else 0.0)

    @classmethod
    def from_bytes(cls, baytlar: bytes) -> 'TeslaTorkPaketi':
        anahtar, tork, rejen = struct.unpack("=Hfd", baytlar[:14])
        return cls(guvenlik_anahtari=anahtar, hedef_tork_nm=tork, rejenerasyon_etkin_mi=(rejen > 0.5))


class TeslaGuvenliKullaniciKopyalayici:
    """
    Linux `copy_from_user` ve `copy_to_user` bellek guvenlik katmani.
    Kullanici adresinin gecerliligini ve sinir asimlarini (buffer overflow) denetler.
    """
    @staticmethod
    def copy_from_user(kullanici_adresi: bytes, beklenen_boyut: int) -> Tuple[Optional[bytes], int]:
        """`copy_from_user(to, from, n)` - Kalan bayt sayisi dondurur (0 = Basarili)."""
        if not kullanici_adresi or len(kullanici_adresi) < beklenen_boyut:
            return None, beklenen_boyut - (len(kullanici_adresi) if kullanici_adresi else 0)
        return bytes(kullanici_adresi[:beklenen_boyut]), 0

    @staticmethod
    def copy_to_user(cekirdek_verisi: bytes, hedef_boyut: int) -> Tuple[bytes, int]:
        """`copy_to_user(to, from, n)`"""
        if len(cekirdek_verisi) > hedef_boyut:
            return cekirdek_verisi[:hedef_boyut], 0
        return cekirdek_verisi, 0


class TeslaTorkKarakterAygiti:
    """
    `/dev/tesla_tork_kontrol` Karakter Sürücüsü (Major: 240, Minor: 0).
    """
    def __init__(self, aygit_yolu: str = "/dev/tesla_tork_kontrol"):
        self.aygit_yolu = aygit_yolu
        self.major_no = 240
        self.minor_no = 0
        self.acik_mi = False
        self.guncel_tork_nm = 0.0
        self.toplam_komut_sayisi = 0
        self.reddedilen_komut_sayisi = 0

    def open(self) -> int:
        """`int tesla_open(struct inode *pinode, struct file *pfile)`"""
        self.acik_mi = True
        return 0  # 0 = Basarili

    def release(self) -> int:
        """`int tesla_release(struct inode *pinode, struct file *pfile)`"""
        self.acik_mi = False
        return 0

    def unlocked_ioctl(self, komut: int, kullanici_arg_baytlari: bytes) -> Tuple[int, str]:
        """
        `long tesla_ioctl(struct file *pfile, unsigned int cmd, unsigned long arg)`
        """
        if not self.acik_mi:
            return -1, "EBADF: Aygıt dosyası açık değil!"

        self.toplam_komut_sayisi += 1

        if komut == IOCTL_TESLA_TORK_YAZ:
            # 1. copy_from_user ile güvenli aktarım
            ham_veri, kalan = TeslaGuvenliKullaniciKopyalayici.copy_from_user(kullanici_arg_baytlari, 14)
            if kalan > 0 or ham_veri is None:
                self.reddedilen_komut_sayisi += 1
                return -14, "EFAULT: Geçersiz kullanıcı bellek adresi (copy_from_user hatası)!"

            paket = TeslaTorkPaketi.from_bytes(ham_veri)

            # 2. ASIL-D Güvenlik Anahtarı Doğrulaması
            if paket.guvenlik_anahtari != ASIL_D_GUVENLIK_ANAHTARI:
                self.reddedilen_komut_sayisi += 1
                return -1, f"EPERM: ASIL-D Güvenlik Anahtarı Geçersiz! (Alınan: 0x{paket.guvenlik_anahtari:X})"

            # 3. Fiziksel Tork Limitleri Kontrolü (-500 Nm ile +1000 Nm)
            if not (-500.0 <= paket.hedef_tork_nm <= 1000.0):
                self.reddedilen_komut_sayisi += 1
                return -22, f"EINVAL: Tork limiti aşıldı! (-500 .. +1000 Nm, İstenen: {paket.hedef_tork_nm:.1f} Nm)"

            self.guncel_tork_nm = paket.hedef_tork_nm
            return 0, f"BAŞARILI: Tork {self.guncel_tork_nm:.1f} Nm olarak uygulandı."

        elif komut == IOCTL_TESLA_DURUM_OKU:
            durum_baytlari = struct.pack("=fd", self.guncel_tork_nm, float(self.toplam_komut_sayisi))
            return 0, f"GÜNCEL TORK: {self.guncel_tork_nm:.1f} Nm"

        self.reddedilen_komut_sayisi += 1
        return -25, "ENOTTY: Desteklenmeyen ioctl komutu!"
