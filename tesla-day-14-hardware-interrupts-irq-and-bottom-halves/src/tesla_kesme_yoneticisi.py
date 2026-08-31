"""
Tesla Donanim Kesmeleri (IRQ), Top-Half ve Bottom-Half Modulu
============================================================
Bu modul; Linux kernel request_threaded_irq mekanizmasini,
HardIRQ (Top-Half - ultra hizli ACK) ile Kernel Thread (Bottom-Half - AEB Radar)
ayrimini ve kesme firtinasi (interrupt storm) onleme algoritmasini gercekler.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

from typing import Dict, Any, List, Optional, Tuple
import time
import math


class TeslaKesmeFirtinasiOnleyici:
    """
    Donanım hatası veya gürültü sebebiyle saniyede yüzbinlerce
    kesme gelmesini engelleyen Token-Bucket Hız Sınırlayıcı (Rate Limiter).
    """
    def __init__(self, maks_irq_hizi_sn: int = 10000):
        self.maks_irq_hizi_sn = maks_irq_hizi_sn
        self.jeton_sayisi = float(maks_irq_hizi_sn)
        self.son_guncelleme = time.perf_counter()
        self.engellenen_kesme_sayisi = 0

    def kesme_kabul_edilebilir_mi(self) -> bool:
        su_an = time.perf_counter()
        gecen_sure = su_an - self.son_guncelleme
        self.son_guncelleme = su_an

        # Jeton yenileme
        self.jeton_sayisi = min(float(self.maks_irq_hizi_sn), self.jeton_sayisi + gecen_sure * self.maks_irq_hizi_sn)

        if self.jeton_sayisi >= 1.0:
            self.jeton_sayisi -= 1.0
            return True
        else:
            self.engellenen_kesme_sayisi += 1
            return False


class TeslaTopHalfHardIRQ:
    """
    HardIRQ (Top-Half): Kesmeler kapalıyken (Interrupts Disabled) çalışır.
    ASLA uyuyamaz (cannot sleep/schedule), kilit bekleyemez, I/O yapamaz.
    Sadece donanım bayrağını (ACK) temizler ve Bottom-Half'i uyandırır (IRQ_WAKE_THREAD).
    """
    def __init__(self, irq_no: int = 42):
        self.irq_no = irq_no
        self.donanim_ack_bayragi = False
        self.top_half_sayaci = 0

    def kesme_isle(self) -> str:
        # 1. Donanım ACK register'ına yaz (simülasyon: bayrak sıfırlama)
        self.donanim_ack_bayragi = True
        self.top_half_sayaci += 1
        # 2. Kernel'e threaded handler'ı uyandırmasını söyle
        return "IRQ_WAKE_THREAD"


class TeslaBottomHalfThreadedIRQ:
    """
    Bottom-Half (Threaded IRQ / Workqueue): İşlemci bağlamında (Process Context) çalışır.
    Uyuyabilir, karmaşık matematiksel hesaplama (AEB Radar Nokta Bulutu & TTC) yapabilir.
    """
    def __init__(self, ttc_esigi_sn: float = 1.2):
        self.ttc_esigi_sn = ttc_esigi_sn
        self.islenen_paket_sayisi = 0
        self.acil_fren_tetiklendi_mi = False

    def radar_nokta_bulutu_isle(self, mesafe_m: float, bagil_hiz_mps: float) -> Dict[str, Any]:
        """
        Radar hedefini değerlendirir ve Çarpışma Zamanını (TTC - Time To Collision) hesaplar:
        TTC = mesafe / (-bagil_hiz)
        """
        self.islenen_paket_sayisi += 1
        
        ttc = float('inf')
        if bagil_hiz_mps < 0:  # Araç öndeki engele yaklaşıyor
            ttc = mesafe_m / (-bagil_hiz_mps)

        acil_fren = (ttc <= self.ttc_esigi_sn and ttc > 0)
        self.acil_fren_tetiklendi_mi = acil_fren

        return {
            "mesafe_m": mesafe_m,
            "bagil_hiz_mps": bagil_hiz_mps,
            "ttc_sn": ttc,
            "acil_fren_tetiklendi": acil_fren,
            "durum": "KRITIK_TEHLIKE" if acil_fren else "GUVENLI_MESAFE"
        }


class TeslaKesmeYonetimSistemi:
    """
    Top-Half ve Bottom-Half orkestrasyon motoru.
    """
    def __init__(self):
        self.firtina_onleyici = TeslaKesmeFirtinasiOnleyici(maks_irq_hizi_sn=10000)
        self.top_half = TeslaTopHalfHardIRQ(irq_no=42)
        self.bottom_half = TeslaBottomHalfThreadedIRQ(ttc_esigi_sn=1.2)

    def donanim_kesmesi_olustur(self, mesafe_m: float, hiz_mps: float) -> Dict[str, Any]:
        if not self.firtina_onleyici.kesme_kabul_edilebilir_mi():
            return {"durum": "KESME_FIRTINASI_ENGELLENDI"}

        # 1. Top-Half
        t_irq_sonuc = self.top_half.kesme_isle()

        # 2. Bottom-Half (Threaded)
        b_sonuc = self.bottom_half.radar_nokta_bulutu_isle(mesafe_m, hiz_mps)

        return {
            "top_half_sonuc": t_irq_sonuc,
            "bottom_half_sonuc": b_sonuc
        }
