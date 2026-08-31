"""
Tesla RAII ve Donanim Kaynak Yoneticisi (Tesla C++20 RAII & Custom Deleters)
=============================================================================
Bu modul, Tesla otonom arac gomulu yazilimlarinda donanim kaynaklarinin (CAN soketleri,
DMA kanallari, GPU doku tamponlari) RAII (Resource Acquisition Is Initialization)
prensibi ve ozel siliciler (custom deleters) ile sifir sizinti garantili yonetimini gercekler.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

from enum import Enum
from typing import Callable, Any, Optional, Dict, List
import time


class DonanimKaynakTipi(Enum):
    CAN_SOKET = "CAN_BUS_SOCKET"
    GPU_TAMPON = "GPU_TEXTURE_BUFFER"
    DMA_KANAL = "DMA_DIRECT_MEMORY_CHANNEL"
    SERI_PORT = "SERIAL_TELEMETRY_PORT"


class TeslaDonanimKaynagi:
    """
    Fiziksel veya sanal bir Tesla gomulu donanim kaynagini temsil eder.
    """
    def __init__(self, kaynak_id: str, tip: DonanimKaynakTipi, aciklayici_no: int = 100):
        self.kaynak_id = kaynak_id
        self.tip = tip
        self.aciklayici_no = aciklayici_no  # File descriptor / Hardware handle
        self.acik_mi = True
        self.tahsis_zamani_ns = time.time_ns()
        self.kapanis_zamani_ns: Optional[int] = None
        self.islem_sayisi = 0

    def veri_gonder(self, veri: bytes) -> bool:
        """Donanim uzerinden veri transferi simule eder."""
        if not self.acik_mi:
            raise RuntimeError(f"HATA: Kapatilmis donanim kaynagina ({self.kaynak_id}) erisim denendi!")
        self.islem_sayisi += 1
        return True

    def donanim_kapat(self):
        """Fiziksel donanim kapama komutunu cagirir."""
        if self.acik_mi:
            self.acik_mi = False
            self.kapanis_zamani_ns = time.time_ns()


class TeslaCANSoketRAII:
    """
    C++20 RAII prensibine uygun Tesla CAN-FD Soket Yoneticisi.
    Kapsam (scope) sonlandiginda soket file descriptor'ini otomatik ve guvenli kapatir.
    """
    def __init__(self, arayuz_adi: str = "can0", ozel_silici: Optional[Callable[[TeslaDonanimKaynagi], None]] = None):
        self.arayuz_adi = arayuz_adi
        self.kaynak = TeslaDonanimKaynagi(
            kaynak_id=f"SOCKET_{arayuz_adi}",
            tip=DonanimKaynakTipi.CAN_SOKET,
            aciklayici_no=hash(arayuz_adi) % 10000
        )
        self.ozel_silici = ozel_silici or (lambda k: k.donanim_kapat())
        self.sahip_mi = True

    def telemetri_yaz(self, can_id: int, veri: bytes) -> bool:
        if not self.sahip_mi or not self.kaynak.acik_mi:
            raise RuntimeError("HATA: Gecersiz veya serbest birakilmis CAN soketine erisim!")
        return self.kaynak.veri_gonder(veri)

    def kapat(self):
        """Kaynagi guvenli bir sekilde serbest birakir (Idempotent)."""
        if self.sahip_mi and self.kaynak.acik_mi:
            self.ozel_silici(self.kaynak)
            self.sahip_mi = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.kapat()

    def __del__(self):
        # Destructor seviyesinde son guvenlik agi (C++ std::unique_ptr davranisi)
        self.kapat()


class OzelSiliciAkilliIsaretci:
    """
    C++ std::unique_ptr<T, Deleter> mimarisini simule eden akilli isaretci.
    Kullanici tanimli ozel silici (Custom Deleter) ile donanim kaynaklarini deterministik yok eder.
    """
    def __init__(self, kaynak: TeslaDonanimKaynagi, silici_fonksiyon: Optional[Callable[[TeslaDonanimKaynagi], None]] = None):
        self._kaynak: Optional[TeslaDonanimKaynagi] = kaynak
        self._silici = silici_fonksiyon or (lambda k: k.donanim_kapat())
        self._tasiyici_gecerli = True

    def al(self) -> TeslaDonanimKaynagi:
        """Icerideki ham donanim isaretcisini dondurur."""
        if not self._tasiyici_gecerli or self._kaynak is None:
            raise RuntimeError("HATA: Null veya tasinmis (moved-from) akilli isaretciye erisim!")
        return self._kaynak

    def serbest_birak_ve_yok_et(self):
        """Kaynagi ozel silici ile guvenle sonlandirir."""
        if self._tasiyici_gecerli and self._kaynak is not None:
            self._silici(self._kaynak)
            self._kaynak = None
            self._tasiyici_gecerli = False

    def tasi(self) -> 'OzelSiliciAkilliIsaretci':
        """C++ std::move semantigini gercekler; sahiplik yeni isaretciye aktarilir."""
        if not self._tasiyici_gecerli or self._kaynak is None:
            raise RuntimeError("HATA: Gecersiz isaretci tasinamaz!")
        yeni_isaretci = OzelSiliciAkilliIsaretci(self._kaynak, self._silici)
        self._kaynak = None
        self._tasiyici_gecerli = False
        return yeni_isaretci

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.serbest_birak_ve_yok_et()

    def __del__(self):
        self.serbest_birak_ve_yok_et()


class TeslaKaynakIzlemeMerkezi:
    """
    Arac genelindeki tum donanim aciklayicilarini (handles) ve sizintilari takip eden merkez.
    """
    def __init__(self):
        self.kayitli_kaynaklar: Dict[str, TeslaDonanimKaynagi] = {}

    def kaydet(self, kaynak: TeslaDonanimKaynagi):
        self.kayitli_kaynaklar[kaynak.kaynak_id] = kaynak

    def aktif_acik_kaynak_sayisi(self) -> int:
        return sum(1 for k in self.kayitli_kaynaklar.values() if k.acik_mi)

    def sizinti_orani(self) -> float:
        toplam = len(self.kayitli_kaynaklar)
        if toplam == 0:
            return 0.0
        return self.aktif_acik_kaynak_sayisi() / toplam
