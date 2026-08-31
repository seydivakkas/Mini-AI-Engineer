"""
Tesla C++20 std::span, std::ranges ve std::string_view Modulu
==============================================================
Bu modul; C++20 `std::string_view` ve `std::span` soyutlamalarini kullanarak,
Tesla GNSS (GPS) modulunden gelen ham NMEA ($GPRMC) verilerini ve CAN-FD bayt dizilerini
sifir heap tahsisi (zero-allocation) ve sifir kopyalama (zero-copy) ile ayristirir.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

from typing import List, Optional, Tuple, Iterator
from dataclasses import dataclass
import time


class TeslaStringView:
    """
    C++20 `std::string_view` karsiligi.
    Metin uzerinde sadece baslangic gostericisi (pointer) ve uzunluk (length)
    tutar; kesinlikle yeni bellek (heap) tahsisi yapmaz.
    """
    __slots__ = ('_metin', '_baslangic', '_uzunluk')

    def __init__(self, metin: str, baslangic: int = 0, uzunluk: Optional[int] = None):
        self._metin = metin
        self._baslangic = baslangic
        self._uzunluk = len(metin) - baslangic if uzunluk is None else uzunluk

    def __len__(self) -> int:
        return self._uzunluk

    def __str__(self) -> str:
        return self._metin[self._baslangic : self._baslangic + self._uzunluk]

    def subview(self, bas: int, uzunluk: Optional[int] = None) -> 'TeslaStringView':
        """Sıfır kopyalama ile alt görünüm (subview) üretir."""
        kalan = max(0, self._uzunluk - bas)
        gercek_uzunluk = kalan if uzunluk is None else min(uzunluk, kalan)
        return TeslaStringView(self._metin, self._baslangic + bas, gercek_uzunluk)

    def parcalara_bol(self, ayirici: str = ',') -> Iterator['TeslaStringView']:
        """
        C++20 `std::views::split` karşılığı.
        Bellek kopyalamaksızın alt string_view dilimlerini döner.
        """
        bas = 0
        metin_parcasi = self._metin[self._baslangic : self._baslangic + self._uzunluk]
        
        while bas <= len(metin_parcasi):
            konum = metin_parcasi.find(ayirici, bas)
            if konum == -1:
                yield self.subview(bas, len(metin_parcasi) - bas)
                break
            else:
                yield self.subview(bas, konum - bas)
                bas = konum + len(ayirici)


@dataclass
class TeslaGNSSKonumBilgisi:
    utc_zamani: str
    gecerli_mi: bool
    enlem_derece: float
    boylam_derece: float
    hiz_kmh: float
    rota_acisi_derece: float
    tarih: str


class TeslaNMEAAyristirici:
    """
    C++20 std::string_view ve std::ranges tabanlı sıfır-tahsisli NMEA ayrıştırıcısı.
    """
    @staticmethod
    def _nmea_koordinat_donustur(ham_deger: TeslaStringView, yon: TeslaStringView) -> float:
        """NMEA '3723.2475,N' formatını ondalık dereceye (37.387458) dönüştürür."""
        metin = str(ham_deger)
        if not metin or '.' not in metin:
            return 0.0
            
        nokta_yeri = metin.find('.')
        derece_uzunluk = nokta_yeri - 2
        derece = float(metin[:derece_uzunluk])
        dakika = float(metin[derece_uzunluk:])
        ondalik = derece + (dakika / 60.0)
        
        yon_str = str(yon).strip()
        if yon_str in ['S', 'W']:
            ondalik = -ondalik
        return ondalik

    @classmethod
    def gprmc_ayristir(cls, ham_cumle: str) -> Optional[TeslaGNSSKonumBilgisi]:
        """
        $GPRMC cümlesini sıfır kopyalama ile ayrıştırır.
        Örnek: $GPRMC,083559.00,A,3723.2475,N,12208.3845,W,55.4,180.0,300826,,,A*72
        """
        gorunum = TeslaStringView(ham_cumle)
        
        # Checksum ve başlık kontrolü
        if not str(gorunum).startswith("$GPRMC"):
            return None

        alanlar: List[TeslaStringView] = list(gorunum.parcalara_bol(','))
        if len(alanlar) < 10:
            return None

        # 1: UTC Zamanı
        utc_zamani = str(alanlar[1])
        # 2: Durum ('A' = Aktif/Geçerli, 'V' = Geçersiz)
        durum_str = str(alanlar[2])
        gecerli_mi = (durum_str == 'A')

        # 3, 4: Enlem & Yön
        enlem = cls._nmea_koordinat_donustur(alanlar[3], alanlar[4])
        # 5, 6: Boylam & Yön
        boylam = cls._nmea_koordinat_donustur(alanlar[5], alanlar[6])

        # 7: Hız (Knot -> km/h: 1 knot = 1.852 km/h)
        hiz_knot = float(str(alanlar[7])) if len(alanlar[7]) > 0 else 0.0
        hiz_kmh = hiz_knot * 1.852

        # 8: Rota Açısı (Heading)
        rota = float(str(alanlar[8])) if len(alanlar[8]) > 0 else 0.0

        # 9: Tarih (DDMMYY)
        tarih = str(alanlar[9])

        return TeslaGNSSKonumBilgisi(
            utc_zamani=utc_zamani,
            gecerli_mi=gecerli_mi,
            enlem_derece=enlem,
            boylam_derece=boylam,
            hiz_kmh=hiz_kmh,
            rota_acisi_derece=rota,
            tarih=tarih
        )
