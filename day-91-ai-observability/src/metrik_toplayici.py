"""
Day 91: Prometheus / OpenTelemetry Uyumlu Metrik Toplayıcı ve İstatistik Motoru
-------------------------------------------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import threading
import time
import numpy as np


@dataclass
class MetrikOzeti:
    toplam_istek: int
    toplam_hata: int
    hata_orani: float
    ortalama_gecikme_ms: float
    p50_gecikme_ms: float
    p95_gecikme_ms: float
    p99_gecikme_ms: float
    maks_gecikme_ms: float
    anlik_rps: float
    sla_ihlal_sayisi: int
    sla_ihlal_orani: float


class MetrikToplayici:
    """
    Canlı AI çıkarım servislerinde Prometheus / OpenTelemetry standartlarında
    metrik toplayan, thread-safe, zaman serisi histogram ve sayaç modülü.
    """

    def __init__(self, sla_gecikme_esigi_ms: float = 25.0, pencere_boyutu_sn: float = 10.0):
        self.sla_gecikme_esigi_ms = float(sla_gecikme_esigi_ms)
        self.pencere_boyutu_sn = float(pencere_boyutu_sn)

        self._kilit = threading.RLock()
        self._toplam_istek: int = 0
        self._toplam_hata: int = 0
        self._sla_ihlal_sayisi: int = 0

        # Zaman serisi kayıtları
        self._zaman_damgalari: List[float] = []
        self._gecikmeler_ms: List[float] = []
        self._hata_kayitlari: List[bool] = []

    def kayit_ekle(self, gecikme_ms: float, hata_olustu: bool = False, zaman_damgasi: Optional[float] = None) -> None:
        """Tekil bir çıkarım isteğinin gecikme ve başarı durumunu kaydeder."""
        if zaman_damgasi is None:
            zaman_damgasi = time.time()

        with self._kilit:
            self._toplam_istek += 1
            if hata_olustu:
                self._toplam_hata += 1
            if gecikme_ms > self.sla_gecikme_esigi_ms:
                self._sla_ihlal_sayisi += 1

            self._zaman_damgalari.append(zaman_damgasi)
            self._gecikmeler_ms.append(float(gecikme_ms))
            self._hata_kayitlari.append(bool(hata_olustu))

    def ozet_rapor_uret(self, son_n_istek: Optional[int] = None) -> MetrikOzeti:
        """Toplanan metriklerin istatistiksel özetini ve yüzdeliklerini döner."""
        with self._kilit:
            if not self._gecikmeler_ms:
                return MetrikOzeti(
                    toplam_istek=0,
                    toplam_hata=0,
                    hata_orani=0.0,
                    ortalama_gecikme_ms=0.0,
                    p50_gecikme_ms=0.0,
                    p95_gecikme_ms=0.0,
                    p99_gecikme_ms=0.0,
                    maks_gecikme_ms=0.0,
                    anlik_rps=0.0,
                    sla_ihlal_sayisi=0,
                    sla_ihlal_orani=0.0,
                )

            gecikmeler = np.array(self._gecikmeler_ms)
            if son_n_istek is not None and son_n_istek > 0:
                gecikmeler = gecikmeler[-son_n_istek:]

            toplam_ist = len(gecikmeler)
            p50 = float(np.percentile(gecikmeler, 50))
            p95 = float(np.percentile(gecikmeler, 95))
            p99 = float(np.percentile(gecikmeler, 99))
            ort = float(np.mean(gecikmeler))
            maks = float(np.max(gecikmeler))

            ihlal_sayisi = int(np.sum(gecikmeler > self.sla_gecikme_esigi_ms))
            ihlal_orani = float(ihlal_sayisi / toplam_ist) if toplam_ist > 0 else 0.0

            hata_sayisi = int(np.sum(self._hata_kayitlari[-toplam_ist:]))
            hata_orani = float(hata_sayisi / toplam_ist) if toplam_ist > 0 else 0.0

            # RPS hesabı
            rps = self.anlik_rps_hesapla()

            return MetrikOzeti(
                toplam_istek=self._toplam_istek,
                toplam_hata=self._toplam_hata,
                hata_orani=hata_orani,
                ortalama_gecikme_ms=ort,
                p50_gecikme_ms=p50,
                p95_gecikme_ms=p95,
                p99_gecikme_ms=p99,
                maks_gecikme_ms=maks,
                anlik_rps=rps,
                sla_ihlal_sayisi=self._sla_ihlal_sayisi,
                sla_ihlal_orani=ihlal_orani,
            )

    def anlik_rps_hesapla(self) -> float:
        """Son kayan pencere süresi içerisindeki saniyelik istek sayısını (RPS) hesaplar."""
        simdi = time.time()
        with self._kilit:
            if not self._zaman_damgalari:
                return 0.0
            pencere_baslangic = simdi - self.pencere_boyutu_sn
            penceredeki_istekler = sum(1 for t in self._zaman_damgalari if t >= pencere_baslangic)
            gecen_sure = max(1.0, min(self.pencere_boyutu_sn, simdi - self._zaman_damgalari[0]))
            return float(penceredeki_istekler / gecen_sure)

    def zaman_serisi_verisi_al(self) -> Dict[str, np.ndarray]:
        """Grafik çizimleri için ham zaman serisi dizilerini döner."""
        with self._kilit:
            return {
                "zaman_damgalari": np.array(self._zaman_damgalari),
                "gecikmeler_ms": np.array(self._gecikmeler_ms),
                "hatalar": np.array(self._hata_kayitlari),
            }

    def sifirla(self) -> None:
        """Tüm metrikleri ve kayıtları sıfırlar."""
        with self._kilit:
            self._toplam_istek = 0
            self._toplam_hata = 0
            self._sla_ihlal_sayisi = 0
            self._zaman_damgalari.clear()
            self._gecikmeler_ms.clear()
            self._hata_kayitlari.clear()
