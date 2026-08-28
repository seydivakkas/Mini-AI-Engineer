"""
Day 91: İstatistiksel Veri ve Tahmin Kayması (Data & Prediction Drift) Dedektörü
-------------------------------------------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats


@dataclass
class OznitelikDriftDetayi:
    oznitelik_adi: str
    ks_istatistigi: float
    ks_p_degeri: float
    psi_degeri: float
    wasserstein_mesafesi: float
    drift_var_mi: bool
    alarm_seviyesi: str  # "NORMAL", "UYARI", "KRITIK"


@dataclass
class DriftRaporu:
    toplam_oznitelik: int
    kayan_oznitelik_sayisi: int
    sistem_drift_orani: float
    genel_durum: str  # "SAGLIKLI", "DIKKAT", "KRITIK_KAYMA"
    oznitelik_detaylari: Dict[str, OznitelikDriftDetayi]
    tahmin_kaymasi_var_mi: bool
    tahmin_psi: float


class DriftDedektoru:
    """
    Referans (baseline/eğitim) dağılımı ile canlı (production) veri akışı
    arasındaki istatistiksel kaymaları çok boyutlu metriklerle tespit eden motor.
    """

    def __init__(
        self,
        ks_alfa_esigi: float = 0.05,
        psi_uyari_esigi: float = 0.10,
        psi_kritik_esik: float = 0.20,
        kutu_sayisi: int = 10,
    ):
        self.ks_alfa_esigi = ks_alfa_esigi
        self.psi_uyari_esigi = psi_uyari_esigi
        self.psi_kritik_esik = psi_kritik_esik
        self.kutu_sayisi = kutu_sayisi

        self._referans_oznitelikler: Optional[np.ndarray] = None
        self._referans_tahminler: Optional[np.ndarray] = None

    def referans_belirle(self, referans_oznitelikler: np.ndarray, referans_tahminler: Optional[np.ndarray] = None) -> None:
        """Eğitim/doğrulama setinden elde edilen temel (baseline) dağılımı kaydeder."""
        if referans_oznitelikler.ndim == 1:
            referans_oznitelikler = referans_oznitelikler.reshape(-1, 1)
        self._referans_oznitelikler = np.copy(referans_oznitelikler)
        if referans_tahminler is not None:
            self._referans_tahminler = np.copy(referans_tahminler)

    @staticmethod
    def hesapla_psi(referans: np.ndarray, canli: np.ndarray, kutu_sayisi: int = 10, eps: float = 1e-6) -> float:
        """
        Population Stability Index (PSI) formülünü hesaplar:
        PSI = sum((P_k - Q_k) * ln(P_k / Q_k))
        """
        if len(referans) == 0 or len(canli) == 0:
            return 0.0

        # Ortak kutu sınırları (Quantile / Eşit Frekanslı Aralıklar)
        yuzdelikler = np.linspace(0, 100, kutu_sayisi + 1)
        kutu_sinirlari = np.unique(np.percentile(referans, yuzdelikler))
        if len(kutu_sinirlari) < 2:
            min_val = float(min(np.min(referans), np.min(canli)))
            max_val = float(max(np.max(referans), np.max(canli)))
            if min_val == max_val:
                min_val -= 1.0
                max_val += 1.0
            kutu_sinirlari = np.linspace(min_val, max_val, kutu_sayisi + 1)
        kutu_sinirlari[0] = -np.inf
        kutu_sinirlari[-1] = np.inf

        ref_sayim, _ = np.histogram(referans, bins=kutu_sinirlari)
        canli_sayim, _ = np.histogram(canli, bins=kutu_sinirlari)

        # Olasılık oranları (P ve Q)
        p_oran = (ref_sayim / len(referans)) + eps
        q_oran = (canli_sayim / len(canli)) + eps

        psi = np.sum((q_oran - p_oran) * np.log(q_oran / p_oran))
        return float(max(0.0, psi))

    def analiz_et(
        self,
        canli_oznitelikler: np.ndarray,
        canli_tahminler: Optional[np.ndarray] = None,
        oznitelik_isimleri: Optional[List[str]] = None,
    ) -> DriftRaporu:
        """
        Canlı veriyi referans dağılımla karşılaştırıp detaylı kayma raporu üretir.
        """
        if self._referans_oznitelikler is None:
            raise ValueError("Referans veri seti belirlenmeden analiz yapılamaz. Önce referans_belirle() çağırın.")

        if canli_oznitelikler.ndim == 1:
            canli_oznitelikler = canli_oznitelikler.reshape(-1, 1)

        num_oznitelik = min(self._referans_oznitelikler.shape[1], canli_oznitelikler.shape[1])
        if oznitelik_isimleri is None:
            oznitelik_isimleri = [f"oznitelik_{i}" for i in range(num_oznitelik)]

        detaylar: Dict[str, OznitelikDriftDetayi] = {}
        kayan_sayisi = 0

        for idx in range(num_oznitelik):
            isim = oznitelik_isimleri[idx]
            ref_kolon = self._referans_oznitelikler[:, idx]
            canli_kolon = canli_oznitelikler[:, idx]

            # 1. Kolmogorov-Smirnov Testi
            ks_sonuc = stats.ks_2samp(ref_kolon, canli_kolon)
            ks_ist = float(ks_sonuc.statistic)
            ks_p = float(ks_sonuc.pvalue)

            # 2. Population Stability Index (PSI)
            psi = self.hesapla_psi(ref_kolon, canli_kolon, kutu_sayisi=self.kutu_sayisi)

            # 3. Wasserstein Distance (Earth Mover's Distance)
            w_mesafe = float(stats.wasserstein_distance(ref_kolon, canli_kolon))

            # Drift ve Alarm Kararı
            drift_var = (ks_p < self.ks_alfa_esigi) or (psi >= self.psi_uyari_esigi)
            if drift_var:
                kayan_sayisi += 1

            if psi >= self.psi_kritik_esik:
                seviye = "KRITIK"
            elif psi >= self.psi_uyari_esigi or ks_p < self.ks_alfa_esigi:
                seviye = "UYARI"
            else:
                seviye = "NORMAL"

            detaylar[isim] = OznitelikDriftDetayi(
                oznitelik_adi=isim,
                ks_istatistigi=ks_ist,
                ks_p_degeri=ks_p,
                psi_degeri=psi,
                wasserstein_mesafesi=w_mesafe,
                drift_var_mi=drift_var,
                alarm_seviyesi=seviye,
            )

        drift_orani = float(kayan_sayisi / num_oznitelik) if num_oznitelik > 0 else 0.0

        if drift_orani >= 0.5:
            genel_durum = "KRITIK_KAYMA"
        elif drift_orani > 0.0:
            genel_durum = "DIKKAT"
        else:
            genel_durum = "SAGLIKLI"

        # Tahmin (Prediction) Drift Analizi
        tahmin_drift = False
        tahmin_psi = 0.0
        if self._referans_tahminler is not None and canli_tahminler is not None:
            tahmin_psi = self.hesapla_psi(self._referans_tahminler, canli_tahminler, kutu_sayisi=self.kutu_sayisi)
            tahmin_drift = tahmin_psi >= self.psi_uyari_esigi

        return DriftRaporu(
            toplam_oznitelik=num_oznitelik,
            kayan_oznitelik_sayisi=kayan_sayisi,
            sistem_drift_orani=drift_orani,
            genel_durum=genel_durum,
            oznitelik_detaylari=detaylar,
            tahmin_kaymasi_var_mi=tahmin_drift,
            tahmin_psi=tahmin_psi,
        )
