"""
Kapalı Çevrim Dokunsal Geri Bildirim Kontrolü (Tactile Closed-Loop) Motoru (Day 254).
Mikro Kayma Tespiti, Değişken Empedans, Sertlik Kestirimi ve Kırılgan Nesne Tutuşu.
"""

from typing import Dict, Any, List, Tuple
import numpy as np


class TactileSlipDetector:
    """1000 Hz Yüksek Frekanslı Mikro Titreşim ve Sürtünme Konisi Kayma Dedektörü."""

    @classmethod
    def detect_slip(
        cls,
        f_normal: float,
        f_tangential: float,
        vib_signal: np.ndarray = None,
        mu_s: float = 0.55,
    ) -> Dict[str, Any]:
        """Sürtünme konisi sınır oranını ve mikro titreşim enerjisini denetler."""
        f_n = max(float(f_normal), 0.05)
        f_t = abs(float(f_tangential))

        # Sürtünme Oranı: eta = |F_t| / F_n
        eta = f_t / f_n
        friction_margin = (mu_s * f_n) - f_t

        # Mikro Titreşim Enerjisi (50-400 Hz FFT Enerjisi Simülasyonu)
        if vib_signal is not None and len(vib_signal) > 0:
            vib_energy = float(np.mean(vib_signal ** 2))
        else:
            vib_energy = 0.0

        # Mikro Kayma Eşiği (%85 sürtünme konisi sınırı veya yüksek titreşim)
        is_imminent_slip = (eta >= 0.85 * mu_s) or (vib_energy > 0.08)

        return {
            "f_normal": round(f_n, 3),
            "f_tangential": round(f_t, 3),
            "surtunme_orani_eta": round(eta, 3),
            "surtunme_marjini_N": round(friction_margin, 3),
            "titresim_enerjisi": round(vib_energy, 4),
            "kayma_tehlikesi_var_mi": bool(is_imminent_slip),
        }


class AdaptiveStiffnessEstimator:
    """Nesne Sertlik (Stiffness) Kestiricisi ve Güvenlik Tavanı Belirleyicisi."""

    @classmethod
    def estimate_stiffness(
        cls,
        delta_f_N: float,
        delta_x_mm: float,
    ) -> Dict[str, Any]:
        """k = Delta F / Delta x oranından nesne esneklik sınıfını ve emniyet tavanını bulur."""
        dx = max(abs(float(delta_x_mm)), 0.01)
        df = max(abs(float(delta_f_N)), 0.01)

        k_est = df / dx  # N/mm

        if k_est < 0.8:
            sinif = "KIRILGAN_YUMUSAK (Yumurta / Çilek)"
            f_safe_max = 3.5  # Max 3.5N
        elif k_est < 3.0:
            sinif = "YARI_ESNEK (Plastik Bardak / Sünger)"
            f_safe_max = 8.0  # Max 8.0N
        else:
            sinif = "RİJİT_SERT (Metal / Ahşap Kutu)"
            f_safe_max = 25.0 # Max 25.0N

        return {
            "sertlik_k_N_mm": round(k_est, 3),
            "nesne_sinifi": sinif,
            "maksimum_guvenli_kuvvet_N": f_safe_max,
        }


class ClosedLoopTactileController:
    """Kapalı Çevrim Değişken Empedanslı Dokunsal Tutuş Kontrolcüsü."""

    def __init__(self, mu_s: float = 0.55):
        self.mu_s = mu_s
        self.durum = "YAKLASMA"  # APPROACH
        self.f_normal = 0.2
        self.gripper_acikligi_mm = 80.0

    def step_control(
        self,
        f_tangential: float,
        delta_x_gripper_mm: float,
        delta_f_sensor_N: float,
        vib_signal: np.ndarray = None,
    ) -> Dict[str, Any]:
        """1 Adımlık Kapalı Çevrim Geri Bildirimli Kontrol Döngüsü."""
        # 1. Sertlik Kestirimi
        stiffness_info = AdaptiveStiffnessEstimator.estimate_stiffness(
            delta_f_N=delta_f_sensor_N,
            delta_x_mm=delta_x_gripper_mm,
        )
        f_max = stiffness_info["maksimum_guvenli_kuvvet_N"]

        # 2. Kayma Tespiti
        slip_info = TactileSlipDetector.detect_slip(
            f_normal=self.f_normal,
            f_tangential=f_tangential,
            vib_signal=vib_signal,
            mu_s=self.mu_s,
        )

        # 3. Adaptif Kuvvet Düzenlemesi (Adaptive Force Regulation)
        if slip_info["kayma_tehlikesi_var_mi"]:
            self.durum = "KAYMA_KOMPANZASYONU"
            # Kaymayı durdurmak için normal kuvveti kontrollü artır (Emniyet tavanını aşmadan)
            self.f_normal = min(self.f_normal + 0.45, f_max)
        else:
            self.durum = "KARARLI_TUTUS"
            # Minimum enerji ve sıfır ezme için kuvveti optimize seviyede tut
            if self.f_normal > 1.2:
                self.f_normal -= 0.05

        return {
            "durum": self.durum,
            "uygulanan_f_normal_N": round(self.f_normal, 3),
            "emniyet_tavani_N": f_max,
            "ezilme_riski_var_mi": self.f_normal > f_max,
            "kayma_bilgisi": slip_info,
            "sertlik_bilgisi": stiffness_info,
        }
