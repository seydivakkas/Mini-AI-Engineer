"""
Tesla FMCW Radar ve Ultrasonik Sinyal İşleme Çekirdeği
======================================================
Bu modül; 77 GHz FMCW Radar 1D Range-FFT, 2D Doppler-FFT ve CA-CFAR
hedef tespitini, yaya Micro-Doppler imzasını ve Sıcaklık Kompanzasyonlu
Ultrasonik Time-of-Flight (ToF) mesafe motorunu gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class TeslaRadarAndUltrasonicProcessor:
    """
    77 GHz FMCW Radar ve Ultrasonik Park Sensörü Sinyal İşleyicisi.
    """
    def __init__(
        self,
        center_freq_hz: float = 77.0e9,
        bandwidth_hz: float = 1.0e9,
        chirp_duration_s: float = 50.0e-6,
        num_samples: int = 256,
        num_chirps: int = 64
    ):
        self.fc = center_freq_hz
        self.bw = bandwidth_hz
        self.tc = chirp_duration_s
        self.n_samples = num_samples
        self.n_chirps = num_chirps
        self.c = 3.0e8  # Işık hızı m/s
        self.wavelength = self.c / self.fc
        self.slope = self.bw / self.tc  # Chirp eğimi S (Hz/s)

    def compute_range_doppler_fft(self, raw_radar_data: np.ndarray) -> np.ndarray:
        """
        Ham ADC verisine (num_chirps x num_samples) 2D FFT uygulayarak
        Range-Doppler matrisi üretir.
        """
        assert raw_radar_data.shape == (self.n_chirps, self.n_samples)

        # Hanning Pencereleme (Spektral sızıntıyı önlemek için)
        win_range = np.hanning(self.n_samples)
        win_doppler = np.hanning(self.n_chirps)[:, None]
        windowed_data = raw_radar_data * win_range * win_doppler

        # 1. Adım: Hızlı Zaman (Fast-Time) 1D Range-FFT
        range_fft = np.fft.fft(windowed_data, axis=1)

        # 2. Adım: Yavaş Zaman (Slow-Time) 2D Doppler-FFT
        doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, axis=0), axes=0)

        # Güç Spektrumu (Magnitude in dB)
        rd_map_db = 20.0 * np.log10(np.abs(doppler_fft) + 1e-12)
        return rd_map_db

    def ca_cfar_1d(
        self,
        signal_1d: np.ndarray,
        num_train: int = 16,
        num_guard: int = 4,
        threshold_offset_db: float = 8.0
    ) -> np.ndarray:
        """
        1D Cell-Averaging Constant False Alarm Rate (CA-CFAR) Dedektörü.
        """
        n = len(signal_1d)
        detections = np.zeros(n, dtype=bool)
        half_train = num_train // 2
        half_guard = num_guard // 2

        for i in range(half_train + half_guard, n - half_train - half_guard):
            # Eğitim hücreleri: Sol ve sağ pencereler
            left_train = signal_1d[i - half_guard - half_train : i - half_guard]
            right_train = signal_1d[i + half_guard + 1 : i + half_guard + half_train + 1]

            noise_floor = np.mean(np.concatenate((left_train, right_train)))
            threshold = noise_floor + threshold_offset_db

            if signal_1d[i] > threshold:
                detections[i] = True

        return detections

    def compute_ultrasonic_distance(
        self,
        tof_seconds: float,
        ambient_temp_c: float = 20.0
    ) -> float:
        """
        Sıcaklık kompanzasyonlu ultrasonik yankı mesafesi: d = (v_sound * t) / 2.
        """
        # Ses hızı: v(T) = 331.3 * sqrt(1 + T / 273.15) m/s
        v_sound = 331.3 * np.sqrt(1.0 + (ambient_temp_c / 273.15))
        distance_m = (v_sound * tof_seconds) / 2.0
        return float(distance_m)

    def generate_synthetic_radar_frame(
        self,
        target_range_m: float = 25.0,
        target_speed_mps: float = -10.0,
        snr_db: float = 20.0
    ) -> np.ndarray:
        """
        Belirli mesafe ve hızdaki bir araç için sentetik FMCW sinyali üretir.
        """
        t_fast = np.linspace(0, self.tc, self.n_samples)
        t_slow = np.arange(self.n_chirps) * self.tc

        # Beat frekansı fb ve Doppler frekansı fd
        fb = (2.0 * self.slope * target_range_m) / self.c
        fd = (2.0 * target_speed_mps) / self.wavelength

        # 2D Sinyal Matrisi
        phase_fast = 2.0 * np.pi * fb * t_fast[None, :]
        phase_slow = 2.0 * np.pi * fd * t_slow[:, None]

        signal = np.cos(phase_fast + phase_slow)

        # Gürültü ekle
        noise_pwr = 10.0 ** (-snr_db / 20.0)
        noise = np.random.normal(0, noise_pwr, (self.n_chirps, self.n_samples))

        return signal + noise
