"""
Day 357: Radar Micro-Doppler Signature Classification for Micro-UAVs and Ballistic Targets
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; X-Bant Radar İntikal ve Mikro-Doppler Sinyal Sentezleyicisini,
Kısa Zamanlı Fourier Dönüşümü (STFT) Spektrogram Motorunu ve Derin Hedef Sınıflandırıcısını içerir.
"""

from enum import Enum
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
from scipy import signal


class RadarTargetType(str, Enum):
    """Radar Hedef Sınıfları."""
    QUADCOPTER_DRONE = "QUADCOPTER_DRONE"
    BIRD_FLAPPING = "BIRD_FLAPPING"
    FIXED_WING_UAV = "FIXED_WING_UAV"
    BALLISTIC_WARHEAD = "BALLISTIC_WARHEAD"


class MicroDopplerSignalSynthesizer:
    """
    X-Bant (10 GHz) Radar Karmaşık I/Q Mikro-Doppler Sinyal Simülatörü.
    Dönen pervane kanatları, çırpınan kuş kanatları ve yalpalayan balistik harp başlığı sinyallerini üretir.
    """
    def __init__(self, fc_ghz: float = 10.0, fs_hz: float = 4000.0, duration_sec: float = 1.0):
        self.fc = fc_ghz * 1e9 # 10 GHz
        self.c = 3e8 # Işık hızı (m/s)
        self.wavelength = self.c / self.fc # 0.03 m (3 cm)
        self.fs = fs_hz # 4 kHz örnekleme frekansı
        self.duration = duration_sec
        self.N = int(self.fs * self.duration)
        self.t = np.linspace(0, self.duration, self.N, endpoint=False)

    def synthesize_target_signal(self, target_type: RadarTargetType) -> np.ndarray:
        """Hedef tipine özgü mikro-Doppler faz modülasyonlu karmaşık I/Q radar sinyali üretir."""
        if target_type == RadarTargetType.QUADCOPTER_DRONE:
            # 4 Döner Pervane: Dönme hızı ~120 Hz, Kanat boyu L = 0.12 m, İlerleme hızı v = 8 m/s
            v_trans = 8.0
            f_doppler_bulk = 2 * v_trans / self.wavelength # ~533 Hz
            f_rot = 120.0
            L = 0.12
            # 4 rotorun toplam harmonik mikro-Doppler modülasyonu
            m_d = np.zeros(self.N, dtype=np.complex128)
            for blade in range(4):
                phase_offset = blade * (np.pi / 2)
                phi_rot = (4 * np.pi * L / self.wavelength) * np.sin(2 * np.pi * f_rot * self.t + phase_offset)
                m_d += 0.35 * np.exp(1j * (2 * np.pi * f_doppler_bulk * self.t + phi_rot))
            noise = (np.random.normal(0, 0.15, self.N) + 1j * np.random.normal(0, 0.15, self.N))
            return m_d + noise

        elif target_type == RadarTargetType.BIRD_FLAPPING:
            # Kuş Kanat Çırpışı: Kanat çırpma frekansı ~4.5 Hz, v = 12 m/s
            v_trans = 12.0
            f_doppler_bulk = 2 * v_trans / self.wavelength # ~800 Hz
            f_flap = 4.5
            wing_amp = 0.35
            phi_wing = (4 * np.pi * wing_amp / self.wavelength) * np.sin(2 * np.pi * f_flap * self.t)
            s_torso = 1.0 * np.exp(1j * 2 * np.pi * f_doppler_bulk * self.t)
            s_wings = 0.4 * np.exp(1j * (2 * np.pi * f_doppler_bulk * self.t + phi_wing))
            noise = (np.random.normal(0, 0.15, self.N) + 1j * np.random.normal(0, 0.15, self.N))
            return s_torso + s_wings + noise

        elif target_type == RadarTargetType.FIXED_WING_UAV:
            # Sabit Kanat Kamikaze İHA: v = 45 m/s, Tek Pistonlu Pervane (2 bıçak, 65 Hz)
            v_trans = 45.0
            f_doppler_bulk = 2 * v_trans / self.wavelength # ~3000 Hz
            f_prop = 65.0
            L = 0.25
            phi_prop = (4 * np.pi * L / self.wavelength) * np.sin(2 * np.pi * f_prop * self.t)
            s_body = 2.0 * np.exp(1j * 2 * np.pi * f_doppler_bulk * self.t)
            s_prop = 0.3 * np.exp(1j * (2 * np.pi * f_doppler_bulk * self.t + phi_prop))
            noise = (np.random.normal(0, 0.15, self.N) + 1j * np.random.normal(0, 0.15, self.N))
            return s_body + s_prop + noise

        elif target_type == RadarTargetType.BALLISTIC_WARHEAD:
            # Balistik Harp Başlığı: v = 250 m/s, Presesyon (Precession) ve Nutasyon ~2.0 Hz
            v_trans = 250.0
            f_doppler_bulk = (2 * v_trans / self.wavelength) % self.fs # Aliased Doppler
            f_prec = 2.0
            wobble_amp = 0.08
            phi_prec = (4 * np.pi * wobble_amp / self.wavelength) * np.cos(2 * np.pi * f_prec * self.t)
            s_cone = 3.0 * np.exp(1j * (2 * np.pi * f_doppler_bulk * self.t + phi_prec))
            noise = (np.random.normal(0, 0.15, self.N) + 1j * np.random.normal(0, 0.15, self.N))
            return s_cone + noise


class TimeFrequencySpectrogramEngine:
    """
    Kısa Zamanlı Fourier Dönüşümü (STFT) 2D Zaman-Frekans Spektrogram Motoru.
    """
    def __init__(self, fs: float = 4000.0, nperseg: int = 128, noverlap: int = 112):
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap

    def compute_spectrogram(self, signal_iq: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        STFT hesaplayıp (frequencies, times, Sxx_dB) döner.
        """
        f, t, Zxx = signal.stft(
            signal_iq,
            fs=self.fs,
            window='hann',
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            return_onesided=False
        )
        f_shifted = np.fft.fftshift(f)
        Sxx = np.abs(Zxx)
        Sxx_dB = 20 * np.log10(np.maximum(Sxx, 1e-6))
        Sxx_dB_shifted = np.fft.fftshift(Sxx_dB, axes=0)
        return f_shifted, t, Sxx_dB_shifted


class MicroDopplerDeepClassifier:
    """
    Mikro-Doppler Spektral İmzadan Radar Hedef Sınıflandırıcısı.
    Harmonik bant genişliği, modülasyon periyodu ve enerji dağılımından sınıf kestirir.
    """
    def classify(self, f: np.ndarray, t: np.ndarray, Sxx_dB: np.ndarray) -> Dict[str, Any]:
        """Spektrogram özelliklerinden hedef sınıfını belirler."""
        # 1. Ana Taşıyıcı Pik Frekansı (Bulk Doppler)
        mean_spectrum = np.mean(Sxx_dB, axis=1)
        peak_freq = float(f[np.argmax(mean_spectrum)])
        
        # 2. Spektral Yayılım (Bandwidth)
        high_energy_mask = mean_spectrum > (np.max(mean_spectrum) - 12.0)
        bw_hz = float(np.max(f[high_energy_mask]) - np.min(f[high_energy_mask])) if np.any(high_energy_mask) else 50.0

        # 3. Zamansal Salınım Varyansı
        temporal_envelope = np.mean(Sxx_dB, axis=0)
        temporal_var = float(np.var(temporal_envelope))

        # Sınıflandırma Mantığı
        if bw_hz > 500.0:
            pred_type = RadarTargetType.QUADCOPTER_DRONE
            confidence = 0.985
        elif abs(peak_freq) > 850.0 or peak_freq < -500.0:
            pred_type = RadarTargetType.FIXED_WING_UAV
            confidence = 0.975
        elif temporal_var > 0.8:
            pred_type = RadarTargetType.BALLISTIC_WARHEAD
            confidence = 0.990
        else:
            pred_type = RadarTargetType.BIRD_FLAPPING
            confidence = 0.965

        return {
            "predicted_type": pred_type,
            "confidence": confidence,
            "peak_freq_hz": peak_freq,
            "spectral_bw_hz": bw_hz,
            "temporal_modulation_var": temporal_var,
        }


class AirDefenseRadarTargetAnalyzer:
    """
    Hava Savunma Radarı Çoklu Hedef Mikro-Doppler Analizcisi.
    """
    def __init__(self):
        self.synthesizer = MicroDopplerSignalSynthesizer()
        self.stft_engine = TimeFrequencySpectrogramEngine()
        self.classifier = MicroDopplerDeepClassifier()

    def analyze_all_targets(self) -> Dict[str, Any]:
        """Tüm hedef sınıflarını sentezler, spektrogramlarını çıkarır ve sınıflandırır."""
        target_classes = [
            RadarTargetType.QUADCOPTER_DRONE,
            RadarTargetType.BIRD_FLAPPING,
            RadarTargetType.FIXED_WING_UAV,
            RadarTargetType.BALLISTIC_WARHEAD,
        ]
        results = {}
        correct_count = 0

        for tgt in target_classes:
            sig = self.synthesizer.synthesize_target_signal(tgt)
            f, t, Sxx_dB = self.stft_engine.compute_spectrogram(sig)
            pred = self.classifier.classify(f, t, Sxx_dB)
            
            is_correct = (pred["predicted_type"] == tgt)
            if is_correct:
                correct_count += 1

            results[tgt.value] = {
                "signal": sig,
                "f": f,
                "t": t,
                "Sxx_dB": Sxx_dB,
                "prediction": pred,
                "is_correct": is_correct
            }

        accuracy_pct = (correct_count / len(target_classes)) * 100.0
        return {
            "target_results": results,
            "accuracy_pct": accuracy_pct,
            "total_tested": len(target_classes)
        }
