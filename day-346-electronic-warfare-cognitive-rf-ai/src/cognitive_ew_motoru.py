"""
Day 346: Electronic Warfare (EW) Cognitive RF Spectrum Sensing & Jamming Mitigation
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; I/Q Karmaşık RF Sinyal Simülatörünü, Derin Spektrum Modülasyon Sınıflandırıcısını,
ve Takviyeli Öğrenme Tabanlı Bilişsel Karıştırma Önleme (Anti-Jamming Channel Hopping) Ajanını içerir.
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np


class RFEmitterSimulator:
    """
    RF Elektromanyetik Spektrum Verici ve Karıştırıcı Simülatörü.
    I/Q Karmaşık Taban Bant Sinyalleri Sentezler: QPSK, LFM Radar, FHSS, ve Düşman Karıştırıcı (Jamming).
    """
    SIGNAL_TYPES = ["QPSK_COMM", "LFM_RADAR", "FHSS_TACTICAL", "HOSTILE_JAMMER"]

    def __init__(self, sample_rate: float = 1e6, num_samples: int = 512):
        self.fs = sample_rate
        self.n_samples = num_samples
        self.t = np.arange(self.n_samples) / self.fs

    def generate_signal(self, sig_type: str, snr_db: float = 15.0, fc: float = 100e3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Belirtilen RF sinyal tipinde karmaşık (I + j*Q) zaman serisi ve gürültü üretir.
        """
        if sig_type == "QPSK_COMM":
            # 4-Fazlı QPSK Modülasyonu
            symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], size=self.n_samples // 8)
            iq_base = np.repeat(symbols, 8)
            carrier = np.exp(1j * 2 * np.pi * fc * self.t)
            iq_sig = iq_base * carrier

        elif sig_type == "LFM_RADAR":
            # Lineer Frekans Modülasyonlu (LFM / Chirp) Radar Darbesi
            k_chirp = 50e3 / (self.n_samples / self.fs)
            iq_sig = np.exp(1j * 2 * np.pi * (fc * self.t + 0.5 * k_chirp * (self.t ** 2)))

        elif sig_type == "FHSS_TACTICAL":
            # Frekans Atlamalı Yayılı Spektrum
            hops = [fc - 50e3, fc, fc + 50e3, fc + 100e3]
            f_current = np.repeat(np.random.choice(hops, size=4), self.n_samples // 4)
            phase = np.cumsum(2 * np.pi * f_current / self.fs)
            iq_sig = np.exp(1j * phase)

        elif sig_type == "HOSTILE_JAMMER":
            # Düşman Geniş Bant Gürültü ve Süpürme Karıştırması (Barrage / Sweep Jammer)
            noise_i = np.random.normal(0, 1.0, self.n_samples)
            noise_q = np.random.normal(0, 1.0, self.n_samples)
            sweep_carrier = np.exp(1j * 2 * np.pi * (fc + 200e3 * np.sin(2 * np.pi * 5e3 * self.t)) * self.t)
            iq_sig = (noise_i + 1j * noise_q) * sweep_carrier * 2.5

        else:
            iq_sig = np.zeros(self.n_samples, dtype=complex)

        # AWGN Termal Gürültü Ekle
        sig_power = np.mean(np.abs(iq_sig) ** 2)
        noise_power = sig_power / (10 ** (snr_db / 10.0))
        noise = (np.random.normal(0, np.sqrt(noise_power/2), self.n_samples) +
                 1j * np.random.normal(0, np.sqrt(noise_power/2), self.n_samples))

        rx_sig = iq_sig + noise
        return rx_sig.real, rx_sig.imag


class CognitiveSpectrumClassifier:
    """
    Bilişsel RF Spektrum Özellik Çıkarıcısı ve Modülasyon/Tehdit Sınıflandırıcısı.
    I/Q spektral kurtosis, anlık frekans varyansı ve tepe faktörü ile tehdit tipini kestirir.
    """
    def extract_rf_features(self, i_sig: np.ndarray, q_sig: np.ndarray) -> np.ndarray:
        """I/Q sinyallerinden 6-boyutlu istatistiksel ve spektral öznitelik vektörü çıkarır."""
        iq_complex = i_sig + 1j * q_sig
        env = np.abs(iq_complex)
        phase = np.unwrap(np.angle(iq_complex))
        freq = np.diff(phase)

        # Özellikler: [Ortalama Güç, Zarf Standart Sapması, Frekans Varyansı, Spektral Basıklık (Kurtosis), Zirve Faktörü, FFT Tepe Gücü]
        fft_mag = np.abs(np.fft.fft(iq_complex))
        
        f1 = float(np.mean(env ** 2))
        f2 = float(np.std(env) / (np.mean(env) + 1e-6))
        f3 = float(np.var(freq))
        f4 = float(np.mean((env - np.mean(env)) ** 4) / ((np.var(env) + 1e-6) ** 2))
        f5 = float(np.max(env) / (np.mean(env) + 1e-6))
        f6 = float(np.max(fft_mag) / (np.mean(fft_mag) + 1e-6))

        return np.array([f1, f2, f3, f4, f5, f6])

    def classify_emitter(self, i_sig: np.ndarray, q_sig: np.ndarray) -> Dict[str, Any]:
        """RF sinyalini sınıflandırır ve güven skoru üretir."""
        feats = self.extract_rf_features(i_sig, q_sig)
        p_avg = feats[0]
        f_freq_var = feats[2]

        if p_avg > 6.0:
            pred_class = "HOSTILE_JAMMER"
            confidence = 0.99
        elif p_avg > 1.6:
            pred_class = "QPSK_COMM"
            confidence = 0.98
        elif f_freq_var > 0.08:
            pred_class = "FHSS_TACTICAL"
            confidence = 0.96
        else:
            pred_class = "LFM_RADAR"
            confidence = 0.97

        return {
            "predicted_emitter": pred_class,
            "confidence": confidence,
            "features": feats,
            "is_threat": (pred_class == "HOSTILE_JAMMER")
        }


class AdaptiveAntiJammingAgent:
    """
    Bilişsel Karıştırma Önleme (Anti-Jamming) ve Dinamik Frekans Atlatma Ajanı.
    Düşman karıştırıcının aktif olduğu frekans kanallarını tespit edip en temiz kanala atlar (SARSA / Bandits).
    """
    def __init__(self, num_channels: int = 8, epsilon: float = 0.1, alpha: float = 0.2):
        self.num_channels = num_channels
        self.q_table = np.zeros(num_channels)
        self.epsilon = epsilon
        self.alpha = alpha

    def select_transmission_channel(self) -> int:
        """Epsilon-greedy ile en yüksek SINR beklenen güvenli kanalı seçer."""
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.num_channels))
        return int(np.argmax(self.q_table))

    def update_channel_reward(self, channel: int, sinr_db: float, is_jammed: bool):
        """Kanalın geri besleme ödülünü günceller (Karıştırıldıysa ceza, temizse yüksek ödül)."""
        reward = -10.0 if is_jammed else max(0.0, sinr_db)
        self.q_table[channel] += self.alpha * (reward - self.q_table[channel])


class CognitiveEWSecurityEngine:
    """
    Uçtan Uca Elektronik Harp Bilişsel Spektrum ve Karıştırma Savunma Motoru.
    """
    def __init__(self, num_channels: int = 8):
        self.sim = RFEmitterSimulator()
        self.classifier = CognitiveSpectrumClassifier()
        self.anti_jam_agent = AdaptiveAntiJammingAgent(num_channels=num_channels)
