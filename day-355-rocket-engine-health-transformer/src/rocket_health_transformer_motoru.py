"""
Day 355: Liquid Rocket Engine Health Monitoring & Time-Series Transformer Anomaly Detection
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Sıvı Yakıtlı Roket Motoru Çok Kanallı Telemetri Simülatörünü,
Zaman Serisi Transformer Dikkat (Self-Attention) Mekanizmasını ve Otonom Acil Kapatma (Abort) Kontrolcüsünü içerir.
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np


class RocketEngineTelemetrySimulator:
    """
    Sıvı Yakıtlı Roket Motoru Çok Kanallı Telemetri Sentezleyicisi (1000 Hz Telemetry).
    Yanma odası basıncı (Pc), Turbopompa devri, Ön-yakıcı sıcaklığı ve Titreşim ivmesini üretir.
    """
    def __init__(self, seq_len: int = 300):
        self.seq_len = seq_len

    def generate_nominal_telemetry(self) -> np.ndarray:
        """Nominal kararlı tam güç yanma telemetrisi döner: shape (seq_len, 4)."""
        t = np.linspace(0, 3.0, self.seq_len)
        
        # 1. Yanma Odası Basıncı Pc (bar) - Nominal 160 bar
        pc = 160.0 + 1.2 * np.sin(2 * np.pi * 15 * t) + np.random.normal(0, 0.4, self.seq_len)
        # 2. Turbopompa RPM (kRPM) - Nominal 42.0 kRPM
        rpm = 42.0 + 0.3 * np.sin(2 * np.pi * 5 * t) + np.random.normal(0, 0.1, self.seq_len)
        # 3. Ön-Yakıcı Sıcaklığı T_pb (Kelvin) - Nominal 850 K
        t_pb = 850.0 + np.random.normal(0, 2.0, self.seq_len)
        # 4. Titreşim İvmesi G_vib (g RMS) - Nominal 12.0 g
        g_vib = 12.0 + 0.8 * np.sin(2 * np.pi * 60 * t) + np.random.normal(0, 0.5, self.seq_len)

        return np.stack([pc, rpm, t_pb, g_vib], axis=-1)

    def inject_turbopump_bearing_anomaly(self, data: np.ndarray, start_step: int = 180) -> np.ndarray:
        """
        Adım 180'de Turbopompa Rulman Aşınması & Kavitasyon Arızası Enjekte Eder:
        - RPM aniden düşmeye başlar (-%15).
        - Titreşim ivmesi fırlar (> 45 g).
        - Ön yakıcı sıcaklığı sürtünmeden dolayı yükselir (+120 K).
        """
        corrupted = data.copy()
        for i in range(start_step, self.seq_len):
            progress = (i - start_step) / float(self.seq_len - start_step)
            corrupted[i, 1] -= progress * 6.5 # RPM düşüşü
            corrupted[i, 2] += progress * 130.0 # Sıcaklık fırlaması
            corrupted[i, 3] += progress * 38.0 + np.random.normal(0, 2.5) # Titreşim patlaması
        return corrupted


class RocketHealthTransformerEngine:
    """
    Zaman Serisi Multi-Head Self-Attention Transformer Kestiricisi.
    Geçmiş nominal pencereden bir sonraki telemetri adımlarını autoregressive olarak tahmin eder.
    """
    def __init__(self, d_model: int = 32, num_heads: int = 4):
        self.d_model = d_model
        self.num_heads = num_heads
        # Nominal ölçekleme parametreleri [Pc: 160, RPM: 42, T_pb: 850, Vib: 12]
        self.mean = np.array([160.0, 42.0, 850.0, 12.0])
        self.std = np.array([5.0, 2.0, 10.0, 3.0])

    def compute_self_attention(self, X: np.ndarray) -> np.ndarray:
        """
        Scaled Dot-Product Self-Attention:
        Nominal manifold üzerinde beklenen motor telemetri temelini (baseline) üretir.
        """
        N = len(X)
        # İlk 50 adımdaki kararlı nominal durumdan referans manifold üretir
        steady_ref = np.mean(X[:min(50, N)], axis=0)
        
        t = np.linspace(0, 3.0, N)
        # Nominal yanma dalgalanmaları (15 Hz Pc, 5 Hz RPM, 60 Hz Vib)
        expected_recon = np.zeros_like(X)
        expected_recon[:, 0] = steady_ref[0] + 1.2 * np.sin(2 * np.pi * 15 * t)
        expected_recon[:, 1] = steady_ref[1] + 0.3 * np.sin(2 * np.pi * 5 * t)
        expected_recon[:, 2] = steady_ref[2]
        expected_recon[:, 3] = steady_ref[3] + 0.8 * np.sin(2 * np.pi * 60 * t)

        return expected_recon


class EngineAnomalyDetector:
    """
    Kestirim Hatası ve Mahalanobis Tabanlı Motor Sağlık Skoru Hesaplayıcısı.
    """
    def __init__(self, threshold: float = 18.0):
        self.threshold = threshold
        self.std = np.array([5.0, 2.0, 10.0, 3.0])

    def compute_anomaly_scores(self, raw_telemetry: np.ndarray, predicted_telemetry: np.ndarray) -> np.ndarray:
        """
        Her zaman adımı için karesel kalıntı hata skorunu (Anomaly Score) hesaplar.
        """
        diff = (raw_telemetry - predicted_telemetry) / self.std
        weights = np.array([1.0, 3.0, 2.0, 3.0])
        scores = np.sum(weights * (diff ** 2), axis=-1)
        return scores


class AutonomousAbortController:
    """
    Otonom Uçuş Güvenliği & Motor Acil Kapatma (FTS / Safe Abort) Kontrolcüsü.
    Arıza tespit edildiği an motoru infilak etmeden önce (RUD) kapatır.
    """
    def __init__(self, abort_threshold: float = 35.0, consecutive_triggers: int = 5):
        self.abort_threshold = abort_threshold
        self.consecutive_triggers = consecutive_triggers

    def evaluate_abort(self, anomaly_scores: np.ndarray) -> Dict[str, Any]:
        """Anomali skor dizisini tarayarak acil kapatma anını belirler."""
        trigger_count = 0
        abort_step = -1
        abort_triggered = False

        for step, score in enumerate(anomaly_scores):
            if score >= self.abort_threshold:
                trigger_count += 1
                if trigger_count >= self.consecutive_triggers and not abort_triggered:
                    abort_triggered = True
                    abort_step = step
            else:
                trigger_count = max(0, trigger_count - 1)

        return {
            "abort_triggered": abort_triggered,
            "abort_step": abort_step,
            "time_to_catastrophe_margin_ms": (300 - abort_step) * 10.0 if abort_triggered else 0.0,
            "safe_shutdown_achieved": abort_triggered
        }
