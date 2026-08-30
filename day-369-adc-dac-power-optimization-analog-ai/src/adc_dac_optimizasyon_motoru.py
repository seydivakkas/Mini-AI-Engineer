"""
Day 369: Mixed-Signal ADC/DAC Power Optimization for Analog AI Accelerators
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Dinamik Hassasiyetli SAR ADC (Analog-Digital Çevirici) Modelini,
PWM DAC Giriş Modülatörünü, Kolon Kapılama (ADC Power Gating) ve Karma-Sinyal Enerji Motorunu içerir.
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np


class SuccessiveApproximationADC:
    """
    Ardışık Yaklaşımlı (SAR - Successive Approximation Register) ADC Modeli.
    Walden Liyakat Katsayısı (FoM): Güç tüketimi 2^N bit hassasiyeti ile orantılıdır.
    """
    def __init__(self, resolution_bits: int = 6, v_ref: float = 1.0, f_sample_mhz: float = 100.0):
        self.resolution = resolution_bits
        self.v_ref = v_ref
        self.f_sample = f_sample_mhz * 1e6 # 100 MHz
        # Walden Enerji Modeli: P = E_step * 2^N * f_s
        self.energy_per_step_fj = 15.0 # 15 fJ/conversion-step
        self.power_uw = (self.energy_per_step_fj * 1e-15) * (2 ** self.resolution) * self.f_sample * 1e6 # mikroWatt

    def quantize(self, analog_signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Analog sinyali belirtilen bit çözünürlüğünde dijitalize eder."""
        levels = 2 ** self.resolution
        step_size = self.v_ref / levels
        
        clipped = np.clip(analog_signal, 0.0, self.v_ref)
        digital_levels = np.round(clipped / step_size)
        reconstructed = digital_levels * step_size
        return digital_levels, reconstructed


class PulseWidthModulationDAC:
    """
    Darbe Genişlik Modülasyonlu (PWM) / Voltaj Tabanlı DAC Giriş Modülatörü.
    """
    def __init__(self, resolution_bits: int = 4, v_max: float = 0.8):
        self.resolution = resolution_bits
        self.v_max = v_max
        self.power_uw_per_channel = 25.0 # 25 uW düşük DAC gücü

    def convert(self, digital_in: np.ndarray) -> np.ndarray:
        """Dijital aktivasyonları analog voltaj seviyelerine dönüştürür."""
        levels = 2 ** self.resolution
        norm = np.clip(digital_in, 0, levels - 1) / (levels - 1)
        return norm * self.v_max


class AdaptiveMixedSignalCrossbar:
    """
    Adaptif Karma Sinyal (Mixed-Signal) ReRAM Çapraz Dizi Hızlandırıcısı.
    Düşük akımlı sütunlarda ADC'leri uyku moduna alır (Power Gating) ve bit derinliğini dinamik ayarlar.
    """
    def __init__(self, rows: int = 16, cols: int = 16):
        self.rows = rows
        self.cols = cols
        self.dac = PulseWidthModulationDAC(resolution_bits=4)
        self.fixed_adc = SuccessiveApproximationADC(resolution_bits=8) # Sabit 8-bit
        self.adaptive_adc = SuccessiveApproximationADC(resolution_bits=5) # Adaptif 5-bit
        self.weights = np.random.uniform(10e-6, 150e-6, (rows, cols)) # İletkenlik matrisi

    def compute_fixed_vs_adaptive(self, x_digital: np.ndarray) -> Dict[str, Any]:
        """Sabit 8-bit ADC vs Adaptif 5-bit Power-Gated ADC karşılaştırmasını yapar."""
        # 1. DAC Dönüşümü
        v_in = self.dac.convert(x_digital) # (Rows,)
        
        # 2. Analog VMM Akımı: I = V * G
        i_analog = v_in @ self.weights # (Cols,)
        
        # Normalizasyon
        i_max = np.max(i_analog) + 1e-8
        v_sensed = (i_analog / i_max) * self.fixed_adc.v_ref

        # 3. Sabit 8-bit ADC Hesaplama
        _, rec_fixed = self.fixed_adc.quantize(v_sensed)
        fixed_power_mw = (self.fixed_adc.power_uw * self.cols) / 1000.0 # mW
        
        # 4. Adaptif 5-bit + Kolon Kapılama (Power Gating)
        # Eşik altı (< %10 tepe değer) sütunlar kapatılır
        active_mask = v_sensed > (0.10 * self.fixed_adc.v_ref)
        num_active_adcs = max(1, int(np.sum(active_mask)))
        
        _, rec_adaptive = self.adaptive_adc.quantize(v_sensed)
        rec_adaptive[~active_mask] = 0.0 # Kapalı sütunlar sıfırlanır
        
        adaptive_power_mw = (self.adaptive_adc.power_uw * num_active_adcs) / 1000.0 # mW
        power_saving_pct = ((fixed_power_mw - adaptive_power_mw) / fixed_power_mw) * 100.0

        # Sadakat (Kosinüs Benzerliği)
        cos_sim = float(np.sum(rec_fixed * rec_adaptive) / (np.linalg.norm(rec_fixed) * np.linalg.norm(rec_adaptive) + 1e-8))

        return {
            "fixed_power_mw": fixed_power_mw,
            "adaptive_power_mw": adaptive_power_mw,
            "power_saving_pct": power_saving_pct,
            "cosine_similarity": cos_sim,
            "num_active_adcs": num_active_adcs,
            "total_adcs": self.cols,
            "v_sensed": v_sensed,
            "rec_fixed": rec_fixed,
            "rec_adaptive": rec_adaptive
        }


class ADCDACPowerBenchmark:
    """
    ADC/DAC Güç Optimizasyon Kıyaslama Testi.
    """
    def __init__(self, size: int = 16):
        self.crossbar = AdaptiveMixedSignalCrossbar(rows=size, cols=size)

    def run_benchmark(self) -> Dict[str, Any]:
        """Karma-sinyal çip güç tasarrufunu ve çıkarım sadakatini ölçer."""
        np.random.seed(42)
        x_dig = np.random.randint(0, 16, self.crossbar.rows)
        return self.crossbar.compute_fixed_vs_adaptive(x_dig)
