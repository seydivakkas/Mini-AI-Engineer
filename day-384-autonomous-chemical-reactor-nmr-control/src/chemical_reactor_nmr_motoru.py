"""
Day 384: Autonomous Chemical Reactor Control with Real-Time NMR Spectroscopy Feedback
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Sürekli Karıştırmalı Tank Reaktöründe (CSTR) Arrhenius kinetiğini,
Çevrimiçi 1H-NMR Spektrometresi Pik Ayrıştırmasını (Lorentzian Deconvolution),
ve Termal Kaçak Önleyici Uyarlamalı Sıcaklık/Dozaj Kontrolcüsünü (MPC/PID) simüle eder.
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np
from dataclasses import dataclass, field


@dataclass
class CSTRReactorState:
    """CSTR Kimyasal Reaktör Durum Modeli."""
    temp_k: float = 330.0
    jacket_temp_k: float = 315.0
    c_a: float = 1.0   # Reaktif A derişimi (mol/L)
    c_b: float = 1.0   # Reaktif B derişimi (mol/L)
    c_c: float = 0.0   # Hedef Ürün C derişimi (mol/L)
    c_d: float = 0.0   # Yan Ürün D derişimi (mol/L)
    flow_rate_l_min: float = 2.0  # Giriş/Çıkış debisi (L/dk)


@dataclass
class NMRSpectrum:
    """Çevrimiçi 1H-NMR Spektrumu."""
    ppm_axis: np.ndarray
    intensity: np.ndarray
    peak_areas: Dict[str, float]
    snr_db: float


class NMRSpectrometerModel:
    """
    Çevrimiçi (Online/Benchtop) 1H-NMR Spektrometre Simülatörü.
    A (2.1 ppm), B (3.4 ppm), C (4.8 ppm) ve D (7.2 ppm) piklerini Lorentzian çizgileriyle üretir.
    """
    def __init__(self, num_points: int = 500, snr_noise_level: float = 0.015):
        self.ppm_axis = np.linspace(0.0, 10.0, num_points)
        self.noise_level = snr_noise_level

        # Kimyasal Kayma Merkezleri ve Çizgi Genişlikleri (PPM)
        self.peak_centers = {"A": 2.1, "B": 3.4, "C": 4.8, "D": 7.2}
        self.peak_widths = {"A": 0.12, "B": 0.14, "C": 0.10, "D": 0.15}

    def generate_spectrum(self, c_a: float, c_b: float, c_c: float, c_d: float) -> NMRSpectrum:
        """
        Reaktör derişimlerine göre 1H-NMR spektrumunu oluşturur.
        I(nu) = sum_i A_i * Gamma / ((nu - nu_0)^2 + Gamma^2) + noise
        """
        spectrum = np.zeros_like(self.ppm_axis)
        concentrations = {"A": c_a, "B": c_b, "C": c_c, "D": c_d}
        peak_areas = {}

        for key, conc in concentrations.items():
            c0 = self.peak_centers[key]
            gamma = self.peak_widths[key]
            lorentzian = conc * (gamma / (np.pi * ((self.ppm_axis - c0) ** 2 + gamma ** 2)))
            spectrum += lorentzian
            peak_areas[key] = float(conc)

        # Gürültü ekleme (Thermal RF Noise)
        noise = np.random.normal(0, self.noise_level, len(self.ppm_axis))
        noisy_spectrum = np.maximum(0.0, spectrum + noise)

        signal_pwr = np.mean(spectrum ** 2)
        noise_pwr = np.mean(noise ** 2)
        snr_db = 10.0 * np.log10(max(1e-6, signal_pwr / max(1e-8, noise_pwr)))

        return NMRSpectrum(
            ppm_axis=self.ppm_axis,
            intensity=noisy_spectrum,
            peak_areas=peak_areas,
            snr_db=round(snr_db, 2)
        )

    def deconvolute_concentrations(self, spectrum: NMRSpectrum) -> Dict[str, float]:
        """
        Spektrumdaki pik alanlarını integralle hesaplayarak derişimleri çıkarır.
        """
        extracted = {}
        for key, c0 in self.peak_centers.items():
            mask = np.abs(self.ppm_axis - c0) <= 0.35
            integral_val = np.trapezoid(spectrum.intensity[mask], self.ppm_axis[mask]) if hasattr(np, 'trapezoid') else np.trapz(spectrum.intensity[mask], self.ppm_axis[mask])
            extracted[key] = float(max(0.0, integral_val))
        return extracted


class CSTRKineticsEngine:
    """
    Sürekli Karıştırmalı Tank Reaktörü (CSTR) Kinetik ve Termal Diferansiyel Denklem Motoru.
    Tepkime: A + B -> C (k1, Hedef) ve C + B -> D (k2, İstenmeyen Yan Ürün)
    """
    def __init__(self, volume_l: float = 10.0):
        self.V = volume_l
        self.c_a_in = 2.0  # Giriş A derişimi (mol/L)
        self.c_b_in = 2.5  # Giriş B derişimi (mol/L)
        self.t_in = 298.15 # Giriş besleme sıcaklığı (K)

        # Arrhenius Parametreleri: k(T) = A * exp(-Ea / (R*T))
        self.R = 8.314e-3  # kJ / (mol * K)
        self.A1 = 1.2e6    # L / (mol * min)
        self.Ea1 = 42.0    # kJ / mol
        self.A2 = 8.5e5    # L / (mol * min)
        self.Ea2 = 48.0    # kJ / mol

        # Termal Parametreler
        self.delta_H1 = -55.0  # kJ / mol (Ekzotermik)
        self.delta_H2 = -35.0  # kJ / mol
        self.rho_cp = 4.18     # kJ / (L * K)
        self.UA = 18.5         # Ceket ısı transfer katsayısı (kJ / (min * K))

    def step_rk4(self, state: CSTRReactorState, dt_min: float = 0.1) -> CSTRReactorState:
        """
        4. Dereceden Runge-Kutta (RK4) ile CSTR kütle ve enerji denklem setini çözer.
        """
        def derivatives(s: CSTRReactorState) -> Tuple[float, float, float, float, float]:
            k1 = self.A1 * np.exp(-self.Ea1 / (self.R * s.temp_k))
            k2 = self.A2 * np.exp(-self.Ea2 / (self.R * s.temp_k))

            r1 = k1 * s.c_a * s.c_b
            r2 = k2 * s.c_c * s.c_b

            tau = self.V / max(0.1, s.flow_rate_l_min)

            dc_a = (self.c_a_in - s.c_a) / tau - r1
            dc_b = (self.c_b_in - s.c_b) / tau - r1 - r2
            dc_c = (0.0 - s.c_c) / tau + r1 - r2
            dc_d = (0.0 - s.c_d) / tau + r2

            # Enerji Dengesi (dT/dt)
            heat_gen = (-self.delta_H1 * r1) + (-self.delta_H2 * r2)
            heat_trans = self.UA * (s.jacket_temp_k - s.temp_k) / self.V
            heat_flow = (s.flow_rate_l_min / self.V) * self.rho_cp * (self.t_in - s.temp_k)
            dT = (heat_gen + heat_trans + heat_flow) / self.rho_cp

            return dc_a, dc_b, dc_c, dc_d, dT

        # RK4 Adımı
        k1_a, k1_b, k1_c, k1_d, k1_T = derivatives(state)
        
        s2 = CSTRReactorState(
            temp_k=state.temp_k + 0.5 * dt_min * k1_T,
            jacket_temp_k=state.jacket_temp_k,
            c_a=max(0.0, state.c_a + 0.5 * dt_min * k1_a),
            c_b=max(0.0, state.c_b + 0.5 * dt_min * k1_b),
            c_c=max(0.0, state.c_c + 0.5 * dt_min * k1_c),
            c_d=max(0.0, state.c_d + 0.5 * dt_min * k1_d),
            flow_rate_l_min=state.flow_rate_l_min
        )
        k2_a, k2_b, k2_c, k2_d, k2_T = derivatives(s2)

        s3 = CSTRReactorState(
            temp_k=state.temp_k + 0.5 * dt_min * k2_T,
            jacket_temp_k=state.jacket_temp_k,
            c_a=max(0.0, state.c_a + 0.5 * dt_min * k2_a),
            c_b=max(0.0, state.c_b + 0.5 * dt_min * k2_b),
            c_c=max(0.0, state.c_c + 0.5 * dt_min * k2_c),
            c_d=max(0.0, state.c_d + 0.5 * dt_min * k2_d),
            flow_rate_l_min=state.flow_rate_l_min
        )
        k3_a, k3_b, k3_c, k3_d, k3_T = derivatives(s3)

        s4 = CSTRReactorState(
            temp_k=state.temp_k + dt_min * k3_T,
            jacket_temp_k=state.jacket_temp_k,
            c_a=max(0.0, state.c_a + dt_min * k3_a),
            c_b=max(0.0, state.c_b + dt_min * k3_b),
            c_c=max(0.0, state.c_c + dt_min * k3_c),
            c_d=max(0.0, state.c_d + dt_min * k3_d),
            flow_rate_l_min=state.flow_rate_l_min
        )
        k4_a, k4_b, k4_c, k4_d, k4_T = derivatives(s4)

        new_ca = max(0.0, state.c_a + (dt_min / 6.0) * (k1_a + 2*k2_a + 2*k3_a + k4_a))
        new_cb = max(0.0, state.c_b + (dt_min / 6.0) * (k1_b + 2*k2_b + 2*k3_b + k4_b))
        new_cc = max(0.0, state.c_c + (dt_min / 6.0) * (k1_c + 2*k2_c + 2*k3_c + k4_c))
        new_cd = max(0.0, state.c_d + (dt_min / 6.0) * (k1_d + 2*k2_d + 2*k3_d + k4_d))
        new_temp = state.temp_k + (dt_min / 6.0) * (k1_T + 2*k2_T + 2*k3_T + k4_T)

        return CSTRReactorState(
            temp_k=new_temp,
            jacket_temp_k=state.jacket_temp_k,
            c_a=new_ca,
            c_b=new_cb,
            c_c=new_cc,
            c_d=new_cd,
            flow_rate_l_min=state.flow_rate_l_min
        )


class ReactorAdaptiveController:
    """
    Termal Kaçak Önleyici ve Verim Maksimizasyonu Yapan Uyarlamalı Reaktör Kontrolcüsü.
    """
    def __init__(self, target_temp_k: float = 338.0, critical_temp_k: float = 360.0):
        self.target_t = target_temp_k
        self.crit_t = critical_temp_k
        self.integral_error = 0.0

    def compute_control_actions(self, current_state: CSTRReactorState, nmr_feedback: Dict[str, float]) -> Tuple[float, float]:
        """
        NMR derişim geri bildirimi ve reaktör sıcaklığına göre (Jacket_Temp, Flow_Rate) belirler.
        """
        t_err = self.target_t - current_state.temp_k
        self.integral_error += t_err * 0.1

        # PID Ceket Sıcaklığı Ayarı
        kp = 1.2
        ki = 0.05
        delta_tj = kp * t_err + ki * self.integral_error

        new_jacket_t = current_state.temp_k + delta_tj
        new_jacket_t = np.clip(new_jacket_t, 285.0, 345.0)

        # Termal Kaçak Koruması
        if current_state.temp_k > self.crit_t - 5.0:
            new_jacket_t = 280.0  # Acil soğutma

        # Besleme Debisi Ayarı (Verim C / (C+D) oranına göre optimizasyon)
        c_prod = nmr_feedback.get("C", current_state.c_c)
        d_byprod = nmr_feedback.get("D", current_state.c_d)
        
        selectivity = c_prod / max(0.01, c_prod + d_byprod)
        if selectivity > 0.85:
            new_flow = 2.2  # Yüksek debi ile üretimi artır
        else:
            new_flow = 1.6  # Debi düşürerek temas süresini optimize et

        return float(new_jacket_t), float(new_flow)


class ChemicalReactorBenchmark:
    """
    Otonom Kimyasal Reaktör ve NMR Spektrometre Başarım Paketi.
    """
    def __init__(self):
        self.nmr = NMRSpectrometerModel()
        self.kinetics = CSTRKineticsEngine(volume_l=10.0)
        self.controller = ReactorAdaptiveController(target_temp_k=338.0)

    def run_benchmark(self, num_steps: int = 50) -> Dict[str, Any]:
        """
        50 zaman adımlı sürekli reaktör sentezini ve NMR tabanlı otonom kontrolü yürütür.
        """
        np.random.seed(42)
        state = CSTRReactorState(temp_k=310.0, jacket_temp_k=320.0, c_a=1.8, c_b=2.0, c_c=0.1, c_d=0.0)

        history_ca = []
        history_cc = []
        history_cd = []
        history_temp = []
        history_yield = []
        nmr_errors = []

        for step in range(num_steps):
            # 1. NMR Spektrumu Al
            spec = self.nmr.generate_spectrum(state.c_a, state.c_b, state.c_c, state.c_d)
            nmr_conc = self.nmr.deconvolute_concentrations(spec)

            # NMR Hata Ölçümü
            nmr_err = abs(nmr_conc["C"] - state.c_c) / max(0.1, state.c_c)
            nmr_errors.append(nmr_err)

            # 2. Kontrolcü Aksiyonu (Jacket Temp & Flow)
            new_tj, new_flow = self.controller.compute_control_actions(state, nmr_conc)
            state.jacket_temp_k = new_tj
            state.flow_rate_l_min = new_flow

            # 3. Kinetik Entegrasyon (RK4)
            state = self.kinetics.step_rk4(state, dt_min=0.2)

            yield_c = (state.c_c / self.kinetics.c_a_in) * 100.0
            history_ca.append(state.c_a)
            history_cc.append(state.c_c)
            history_cd.append(state.c_d)
            history_temp.append(state.temp_k)
            history_yield.append(yield_c)

        final_yield = history_yield[-1]
        max_temp = float(np.max(history_temp))
        thermal_runaway_safe = bool(max_temp < 360.0)

        return {
            "num_steps": num_steps,
            "final_yield_pct": round(float(final_yield), 2),
            "final_product_c_mol_l": round(float(state.c_c), 3),
            "final_byproduct_d_mol_l": round(float(state.c_d), 3),
            "max_reactor_temp_k": round(max_temp, 2),
            "thermal_runaway_safe": thermal_runaway_safe,
            "avg_nmr_estimation_error_pct": round(float(np.mean(nmr_errors)) * 100.0, 2),
            "history_cc": history_cc,
            "history_ca": history_ca,
            "history_cd": history_cd,
            "history_temp": history_temp,
            "history_yield": history_yield,
            "last_spectrum": spec
        }

    def kos(self, num_steps: int = 50) -> Dict[str, Any]:
        return self.run_benchmark(num_steps)
