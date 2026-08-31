"""
Tesla Batarya Hücre Kimyası ve Eşdeğer Devre Modeli (ECM)
==========================================================
Bu modül; LFP (Lityum Demir Fosfat) ve NMC/NCA hücre kimyalarını,
1-RC Thevenin ve 2-RC Dual Polarization eşdeğer devre modellerini (ECM),
OCV-SoC ilişkisini ve sıcaklığa bağlı iç direnç (Arrhenius) dinamiklerini gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import IntEnum
import numpy as np


class BatteryChemistry(IntEnum):
    LFP = 0  # Lithium Iron Phosphate (Model 3 Standard Range - Düz OCV Platosu)
    NMC = 1  # Nickel Manganese Cobalt (Model 3/Y Long Range / Performance)
    NCA = 2  # Nickel Cobalt Aluminum (Tesla 2170 / 4680 Hücreleri)


@dataclass
class BatteryCellParameters:
    chemistry: BatteryChemistry
    nominal_capacity_ah: float = 75.0   # Nominal Kapasite (Ah)
    r0_ohmic_ohm: float = 0.0015        # Ohmik İç Direnç (1.5 mOhm)
    r1_polarization_ohm: float = 0.0010 # Hızlı Polarizasyon Direnci (1.0 mOhm)
    c1_polarization_f: float = 2500.0   # Hızlı Çift Katman Kapasitansı (2500 F)
    r2_diffusion_ohm: float = 0.0008    # Yavaş Difüzyon Direnci (0.8 mOhm)
    c2_diffusion_f: float = 20000.0     # Yavaş Difüzyon Kapasitansı (20000 F)
    t_ref_k: float = 298.15             # 25 °C Referans Sıcaklık
    arrhenius_ea: float = 25000.0       # Aktivasyon Enerjisi (J/mol)


class TeslaBatteryECM:
    """
    Tesla 2-RC Dual Polarization Eşdeğer Devre Modeli (ECM).
    V_t = OCV(SoC) - I * R0 - V_rc1 - V_rc2
    """
    def __init__(self, params: BatteryCellParameters, initial_soc: float = 0.90, initial_temp_c: float = 25.0):
        self.params = params
        self.soc = float(initial_soc)
        self.temp_c = float(initial_temp_c)
        self.v_rc1 = 0.0
        self.v_rc2 = 0.0
        self.nominal_coulombs = self.params.nominal_capacity_ah * 3600.0

    def compute_ocv(self, soc: float) -> float:
        """SoC değerine göre Açık Devre Voltajı (OCV) hesaplar."""
        soc_clamped = np.clip(soc, 0.0, 1.0)
        if self.params.chemistry == BatteryChemistry.LFP:
            # LFP Kimyası: %15 ile %85 arasında çok düz 3.28V-3.32V platosu
            if soc_clamped < 0.10:
                return float(2.80 + 4.0 * soc_clamped)
            elif soc_clamped > 0.90:
                return float(3.34 + 2.6 * (soc_clamped - 0.90))
            else:
                # Orta plato
                return float(3.20 + 0.15 * soc_clamped + 0.05 * np.sin(np.pi * soc_clamped))
        else:
            # NMC / NCA Kimyası: 3.0V ile 4.2V arasında belirgin eğimli OCV eğrisi
            return float(3.00 + 1.20 * soc_clamped + 0.05 * np.log(soc_clamped + 1e-4) - 0.02 * np.exp(-15 * soc_clamped))

    def get_temperature_adjusted_r0(self) -> float:
        """Arrhenius denklemine göre sıcaklığa bağlı iç direnci döner."""
        t_k = self.temp_c + 273.15
        r_gas = 8.314  # J/(mol*K)
        # Soğukta iç direnç katlanarak artar!
        factor = np.exp((self.params.arrhenius_ea / r_gas) * (1.0 / t_k - 1.0 / self.params.t_ref_k))
        return float(self.params.r0_ohmic_ohm * factor)

    def step(self, current_a: float, dt_s: float = 0.1) -> Dict[str, float]:
        """
        1 Adım ECM Simülasyonu İlerletir (Disşarj: I > 0, Rejenerasyon/Şarj: I < 0).
        """
        # 1. Coulomb Counting ile SoC Güncellemesi
        delta_soc = (current_a * dt_s) / self.nominal_coulombs
        self.soc = float(np.clip(self.soc - delta_soc, 0.0, 1.0))

        # 2. RC Dalı Dinamikleri (Exact Exponential Discretization)
        tau1 = self.params.r1_polarization_ohm * self.params.c1_polarization_f
        tau2 = self.params.r2_diffusion_ohm * self.params.c2_diffusion_f

        exp1 = np.exp(-dt_s / tau1)
        exp2 = np.exp(-dt_s / tau2)

        self.v_rc1 = float(exp1 * self.v_rc1 + self.params.r1_polarization_ohm * (1.0 - exp1) * current_a)
        self.v_rc2 = float(exp2 * self.v_rc2 + self.params.r2_diffusion_ohm * (1.0 - exp2) * current_a)

        # 3. Terminal Voltajı Hesabı
        ocv = self.compute_ocv(self.soc)
        r0 = self.get_temperature_adjusted_r0()
        v_terminal = ocv - (current_a * r0) - self.v_rc1 - self.v_rc2

        # 4. Joule Isınması (P_loss = I^2 * R0 + I*V_rc1 + I*V_rc2)
        p_loss_w = (current_a ** 2) * r0 + current_a * (self.v_rc1 + self.v_rc2)
        # Basit termal model: C_th * dT/dt = P_loss - h * (T - T_amb)
        heat_cap_j_k = 850.0  # Hücre ısı kapasitesi
        h_cool_w_k = 2.5      # Soğutma katsayısı
        t_amb_c = 25.0
        self.temp_c += float((p_loss_w - h_cool_w_k * (self.temp_c - t_amb_c)) * dt_s / heat_cap_j_k)

        return {
            "soc": self.soc,
            "ocv_v": ocv,
            "v_terminal": float(v_terminal),
            "v_rc1": self.v_rc1,
            "v_rc2": self.v_rc2,
            "r0_ohm": r0,
            "temp_c": self.temp_c,
            "p_loss_w": float(p_loss_w)
        }
