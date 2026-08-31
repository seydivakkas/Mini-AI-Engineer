r"""
Tesla LLC Rezonant Dönüştürücü ve SiC MOSFET Güç Kaybı Çekirdeği
================================================================
Bu modül; Tesla Supercharger ve Onboard Charger (OBC) DC-DC güç katı için
LLC Rezonans devresi analizini ($f_r \approx 265\text{ kHz}$), Sıfır Gerilimde
Anahtarlama (ZVS) davranışını, Silisyum Karbür (SiC) MOSFET iletim ve
anahtarlama kayıp modellerini ve %98.5+ dönüştürücü verimliliğini gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
import math
import numpy as np


class TeslaLLCResonantConverter:
    """
    Tesla LLC Rezonant Dönüştürücü ve SiC MOSFET Kayıp Hesaplayıcısı.
    """
    def __init__(
        self,
        l_r_henry: float = 15.0e-6,      # 15 uH Rezonans Bobini
        c_r_farad: float = 24.0e-9,      # 24 nF Rezonans Kondansatörü
        l_m_henry: float = 75.0e-6,      # 75 uH Mıknatıslanma Bobini
        r_dson_25c_ohm: float = 0.015,   # 15 mOhm SiC MOSFET
        e_sw_nominal_j: float = 1.2e-4   # 0.12 mJ Nominal Anahtarlama Enerjisi
    ):
        self.l_r = l_r_henry
        self.c_r = c_r_farad
        self.l_m = l_m_henry
        self.r_dson_25 = r_dson_25c_ohm
        self.e_sw_nom = e_sw_nominal_j

        # Rezonans Frekansı fr = 1 / (2*pi*sqrt(Lr*Cr))
        self.f_r_hz = 1.0 / (2.0 * math.pi * math.sqrt(self.l_r * self.c_r))

    def calculate_r_dson(self, junction_temp_c: float) -> float:
        """Sıcaklığa bağlı SiC MOSFET iletim direnci (pozitif sıcaklık katsayısı)."""
        temp_delta = max(0.0, junction_temp_c - 25.0)
        return float(self.r_dson_25 * (1.0 + 0.005 * temp_delta))

    def calculate_losses(
        self,
        i_rms_a: float = 40.0,
        f_sw_hz: Optional[float] = None,
        junction_temp_c: float = 75.0,
        v_out_v: float = 800.0,
        enable_zvs: bool = True
    ) -> Dict[str, Any]:
        """
        SiC MOSFET iletim, anahtarlama ve toplam dönüştürücü kaybı/verimlilik hesabı.
        """
        f_sw = f_sw_hz if f_sw_hz is not None else self.f_r_hz

        # 1. İletim Kaybı (Conduction Loss) - 4 MOSFET (H-Köprüsü)
        r_on = self.calculate_r_dson(junction_temp_c)
        p_cond_single = (i_rms_a ** 2) * r_on
        p_cond_total = 4.0 * p_cond_single

        # 2. Anahtarlama Kaybı (Switching Loss)
        # ZVS (Sıfır Gerilimde Anahtarlama) aktifken anahtarlama enerjisi %85 azalır
        e_effective = self.e_sw_nom * (0.15 if enable_zvs else 1.0)
        p_sw_single = e_effective * f_sw
        p_sw_total = 4.0 * p_sw_single

        # 3. Manyetik Çekirdek ve Ek Kayıplar (Yaklaşık %10 toplam kayıp katkısı)
        p_magnetic = (p_cond_total + p_sw_total) * 0.12

        total_loss_w = p_cond_total + p_sw_total + p_magnetic

        # Çıkış Gücü
        p_out_w = v_out_v * (i_rms_a * 0.95)  # DC ortalama akım
        p_in_w = p_out_w + total_loss_w

        efficiency_pct = (p_out_w / p_in_w) * 100.0 if p_in_w > 0 else 0.0

        return {
            "resonant_freq_hz": float(np.round(self.f_r_hz, 2)),
            "switching_freq_hz": float(np.round(f_sw, 2)),
            "zvs_active": enable_zvs,
            "junction_temp_c": junction_temp_c,
            "r_dson_ohm": float(np.round(r_on, 5)),
            "p_conduction_w": float(np.round(p_cond_total, 2)),
            "p_switching_w": float(np.round(p_sw_total, 2)),
            "p_magnetic_w": float(np.round(p_magnetic, 2)),
            "total_loss_w": float(np.round(total_loss_w, 2)),
            "p_out_w": float(np.round(p_out_w, 2)),
            "efficiency_pct": float(np.round(efficiency_pct, 2))
        }
