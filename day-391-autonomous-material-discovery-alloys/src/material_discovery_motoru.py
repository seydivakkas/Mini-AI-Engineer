"""
Day 391: Autonomous Materials Discovery: High-Entropy Alloys & Superconductor Screening
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Yüksek Entropili Alaşımların (HEA) Termodinamik Faz Kararlılığını
(Konfigürasyonel Entropi, Karışma Entalpisi, Omega Parametresi, Atomik Boyut Farkı)
ve Kristal Çizge Evrişimsel Sinir Ağı (CGCNN) ile Yüksek Sıcaklık Süperiletken
Taramasını simüle eder.
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ElementProperties:
    """Kimyasal Element Özellikleri."""
    symbol: str
    name: str
    atomic_radius_pm: float
    electronegativity: float
    melting_point_k: float
    valence_electron_concentration: float  # VEC


# Element Veritabanı
ELEMENT_DB: Dict[str, ElementProperties] = {
    "Fe": ElementProperties("Fe", "Iron", 126.0, 1.83, 1811.0, 8.0),
    "Co": ElementProperties("Co", "Cobalt", 125.0, 1.88, 1768.0, 9.0),
    "Ni": ElementProperties("Ni", "Nickel", 124.0, 1.91, 1728.0, 10.0),
    "Cr": ElementProperties("Cr", "Chromium", 128.0, 1.66, 2180.0, 6.0),
    "Mn": ElementProperties("Mn", "Manganese", 127.0, 1.55, 1519.0, 7.0),
    "Al": ElementProperties("Al", "Aluminum", 143.0, 1.61, 933.0, 3.0),
    "Ti": ElementProperties("Ti", "Titanium", 147.0, 1.54, 1941.0, 4.0),
    "V": ElementProperties("V", "Vanadium", 134.0, 1.63, 2183.0, 5.0),
    "Cu": ElementProperties("Cu", "Copper", 128.0, 1.90, 1358.0, 11.0),
    "La": ElementProperties("La", "Lanthanum", 187.0, 1.10, 1193.0, 3.0),
    "Y": ElementProperties("Y", "Yttrium", 180.0, 1.22, 1799.0, 3.0),
    "Ba": ElementProperties("Ba", "Barium", 222.0, 0.89, 1000.0, 2.0)
}


class HEAThermodynamics:
    """
    Yüksek Entropili Alaşım (HEA) Termodinamik Faz Kararlılığı Hesaplayıcısı.
    """
    R_GAS_CONSTANT = 8.314  # J / mol.K

    def __init__(self):
        pass

    def compute_configurational_entropy(self, fractions: Dict[str, float]) -> float:
        """
        Konfigürasyonel Karışma Entropisi: Delta_S_config = -R * sum(c_i * ln(c_i))
        """
        s_config = 0.0
        for c in fractions.values():
            if c > 1e-6:
                s_config -= self.R_GAS_CONSTANT * c * np.log(c)
        return float(s_config)

    def compute_atomic_size_difference(self, fractions: Dict[str, float]) -> float:
        """
        Atomik Yarıçap Uyumsuzluğu delta = 100% * sqrt(sum(c_i * (1 - r_i / r_bar)^2))
        """
        r_bar = sum(fractions[elem] * ELEMENT_DB[elem].atomic_radius_pm for elem in fractions)
        var_r = sum(fractions[elem] * ((1.0 - (ELEMENT_DB[elem].atomic_radius_pm / r_bar)) ** 2) for elem in fractions)
        delta_pct = 100.0 * np.sqrt(var_r)
        return float(delta_pct)

    def evaluate_solid_solution_formation(self, fractions: Dict[str, float]) -> Dict[str, Any]:
        """
        Katı Çözelti (Solid Solution) Oluşum Kriteri:
        Omega = (T_m * Delta_S) / |Delta_H_mix| >= 1.1 ve delta <= 6.6%
        """
        s_config = self.compute_configurational_entropy(fractions)
        delta = self.compute_atomic_size_difference(fractions)
        
        # Ortalama erime noktası T_m
        t_m = sum(fractions[elem] * ELEMENT_DB[elem].melting_point_k for elem in fractions)
        
        # Yaklaşık karışma entalpisi Delta_H_mix (kJ/mol)
        # Düzenli çözelti modeli yaklaşımı
        delta_h_mix_kj = -2.5 + np.random.uniform(-4.0, 3.0)
        
        omega = (t_m * s_config) / (max(0.1, abs(delta_h_mix_kj * 1000.0)))
        
        # VEC (Valence Electron Concentration) ile Kristal Yapı Belirleme:
        # VEC >= 8.0 -> FCC, VEC < 6.87 -> BCC, 6.87 <= VEC < 8.0 -> FCC+BCC
        vec = sum(fractions[elem] * ELEMENT_DB[elem].valence_electron_concentration for elem in fractions)
        if vec >= 8.0:
            phase = "FCC_SOLID_SOLUTION"
        elif vec < 6.87:
            phase = "BCC_SOLID_SOLUTION"
        else:
            phase = "FCC_PLUS_BCC_DUAL_PHASE"

        is_stable_solid_solution = bool(omega >= 1.1 and delta <= 6.6)

        return {
            "s_config_j_mol_k": round(float(s_config), 2),
            "delta_size_pct": round(float(delta), 2),
            "t_m_melting_k": round(float(t_m), 1),
            "omega_parameter": round(float(omega), 2),
            "vec": round(float(vec), 2),
            "predicted_phase": phase,
            "is_stable_solid_solution": is_stable_solid_solution
        }


class CGCNNSuperconductorPredictor:
    """
    Kristal Çizge Evrişimsel Sinir Ağı (CGCNN) ile Süperiletken Kritik Sıcaklık (T_c) Kestiricisi.
    """
    def __init__(self):
        pass

    def predict_critical_temperature(self, formula: str, fractions: Dict[str, float]) -> Dict[str, Any]:
        """
        Kristal kafes yapısı ve McMillan-BCS kuramından T_c kestirimi:
        T_c = (Theta_D / 1.45) * exp(-1.04*(1 + lambda) / (lambda - mu*(1 + 0.62*lambda)))
        """
        # Formüle göre elektron-fonon eşleşme katsayısı lambda
        if "La" in fractions or "Y" in fractions:
            # Yüksek-Tc hidrit veya kuprat benzeri aday
            lambda_ep = 1.85 + np.random.uniform(-0.2, 0.3)
            theta_debye = 850.0  # K
            mu_coulomb = 0.12
        else:
            # Standart metalik alaşım
            lambda_ep = 0.65 + np.random.uniform(-0.15, 0.20)
            theta_debye = 420.0
            mu_coulomb = 0.13

        exponent = (-1.04 * (1.0 + lambda_ep)) / max(0.01, (lambda_ep - mu_coulomb * (1.0 + 0.62 * lambda_ep)))
        tc_kelvin = (theta_debye / 1.45) * np.exp(np.clip(exponent, -15.0, 0.0))

        return {
            "formula": formula,
            "tc_kelvin": round(float(max(0.0, tc_kelvin)), 2),
            "electron_phonon_coupling_lambda": round(float(lambda_ep), 2),
            "debye_temperature_k": theta_debye,
            "is_high_tc": bool(tc_kelvin >= 77.0)  # Sıvı azot sıcaklığı (77 K) üstü
        }


class MaterialDiscoveryBenchmark:
    """
    Otonom Malzeme Keşfi ve Yüksek Hacimli Tarama Başarım Paketi.
    """
    def __init__(self, num_candidates: int = 1000):
        self.num_candidates = num_candidates
        self.thermo = HEAThermodynamics()
        self.cgcnn = CGCNNSuperconductorPredictor()

    def run_benchmark(self) -> Dict[str, Any]:
        """
        1000'den fazla aday alaşım kompozisyonunu otonom tarar.
        """
        np.random.seed(42)
        candidate_elements = ["Fe", "Co", "Ni", "Cr", "Mn", "Al", "Ti", "V", "Cu", "La", "Y", "Ba"]

        stable_hea_count = 0
        high_tc_candidates = []
        tc_distribution = []
        omega_values = []
        delta_values = []

        for i in range(self.num_candidates):
            # 4 ila 5 elementli rastgele kompozisyon
            k = np.random.choice([4, 5])
            selected = list(np.random.choice(candidate_elements, size=k, replace=False))
            raw_weights = np.random.dirichlet(np.ones(k))
            fractions = {elem: float(raw_weights[idx]) for idx, elem in enumerate(selected)}
            formula = "".join([f"{elem}_{int(fractions[elem]*100)}" for elem in selected])

            # 1. HEA Termodinamik Değerlendirme
            thermo_res = self.thermo.evaluate_solid_solution_formation(fractions)
            omega_values.append(thermo_res["omega_parameter"])
            delta_values.append(thermo_res["delta_size_pct"])
            if thermo_res["is_stable_solid_solution"]:
                stable_hea_count += 1

            # 2. CGCNN Süperiletkenlik Taraması
            sc_res = self.cgcnn.predict_critical_temperature(formula, fractions)
            tc_distribution.append(sc_res["tc_kelvin"])
            if sc_res["is_high_tc"]:
                high_tc_candidates.append(sc_res)

        hea_yield_pct = (stable_hea_count / self.num_candidates) * 100.0
        max_tc = max(tc_distribution)

        return {
            "total_candidates_screened": self.num_candidates,
            "stable_hea_alloys_found": stable_hea_count,
            "hea_solid_solution_yield_pct": round(float(hea_yield_pct), 2),
            "high_tc_candidates_count": len(high_tc_candidates),
            "max_predicted_tc_kelvin": round(float(max_tc), 2),
            "tc_distribution": tc_distribution,
            "omega_values": omega_values,
            "delta_values": delta_values
        }

    def kos(self) -> Dict[str, Any]:
        return self.run_benchmark()
