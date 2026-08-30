"""
Day 331: Astrocyte-Neuron Metabolic Interaction & Slow Neuromodulation
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Astrosit Kalsiyum Dinamiklerini (Ca2+ Salınımları), Üçlü Sinaps (Tripartite Synapse)
ve Astrosit-Nöron Laktat Mekiği (ANLS) Metabolik Destek Ağını içerir.
"""

from typing import Tuple, Dict, Any, List, Optional
import math
import numpy as np


class AstrocyteCalciumModel:
    """
    Astrosit İçi Kalsiyum [Ca2+] ve IP3 Dinamikleri Modeli.
    Sinaptik glutamat uyarımı ile hücresel kalsiyum salınımını ve gliotransmiter fırlatmayı simüle eder.
    """
    def __init__(self, ca_rest: float = 0.05, theta_ca: float = 0.35, tau_ca: float = 15.0):
        self.ca_rest = ca_rest
        self.theta_ca = theta_ca
        self.tau_ca = tau_ca
        self.ca_conc = ca_rest

    def update_calcium(self, glutamate_input: float, dt: float = 1.0) -> Tuple[float, float]:
        """
        Girdi: Sinaptik Glutamat Yoğunluğu -> Çıktı: ([Ca2+] Yoğunluğu, Gliotransmiter Salınım Düzeyi)
        """
        # Kalsiyum artış diferansiyeli
        d_ca = (-(self.ca_conc - self.ca_rest) / self.tau_ca + 0.15 * glutamate_input) * dt
        self.ca_conc = float(np.clip(self.ca_conc + d_ca, self.ca_rest, 2.0))

        # Eşik üstü Kalsiyum dalgası gliotransmiter salgılar
        gliotransmitter = 0.0
        if self.ca_conc >= self.theta_ca:
            gliotransmitter = float(1.0 / (1.0 + np.exp(-10.0 * (self.ca_conc - self.theta_ca))))

        return self.ca_conc, gliotransmitter


class TripartiteSynapse:
    """
    Üçlü Sinaps Modeli (Presinaptik Nöron - Postsinaptik Nöron - Astrosit).
    Astrosit gliotransmiter salınımı presinaptik nörondaki salınım olasılığını (Pre-P_release) düzenler.
    """
    def __init__(self, p_base: float = 0.4, alpha_astro: float = 0.45):
        self.p_base = p_base
        self.alpha_astro = alpha_astro
        self.astrocyte = AstrocyteCalciumModel()
        self.current_p_release = p_base

    def step_synapse(self, presynaptic_spike: bool) -> Tuple[bool, float, float]:
        """
        Girdi: Presinaptik Spike (True/False) -> Çıktı: (Postsinaptik İletim Spike, Ca2+ Düzeyi, P_release)
        """
        glutamate = 1.0 if presynaptic_spike else 0.0
        ca_level, gliotransmitter = self.astrocyte.update_calcium(glutamate)

        # Yavaş nöromodülasyon ile P_release güncellemesi
        self.current_p_release = float(np.clip(self.p_base + self.alpha_astro * gliotransmitter, 0.1, 0.95))

        # Olasılıksal sinaptik vesikül salınımı
        transmitted = False
        if presynaptic_spike:
            if np.random.rand() < self.current_p_release:
                transmitted = True

        return transmitted, ca_level, self.current_p_release


class AstrocyteMetabolicNetwork:
    """
    Astrosit-Nöron Metabolik Etkileşim ve ANLS (Lactate Shuttle) Ağı.
    Yüksek nöronal ateşleme frekansında astrositler nöronlara laktat/ATP enerjisi sağlar.
    """
    def __init__(self, num_neurons: int = 10):
        self.num_neurons = num_neurons
        self.synapses = [TripartiteSynapse() for _ in range(num_neurons)]
        self.atp_levels = np.ones(num_neurons, dtype=np.float32) * 100.0  # %100 ATP

    def simulate_step(self, spike_vector: np.ndarray) -> Dict[str, Any]:
        """
        Ağ Adım Simülasyonu.
        """
        ca_levels = []
        p_releases = []
        transmitted_count = 0

        for i in range(self.num_neurons):
            spike = bool(spike_vector[i])
            tx, ca, p_rel = self.synapses[i].step_synapse(spike)
            ca_levels.append(ca)
            p_releases.append(p_rel)
            if tx:
                transmitted_count += 1

            # ATP Tüketimi ve Astrosit Laktat İkmalı (ANLS)
            if spike:
                self.atp_levels[i] -= 2.0  # Tüketim
            
            # Astrosit metabolik yenileme
            if ca > 0.3:
                self.atp_levels[i] = min(100.0, self.atp_levels[i] + 3.0)  # Laktat Mekiği İkmalı

        return {
            "transmitted_count": transmitted_count,
            "mean_ca": float(np.mean(ca_levels)),
            "mean_p_release": float(np.mean(p_releases)),
            "mean_atp": float(np.mean(self.atp_levels)),
        }
