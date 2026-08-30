"""
Day 327: Closed-Loop Neuro-Prosthetic Control & Haptic Feedback
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Motor Korteks (M1) hız dekoderini, 2-DOF protez kol fiziğini,
Dokunsal Kuvvet Algılayıcısını ve S1 İntrakortikal Mikrostimülasyon (ICMS) Somatosensörel Geri Bildirim Motorunu içerir.
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np


class MotorCortexDecoder:
    """
    Motor Korteks (M1) Nöron Ateşleme Oranlarından 2D Protez Kol Hızı Dekoderi.
    Hız Denklemi: v(t) = K_v * r(t) + b_v
    """
    def __init__(self, num_neurons: int = 20, seed: int = 42):
        self.num_neurons = num_neurons
        np.random.seed(seed)
        # Nöronların tercih edilen yön vektörleri (Preferred Directions)
        angles = np.linspace(0, 2 * np.pi, num_neurons, endpoint=False)
        self.preferred_dirs = np.vstack([np.cos(angles), np.sin(angles)])  # (2, N)
        self.k_v = self.preferred_dirs * 0.05
        self.b_v = np.zeros(2)

    def decode_velocity(self, firing_rates: np.ndarray) -> np.ndarray:
        """
        Nöron Ateşleme Oranları r(t) in R^N -> 2D Hız [v_x, v_y]
        """
        v = self.k_v @ firing_rates + self.b_v
        return v


class ProstheticArmPlant:
    """
    2-DOF Protez Kol Fizik ve Dokunma Kuvveti Algılama Modeli.
    """
    def __init__(self, initial_pos: Tuple[float, float] = (0.0, 0.0), dt: float = 0.05):
        self.pos = np.array(initial_pos, dtype=np.float32)
        self.dt = dt
        self.trajectory: List[np.ndarray] = [self.pos.copy()]
        self.velocities: List[np.ndarray] = []

    def update_step(self, velocity: np.ndarray, object_pos: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        Protez kol konumunu günceller ve nesneyle temas varsa dokunma kuvvetini (N) hesaplar.
        """
        self.pos += velocity * self.dt
        self.trajectory.append(self.pos.copy())
        self.velocities.append(velocity.copy())

        contact_force = 0.0
        if object_pos is not None:
            dist_to_obj = np.linalg.norm(self.pos - object_pos)
            if dist_to_obj < 0.15:
                # Dokunma Kuvveti (Newton)
                contact_force = float(np.clip((0.15 - dist_to_obj) * 20.0, 0.0, 5.0))

        return self.pos.copy(), contact_force


class ICMSSomatosensoryEncoder:
    """
    Intracortical Microstimulation (ICMS) Somatosensörel Dokunsal Geri Bildirim Kodlayıcısı.
    Dokunma kuvvetini (N) Birincil Duyu Korteksine (S1) elektrik akım palaslarına dönüştürür.
    """
    def __init__(self, base_amplitude_ua: float = 10.0, base_frequency_hz: float = 20.0):
        self.base_amplitude_ua = base_amplitude_ua
        self.base_frequency_hz = base_frequency_hz

    def encode_haptic_feedback(self, contact_force_n: float) -> Tuple[float, float]:
        """
        Girdi: Dokunma Kuvveti F (N) -> Çıktı: (Akım Genliği (uA), Stimülasyon Frekansı (Hz))
        """
        if contact_force_n <= 0.0:
            return 0.0, 0.0

        amp_ua = min(100.0, self.base_amplitude_ua + contact_force_n * 15.0)
        freq_hz = min(250.0, self.base_frequency_hz + contact_force_n * 35.0)
        return float(amp_ua), float(freq_hz)


class ClosedLoopNeuroProstheticSimulator:
    """
    Kapalı Çevrim Nöro-Protez Kontrolü & Dokunsal Geri Bildirim Simülatörü.
    """
    def __init__(self, num_neurons: int = 20):
        self.num_neurons = num_neurons
        self.decoder = MotorCortexDecoder(num_neurons=num_neurons)
        self.icms_encoder = ICMSSomatosensoryEncoder()

    def run_reaching_simulation(
        self,
        target_pos: np.ndarray = np.array([1.0, 0.8]),
        object_pos: np.ndarray = np.array([0.9, 0.7]),
        num_steps: int = 60,
        closed_loop: bool = True
    ) -> Dict[str, Any]:
        """
        Hedefe Ulaşma ve Dokunsal Geri Bildirim Simülasyonunu Çalıştırır.
        """
        arm = ProstheticArmPlant(initial_pos=(0.0, 0.0), dt=0.05)
        forces = []
        amps_ua = []
        freqs_hz = []
        errors = []

        for step in range(num_steps):
            curr_pos = arm.pos
            direction_to_target = target_pos - curr_pos
            dist_to_target = float(np.linalg.norm(direction_to_target))
            errors.append(dist_to_target)

            # Nöron Popülasyon Ateşleme Oranı Simülasyonu
            if dist_to_target > 0.02:
                desired_dir = direction_to_target / dist_to_target
            else:
                desired_dir = np.zeros(2)

            # Preferred Direction'lara göre M1 ateşleme oranları
            rates = np.maximum(0.0, self.decoder.preferred_dirs.T @ desired_dir * 30.0 + 10.0 + np.random.randn(self.num_neurons)*2.0)

            # Hız Dekodlaması
            v_decoded = self.decoder.decode_velocity(rates)

            # Arm adımı ve Temas Kuvveti
            new_pos, force = arm.update_step(v_decoded, object_pos=object_pos)
            forces.append(force)

            # S1 ICMS Geri Bildirim
            amp_ua, freq_hz = self.icms_encoder.encode_haptic_feedback(force)
            amps_ua.append(amp_ua)
            freqs_hz.append(freq_hz)

            # Kapalı Çevrimde Dokunma Hissi Motor Kortekse Ulaşınca Hız Yavaşlatılır
            if closed_loop and force > 0.5:
                v_decoded *= 0.3  # Hassas Dokunma Kontrolü

        return {
            "trajectory": np.array(arm.trajectory),
            "velocities": np.array(arm.velocities) if arm.velocities else np.zeros((1, 2)),
            "forces": np.array(forces),
            "amps_ua": np.array(amps_ua),
            "freqs_hz": np.array(freqs_hz),
            "errors": np.array(errors),
            "target_pos": target_pos,
            "object_pos": object_pos,
        }
