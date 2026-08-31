r"""
Tesla HVAC Dokunmatik Kontrol ve Step Motor PID Sürücü Çekirdeği
================================================================
Bu modül; Tesla Model 3/Y patentli gizli hava menfezi (Hidden HVAC Air Vent)
akışkanlar mekaniğini (Fluidic Coanda Effect), dokunmatik ekran koordinatlarından
step motor açılarına dönüşümü ve kabin sıcaklığı kapalı döngü PID kontrolünü gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class TeslaHVACPIDController:
    """
    Tesla HVAC Kabin Sıcaklık PID Kontrolcüsü ve Step Motor Sürücüsü.
    """
    def __init__(
        self,
        kp: float = 2.5,
        ki: float = 0.05,
        kd: float = 1.2,
        dt: float = 0.1,
        target_temp_c: float = 21.5,
        initial_temp_c: float = 35.0,
        ambient_temp_c: float = 38.0
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.target_temp = target_temp_c
        self.current_temp = initial_temp_c
        self.ambient_temp = ambient_temp_c

        self.integral_error = 0.0
        self.prev_error = 0.0
        self.max_integral = 50.0  # Anti-windup clamping

        # Step motor parametreleri (1.8 derece / adım)
        self.deg_per_step = 1.8

    def calculate_stepper_pulses(self, target_angle_deg: float) -> int:
        """Hedef flap açısını step motor darbe sayısına dönüştürür."""
        clamped_deg = float(np.clip(target_angle_deg, -45.0, 45.0))
        return int(np.round(clamped_deg / self.deg_per_step))

    def step(self, target_temp: Optional[float] = None) -> Dict[str, Any]:
        """Tek bir PID adımı çalıştırır ve kabin termal durumunu günceller."""
        if target_temp is not None:
            self.target_temp = target_temp

        # Hata: Sıcaklık farkı (Mevcut - Hedef). Pozitif ise soğutma gerekir.
        error = self.current_temp - self.target_temp

        self.integral_error += error * self.dt
        self.integral_error = float(np.clip(self.integral_error, -self.max_integral, self.max_integral))

        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error

        # PID Kontrol Çıktısı (Kompresör ve Fan Soğutma Gücü %0 - %100)
        u_raw = self.kp * error + self.ki * self.integral_error + self.kd * derivative
        u_power = float(np.clip(u_raw, 0.0, 100.0))

        # Termal Dinamik Simülasyonu: dT/dt = -alpha * u + beta * (T_ambient - T_cabin)
        cooling_effect = 0.45 * (u_power / 100.0)
        ambient_heat_gain = 0.01 * (self.ambient_temp - self.current_temp)

        self.current_temp += (-cooling_effect + ambient_heat_gain) * self.dt

        return {
            "current_temp_c": self.current_temp,
            "target_temp_c": self.target_temp,
            "cooling_power_pct": u_power,
            "error_c": error
        }

    def simulate_cooling_trajectory(self, duration_s: float = 60.0) -> Dict[str, Any]:
        """Belirtilen süre boyunca kabin soğutma sürecini simüle eder."""
        steps = int(duration_s / self.dt)
        zamanlar = []
        sicakliklar = []
        gucler = []
        hatalar = []

        for i in range(steps):
            t = i * self.dt
            res = self.step()
            zamanlar.append(t)
            sicakliklar.append(res["current_temp_c"])
            gucler.append(res["cooling_power_pct"])
            hatalar.append(res["error_c"])

        return {
            "zamanlar_s": zamanlar,
            "sicakliklar_c": sicakliklar,
            "gucler_pct": gucler,
            "hatalar_c": hatalar,
            "final_temp_c": self.current_temp,
            "settling_achieved": bool(abs(self.current_temp - self.target_temp) < 0.5)
        }
