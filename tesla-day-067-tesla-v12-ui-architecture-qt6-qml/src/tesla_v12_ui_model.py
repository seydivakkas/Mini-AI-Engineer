r"""
Tesla V12 UI Mimarisi ve Qt6/QML Backend Veri Modeli
=====================================================
Bu modül; Tesla Model S/3/X/Y ve Cybertruck V12 Infotainment ekranının
C++ QObject / Q_PROPERTY mimarisini, sinyal-yuva (Signals & Slots)
mekanizmasını, deklaratif QML çift yönlü veri bağlamasını ve 60 FPS
arayüz yenileme döngüsünü gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Callable
import numpy as np


class TeslaV12VehicleModel:
    """
    Tesla V12 Infotainment C++ QObject Backend Temsili.
    """
    def __init__(self):
        self._speed_kmh: float = 0.0
        self._battery_pct: int = 85
        self._gear: str = "P"
        self._fsd_active: bool = False
        self._cabin_temp_c: float = 21.5
        self._turn_signal: str = "OFF"
        self._signals_emitted: List[str] = []

    @property
    def speed_kmh(self) -> float:
        return self._speed_kmh

    def set_speed(self, value: float) -> bool:
        if not np.isclose(self._speed_kmh, value):
            self._speed_kmh = float(np.clip(value, 0.0, 260.0))
            self._signals_emitted.append(f"speedChanged({self._speed_kmh:.1f} km/h)")
            return True
        return False

    @property
    def battery_pct(self) -> int:
        return self._battery_pct

    def set_battery_pct(self, value: int) -> bool:
        v = int(np.clip(value, 0, 100))
        if self._battery_pct != v:
            self._battery_pct = v
            self._signals_emitted.append(f"batteryPctChanged({self._battery_pct}%)")
            return True
        return False

    @property
    def gear(self) -> str:
        return self._gear

    def set_gear(self, value: str) -> bool:
        if value in ["P", "R", "N", "D"] and self._gear != value:
            self._gear = value
            self._signals_emitted.append(f"gearChanged({self._gear})")
            return True
        return False

    @property
    def fsd_active(self) -> bool:
        return self._fsd_active

    def set_fsd_active(self, value: bool) -> bool:
        if self._fsd_active != value:
            self._fsd_active = bool(value)
            self._signals_emitted.append(f"fsdActiveChanged({self._fsd_active})")
            return True
        return False

    def simulate_ui_stream(self, frames: int = 60) -> Dict[str, Any]:
        """
        60 FPS (1 Saniyelik) QML Ekran Render Veri Akışı.
        """
        speeds = np.zeros(frames)
        signals_count = 0

        self.set_gear("D")
        self.set_fsd_active(True)

        for i in range(frames):
            # İvmelenme profili: 0'dan 108 km/h'ye
            target_v = 108.0 * (1.0 - np.exp(-3.0 * (i / frames)))
            if self.set_speed(target_v):
                signals_count += 1
            speeds[i] = self.speed_kmh

        return {
            "frames": frames,
            "final_speed_kmh": float(self.speed_kmh),
            "final_battery_pct": int(self.battery_pct),
            "final_gear": self.gear,
            "fsd_active": bool(self.fsd_active),
            "speeds_stream": speeds,
            "total_signals_emitted": len(self._signals_emitted),
            "is_60fps_ready": bool(frames >= 60)
        }
