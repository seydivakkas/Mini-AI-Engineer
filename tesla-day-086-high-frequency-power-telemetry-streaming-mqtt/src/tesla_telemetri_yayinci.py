r"""
Tesla Yüksek Frekanslı Güç Telemetrisi ve MQTT/Kafka Akış Çekirdeği
====================================================================
Bu modül; 100 Hz örneklemeli elektrik gücü telemetrisini (Gerilim, Akım,
P, Q, Frekans, Sıcaklık), 32-baytlık kompakt binary serileştirmeyi,
1000 elemanlı halka arabelleği (Circular Buffer) ve 1 saniyelik kayan
pencere (Sliding Window) istatistiksel özetleyicisini gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import struct
import numpy as np


class TeslaPowerTelemetryStreamer:
    """
    Tesla 100 Hz Güç Telemetri Yayıncısı ve Sliding Window Toplayıcısı.
    """
    # 32-bayt Binary Paket Biçimi:
    # Q (uint64 timestamp_ns) + f (float32 voltage) + f (float32 current) +
    # f (float32 active_p) + f (float32 reactive_q) + f (float32 freq) + f (float32 temp)
    STRUCT_FORMAT = ">Qffffff"
    PACKET_SIZE_BYTES = struct.calcsize(STRUCT_FORMAT)  # 32 Bayt

    def __init__(self, buffer_capacity: int = 1000):
        self.buffer_capacity = buffer_capacity
        self.ring_buffer = deque(maxlen=buffer_capacity)
        self.sliding_window = deque(maxlen=100)  # 100 Hz -> 1 saniyelik pencere

    def pack_telemetry(
        self,
        timestamp_ns: int,
        voltage_v: float,
        current_a: float,
        active_power_kw: float,
        reactive_power_kvar: float,
        freq_hz: float,
        temp_c: float
    ) -> bytes:
        """Telemetri paketini 32 baytlık optimize binary biçimine serileştirir."""
        return struct.pack(
            self.STRUCT_FORMAT,
            timestamp_ns,
            voltage_v,
            current_a,
            active_power_kw,
            reactive_power_kvar,
            freq_hz,
            temp_c
        )

    def unpack_telemetry(self, raw_bytes: bytes) -> Dict[str, Any]:
        """32 baytlık binary paketi ayrıştırır."""
        fields = struct.unpack(self.STRUCT_FORMAT, raw_bytes)
        return {
            "timestamp_ns": fields[0],
            "voltage_v": fields[1],
            "current_a": fields[2],
            "active_power_kw": fields[3],
            "reactive_power_kvar": fields[4],
            "freq_hz": fields[5],
            "temp_c": fields[6]
        }

    def push_sample(
        self,
        timestamp_ns: int,
        voltage_v: float,
        current_a: float,
        active_power_kw: float,
        reactive_power_kvar: float,
        freq_hz: float,
        temp_c: float
    ) -> bytes:
        """Yeni örneği halka arabelleğe ve kayan pencereye yazar."""
        raw = self.pack_telemetry(
            timestamp_ns, voltage_v, current_a, active_power_kw, reactive_power_kvar, freq_hz, temp_c
        )
        self.ring_buffer.append(raw)
        self.sliding_window.append(active_power_kw)
        return raw

    def get_sliding_window_stats(self) -> Dict[str, float]:
        """Son 1 saniyelik (100 örnek) aktif güç istatistiklerini hesaplar."""
        if not self.sliding_window:
            return {"mean_kw": 0.0, "min_kw": 0.0, "max_kw": 0.0, "std_kw": 0.0, "count": 0}

        arr = np.array(self.sliding_window)
        return {
            "count": len(arr),
            "mean_kw": float(np.round(np.mean(arr), 2)),
            "min_kw": float(np.round(np.min(arr), 2)),
            "max_kw": float(np.round(np.max(arr), 2)),
            "std_kw": float(np.round(np.std(arr), 2))
        }
