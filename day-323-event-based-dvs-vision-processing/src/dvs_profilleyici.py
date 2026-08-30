"""
Day 323: Dynamic Vision Sensors (DVS) & Event-Based Processing
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; DVS olay akışı işleme hızını (Events/sec), veri sıkıştırma kazancını (Compression Gain)
ve gecikme (latency) metriklerini profillemek için kullanılır.
"""

from typing import Dict, Any
import numpy as np


class DVSProfilleyici:
    """
    DVS Nöromorfik Görsel İşleme ve Veri Hacmi Profilleyicisi.
    """
    BYTES_PER_EVENT: int = 8  # 2B x, 2B y, 3B t_us, 1B polarity = 8 Bytes

    @staticmethod
    def profille(
        events: np.ndarray,
        height: int,
        width: int,
        duration_us: float,
        fps_equivalent: float = 60.0
    ) -> Dict[str, Any]:
        """
        Olay akışının bant genişliğini ve geleneksel kare kameralara göre sıkıştırma oranını hesaplar.
        """
        num_events = len(events)
        duration_sec = (duration_us / 1e6) + 1e-9

        # DVS Veri Hacmi (Bytes)
        dvs_bytes = num_events * DVSProfilleyici.BYTES_PER_EVENT

        # Geleneksel RGB Kare Kamera Veri Hacmi (8-bit RGB = 3 Bytes per pixel)
        total_frames = int(duration_sec * fps_equivalent)
        frame_bytes = total_frames * height * width * 3

        # Sıkıştırma Kazancı (Data Reduction Ratio)
        compression_ratio = float(frame_bytes / (dvs_bytes + 1e-9))

        # Olay İşleme Hızı (Events / sec)
        throughput_eps = float(num_events / duration_sec)

        # Olay Seyreklik Oranı (Pixels with events vs total space-time pixels)
        total_space_time_units = height * width * total_frames
        sparsity = float(1.0 - (num_events / (total_space_time_units + 1e-9)))

        return {
            "num_events": num_events,
            "duration_ms": duration_us / 1000.0,
            "dvs_bytes": dvs_bytes,
            "frame_bytes": frame_bytes,
            "compression_ratio_x": compression_ratio,
            "throughput_events_per_sec": throughput_eps,
            "temporal_sparsity_pct": max(0.0, min(100.0, sparsity * 100.0)),
        }
