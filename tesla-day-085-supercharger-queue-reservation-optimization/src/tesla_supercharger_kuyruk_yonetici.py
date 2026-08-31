r"""
Tesla Supercharger Dinamik Kuyruk ve Rezervasyon Optimizasyon Çekirdeği
========================================================================
Bu modül; $M/M/c$ çoklu sunuculu kuyruk teorisi modelini, trafik yoğunluğunu
($\rho$), ortalama bekleme süresini ($W_q$), FSD navigasyon varış zamanı (ETA)
rezervasyonunu ve aşırı yoğunlukta alternatif istasyon yönlendirmesini gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
import math
import numpy as np


class TeslaSuperchargerQueueManager:
    """
    Tesla Supercharger M/M/c Kuyruk ve Dinamik Rota Rezervasyon Yöneticisi.
    """
    def __init__(
        self,
        num_stalls: int = 12,
        service_rate_per_stall_per_hour: float = 3.0,  # 20 dakikalık şarj süresi
        max_acceptable_wait_mins: float = 15.0
    ):
        self.c = num_stalls
        self.mu = service_rate_per_stall_per_hour
        self.max_wait_mins = max_acceptable_wait_mins

    def calculate_mmc_metrics(self, arrival_rate_lambda: float) -> Dict[str, Any]:
        """
        M/M/c Kuyruk Modeli Analitik Metriklerini Hesabeder.
        """
        c = self.c
        mu = self.mu
        lam = arrival_rate_lambda

        # Trafik yoğunluğu
        rho = lam / (c * mu)

        if rho >= 1.0:
            return {
                "arrival_rate": lam,
                "utilization_rho": float(rho),
                "is_stable": False,
                "p0": 0.0,
                "avg_queue_length_lq": float('inf'),
                "avg_wait_time_mins": float('inf'),
                "reroute_recommended": True
            }

        # 1. P0 Hesabı (Sistemde 0 araç olma olasılığı)
        sum_term = sum(( (c * rho) ** n ) / math.factorial(n) for n in range(c))
        tail_term = ((c * rho) ** c) / (math.factorial(c) * (1.0 - rho))
        p0 = 1.0 / (sum_term + tail_term)

        # 2. Lq Hesabı (Kuyruktaki ortalama araç sayısı)
        lq = (p0 * ((c * rho) ** c) * rho) / (math.factorial(c) * ((1.0 - rho) ** 2))

        # 3. Wq Hesabı (Ortalama bekleme süresi - saat ve dakika)
        wq_hours = lq / max(lam, 1e-4)
        wq_mins = wq_hours * 60.0

        reroute = bool(wq_mins > self.max_wait_mins)

        return {
            "num_stalls": c,
            "arrival_rate": lam,
            "service_rate_per_stall": mu,
            "utilization_rho": float(np.round(rho, 4)),
            "is_stable": True,
            "p0": float(np.round(p0, 4)),
            "avg_queue_length_lq": float(np.round(lq, 2)),
            "avg_wait_time_mins": float(np.round(wq_mins, 2)),
            "reroute_recommended": reroute
        }

    def evaluate_fsd_route_reservation(
        self,
        current_arrival_rate: float,
        eta_minutes: float,
        alternate_station_wait_mins: float = 3.0
    ) -> Dict[str, Any]:
        """
        FSD aracının varış süresi ve mevcut istasyon yoğunluğuna göre slot tahsisi
        veya alternatif istasyon yönlendirme kararı verir.
        """
        metrics = self.calculate_mmc_metrics(current_arrival_rate)
        wq = metrics["avg_wait_time_mins"]

        if not metrics["is_stable"] or wq > self.max_wait_mins:
            decision = "REROUTE_TO_ALTERNATE_STATION"
            assigned_wait = alternate_station_wait_mins
        else:
            decision = "CONFIRM_SUPERCHARGER_RESERVATION"
            assigned_wait = wq

        return {
            "eta_minutes": eta_minutes,
            "current_station_wait_mins": wq,
            "decision": decision,
            "assigned_wait_mins": assigned_wait,
            "stalls_busy_expected": min(self.c, int(np.ceil(metrics["utilization_rho"] * self.c)))
        }
