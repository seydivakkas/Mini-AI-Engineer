"""
Tesla Autobidder Profilleyici Modülü
====================================
Bu modül; Autobidder arbitraj karar süresini ve 24 saatlik ticaret
simülasyonu çözüm hızını profiller.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import time
import numpy as np
from typing import Dict, Any, List
from src.tesla_autobidder_ticaret_motoru import TeslaAutobidderTrader


class TeslaAutobidderProfilleyici:
    """
    Tesla Autobidder Performans Profilleyicisi.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations

    def benchmark_autobidder(self) -> Dict[str, Any]:
        # 24 Saatlik Örnek Fiyat Eğrisi ($/MWh): Gece ucuz (20$), Akşam pik (220$)
        prices = [
            25.0, 22.0, 20.0, 18.0, 25.0, 45.0,
            80.0, 110.0, 95.0, 70.0, 60.0, 55.0,
            50.0, 65.0, 90.0, 140.0, 190.0, 240.0,
            260.0, 210.0, 160.0, 110.0, 70.0, 40.0
        ]

        gecikmeler_us: List[float] = []

        for _ in range(self.iterations):
            trader_inst = TeslaAutobidderTrader()
            t0 = time.perf_counter_ns()
            _ = trader_inst.simulate_24h_trading(prices)
            t1 = time.perf_counter_ns()
            gecikmeler_us.append(float(t1 - t0) / 1000.0)

        sim_trader = TeslaAutobidderTrader()
        sim_res = sim_trader.simulate_24h_trading(prices)

        dizi = np.array(gecikmeler_us)
        t_avg_us = float(np.mean(dizi))
        per_hour_us = t_avg_us / 24.0

        return {
            "trading_step_ortalama_us": per_hour_us,
            "sim_24h_ortalama_us": t_avg_us,
            "sim_24h_p99_us": float(np.percentile(dizi, 99)),
            "saniyelik_karar_kapasitesi": int(1e6 / max(per_hour_us, 1e-4)),
            "revenue_usd": sim_res["total_revenue_usd"],
            "cost_usd": sim_res["total_charging_cost_usd"],
            "deg_cost_usd": sim_res["total_degradation_cost_usd"],
            "profit_usd": sim_res["net_profit_usd"],
            "prices": prices,
            "powers": sim_res["powers_mw"],
            "soc_hist": sim_res["soc_history"],
            "gecikmeler": gecikmeler_us[:200]
        }
