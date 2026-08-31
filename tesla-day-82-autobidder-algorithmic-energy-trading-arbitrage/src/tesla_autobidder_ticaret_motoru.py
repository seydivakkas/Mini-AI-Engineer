r"""
Tesla Autobidder Algoritmik Enerji Ticareti ve Arbitraj Çekirdeği
==================================================================
Bu modül; Tesla Autobidder yapay zeka tabanlı otonom enerji tüccarı platformunu,
elektrik toptan spot piyasası (Day-Ahead / Real-Time) fiyat sinyallerini,
batarya döngü yıpranma maliyetini ($40\text{ \$/MWh}$) ve kar maksimizasyonunu
gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class TeslaAutobidderTrader:
    """
    Tesla Autobidder Otonom Enerji Ticareti ve Arbitraj Motoru.
    """
    def __init__(
        self,
        capacity_mwh: float = 3.9,
        max_power_mw: float = 1.95,
        degradation_cost_usd_per_mwh: float = 40.0,
        high_price_threshold_usd: float = 150.0,
        low_price_threshold_usd: float = 30.0,
        initial_soc_pct: float = 50.0
    ):
        self.capacity_mwh = capacity_mwh
        self.max_power_mw = max_power_mw
        self.deg_cost = degradation_cost_usd_per_mwh
        self.high_price = high_price_threshold_usd
        self.low_price = low_price_threshold_usd

        self.soc_pct = initial_soc_pct

    def decide_trade_action(
        self,
        spot_price_usd_mwh: float,
        soc_pct: float
    ) -> Tuple[str, float]:
        """
        Spot fiyat ve SoC durumuna göre alım (şarj), satım (deşarj) veya bekleme kararı verir.
        """
        # 1. Yüksek Fiyat: Şebekeye Satış (Deşarj)
        if spot_price_usd_mwh > self.high_price and soc_pct > 20.0:
            # Kar marjı kontrolü: Fiyat > Yıpranma Maliyeti
            if spot_price_usd_mwh > self.deg_cost:
                return "DISCHARGE_TO_GRID (SELL)", self.max_power_mw

        # 2. Düşük Fiyat: Şebekeden Alış (Şarj)
        elif spot_price_usd_mwh < self.low_price and soc_pct < 95.0:
            return "CHARGE_FROM_GRID (BUY)", -self.max_power_mw

        # 3. Orta Fiyat: Bekleme veya Frekans Dengeleme
        return "STANDBY_OPTIMAL", 0.0

    def simulate_24h_trading(self, hourly_spot_prices: List[float]) -> Dict[str, Any]:
        """
        24 saatlik spot fiyat profili üzerinden kümülatif gelir ve net kar simülasyonu.
        """
        total_revenue_usd = 0.0
        total_charging_cost_usd = 0.0
        total_degradation_usd = 0.0
        actions = []
        powers_mw = []
        soc_history = [self.soc_pct]

        for price in hourly_spot_prices:
            action, p_mw = self.decide_trade_action(price, self.soc_pct)
            actions.append(action)
            powers_mw.append(p_mw)

            energy_mwh = abs(p_mw) * 1.0  # 1 saatlik adım

            if p_mw > 0:  # Satış (Deşarj)
                revenue = energy_mwh * price
                deg = energy_mwh * self.deg_cost
                total_revenue_usd += revenue
                total_degradation_usd += deg
                # SoC Düşüşü
                delta_soc = (energy_mwh / self.capacity_mwh) * 100.0
                self.soc_pct = max(10.0, self.soc_pct - delta_soc)
            elif p_mw < 0:  # Alış (Şarj)
                cost = energy_mwh * price
                total_charging_cost_usd += cost
                # SoC Artışı
                delta_soc = (energy_mwh / self.capacity_mwh) * 100.0
                self.soc_pct = min(95.0, self.soc_pct + delta_soc)

            soc_history.append(self.soc_pct)

        net_profit_usd = total_revenue_usd - total_charging_cost_usd - total_degradation_usd

        return {
            "hours": len(hourly_spot_prices),
            "total_revenue_usd": float(np.round(total_revenue_usd, 2)),
            "total_charging_cost_usd": float(np.round(total_charging_cost_usd, 2)),
            "total_degradation_cost_usd": float(np.round(total_degradation_usd, 2)),
            "net_profit_usd": float(np.round(net_profit_usd, 2)),
            "final_soc_pct": float(np.round(self.soc_pct, 2)),
            "actions": actions,
            "powers_mw": powers_mw,
            "soc_history": soc_history
        }
