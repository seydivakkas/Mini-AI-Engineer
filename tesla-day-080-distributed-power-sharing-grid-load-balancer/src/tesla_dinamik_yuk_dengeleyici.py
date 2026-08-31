r"""
Tesla Dağıtık Güç Dağıtımı ve Dinamik Şebeke Yük Dengeleme Çekirdeği
====================================================================
Bu modül; Tesla Supercharger istasyonlarındaki çoklu şarj istasyonu (Stall)
güç dağıtımını, yerel trafo kapasitesini ($1\text{ MW}$) aşmadan araçların
batarya SoC durumlarına göre orantılı/ters orantılı paylaştıran dinamik
yük dengeleme algoritmasını gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class TeslaDynamicLoadBalancer:
    """
    Tesla Supercharger Dinamik Şebeke Yük Dengeleyicisi.
    """
    def __init__(
        self,
        grid_capacity_kw: float = 1000.0,
        max_stall_power_kw: float = 250.0
    ):
        self.grid_capacity = grid_capacity_kw
        self.max_stall_power = max_stall_power_kw

    def balance_power(self, soc_list: List[float]) -> Dict[str, Any]:
        """
        Bağlı araçların SoC (%) değerlerine göre toplam trafo gücünü paylaştırır.
        Düşük SoC'ye sahip araç daha yüksek güç talep eder ($D_i = 100 - \text{SoC}_i$).
        """
        n_stalls = len(soc_list)
        if n_stalls == 0:
            return {
                "allocated_powers_kw": [],
                "total_allocated_kw": 0.0,
                "grid_headroom_kw": self.grid_capacity,
                "overload_prevented": True
            }

        # 1. Talep Ağırlıkları (Düşük SoC -> Yüksek Talep)
        demands = np.array([max(1.0, 100.0 - s) for s in soc_list], dtype=float)
        total_demand = np.sum(demands)

        # 2. Oransal Ön Dağıtım
        initial_allocations = (demands / total_demand) * self.grid_capacity

        # 3. Stall Limitine (250 kW) Göre Kırpma ve Kalan Gücün Yeniden Dağıtımı
        clamped_allocations = np.minimum(initial_allocations, self.max_stall_power)

        # Kırpma sonrası arta kalan güç
        used_power = np.sum(clamped_allocations)
        residual_power = self.grid_capacity - used_power

        # Eğer hala sınırına ulaşmamış stall'lar varsa artık gücü onlara dağıt
        unclamped_mask = clamped_allocations < self.max_stall_power
        if np.any(unclamped_mask) and residual_power > 0:
            unclamped_demands = demands[unclamped_mask]
            if np.sum(unclamped_demands) > 0:
                bonus_alloc = (unclamped_demands / np.sum(unclamped_demands)) * residual_power
                clamped_allocations[unclamped_mask] = np.minimum(
                    clamped_allocations[unclamped_mask] + bonus_alloc,
                    self.max_stall_power
                )

        final_powers = [float(np.round(p, 2)) for p in clamped_allocations]
        total_used = float(np.sum(final_powers))

        return {
            "stalls_count": n_stalls,
            "allocated_powers_kw": final_powers,
            "total_allocated_kw": total_used,
            "grid_headroom_kw": float(np.round(self.grid_capacity - total_used, 2)),
            "overload_prevented": bool(total_used <= self.grid_capacity + 1e-3)
        }
