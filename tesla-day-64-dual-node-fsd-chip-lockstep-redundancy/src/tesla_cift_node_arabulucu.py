r"""
Tesla Çift Düğüm (Dual-Node) FSD Çip Yedekliliği ve Arabulucu Çekirdeği
========================================================================
Bu modül; Tesla FSD HW3/HW4 Çift Bağımsız NPU (Node A ve Node B) Eşzamanlı
Çıkarımını, Karar Arabulucusunu (Hardware Arbiter & Voting Mechanism),
Lockstep Canlılık Doğrulamasını ve Uyuşmazlık Güvenli Durus Mantığını gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import numpy as np


class ArbiterMode(Enum):
    FULL_CONSENSUS = "FULL_CONSENSUS"
    FAILOVER_NODE_A = "FAILOVER_NODE_A"
    FAILOVER_NODE_B = "FAILOVER_NODE_B"
    DISCREPANCY_SAFE_STOP = "DISCREPANCY_SAFE_STOP"
    TOTAL_SYSTEM_FAILURE = "TOTAL_SYSTEM_FAILURE"


class FSDHardwareArbiter:
    """
    Tesla FSD Çift Düğüm Karar Arabulucusu ve Oylama Mekanizması.
    """
    def __init__(
        self,
        max_steer_diff_rad: float = 0.05,   # ~2.86 derece uyuşmazlık eşiği
        max_acc_diff_mps2: float = 0.50     # 0.5 m/s^2 ivme uyuşmazlık eşiği
    ):
        self.max_steer_diff = max_steer_diff_rad
        self.max_acc_diff = max_acc_diff_mps2

    def arbitrate_decision(
        self,
        node_a_steer_rad: float,
        node_b_steer_rad: float,
        node_a_acc_mps2: float,
        node_b_acc_mps2: float,
        node_a_healthy: bool = True,
        node_b_healthy: bool = True
    ) -> Dict[str, Any]:
        """
        İki bağımsız FSD işlemcisinin ürettiği direksiyon ve ivme komutlarını oylar.
        """
        steer_diff = abs(node_a_steer_rad - node_b_steer_rad)
        acc_diff = abs(node_a_acc_mps2 - node_b_acc_mps2)
        is_agreeing = (steer_diff <= self.max_steer_diff) and (acc_diff <= self.max_acc_diff)

        # 1. Senaryo: Her iki düğüm sağlıklı ve uzlaşı içinde
        if node_a_healthy and node_b_healthy and is_agreeing:
            mode = ArbiterMode.FULL_CONSENSUS
            final_steer = (node_a_steer_rad + node_b_steer_rad) / 2.0
            final_acc = (node_a_acc_mps2 + node_b_acc_mps2) / 2.0
            status_desc = "TAM UZLAŞI: Node A ve Node B Çıkarımları Eşleşti"

        # 2. Senaryo: Node A sağlıklı, Node B donanım arızası (Watchdog düştü)
        elif node_a_healthy and not node_b_healthy:
            mode = ArbiterMode.FAILOVER_NODE_A
            final_steer = node_a_steer_rad
            final_acc = node_a_acc_mps2
            status_desc = "FAILOVER: Node B Arızalandı -> Node A Devrede"

        # 3. Senaryo: Node B sağlıklı, Node A donanım arızası
        elif node_b_healthy and not node_a_healthy:
            mode = ArbiterMode.FAILOVER_NODE_B
            final_steer = node_b_steer_rad
            final_acc = node_b_acc_mps2
            status_desc = "FAILOVER: Node A Arızalandı -> Node B Devrede"

        # 4. Senaryo: Her iki düğüm çalışıyor fakat karar ayrışması var (Discrepancy)
        elif node_a_healthy and node_b_healthy and not is_agreeing:
            mode = ArbiterMode.DISCREPANCY_SAFE_STOP
            final_steer = 0.0  # Güvenli düz hat
            final_acc = -1.5   # Yumuşak güvenli yavaşlama
            status_desc = "UYARI: FSD Node A ve Node B Karar Ayrışması! Güvenli Duruş Devrede."

        # 5. Senaryo: Her iki düğüm de çöktü
        else:
            mode = ArbiterMode.TOTAL_SYSTEM_FAILURE
            final_steer = 0.0
            final_acc = -3.0   # Acil durum freni
            status_desc = "KRİTİK: Çift Çip Çökmesi! İkincil Mikrodenetleyici Devrede."

        return {
            "arbiter_mode": mode.value,
            "applied_steering_rad": float(final_steer),
            "applied_acc_mps2": float(final_acc),
            "steer_diff_rad": float(steer_diff),
            "acc_diff_mps2": float(acc_diff),
            "status_desc": status_desc,
            "is_nominal": bool(mode == ArbiterMode.FULL_CONSENSUS),
            "is_emergency_stop": bool(mode in [ArbiterMode.DISCREPANCY_SAFE_STOP, ArbiterMode.TOTAL_SYSTEM_FAILURE])
        }
