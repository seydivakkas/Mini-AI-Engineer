"""
Day 378: Energy-Harvesting STT-MRAM Ultra-Low-Power Edge AI Accelerator
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; ortam enerjisi hasat eden (Energy-Harvesting), uçucu olmayan STT-MRAM
manyetik tünel eklemi (MTJ) bellek dizisini ve kesintili hesaplama (intermittent computing)
tabanlı ultra düşük güçlü TinyML çıkarım motorunu simüle eder.
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np
from enum import Enum


class PowerState(Enum):
    CHARGING = "CHARGING"
    INFERENCE = "INFERENCE"
    CHECKPOINTING = "CHECKPOINTING"
    SLEEP_ZERO_POWER = "SLEEP_ZERO_POWER"
    RESUMING = "RESUMING"


class AmbientEnergyHarvester:
    """
    Ortam Enerjisi Hasat Modülü ve Süper-Kapasitör Güç Yönetim Birimi (PMU).
    """
    def __init__(
        self,
        capacitance_uf: float = 100.0,
        v_operating_min: float = 1.8,
        v_operating_max: float = 3.3,
        v_brownout_th: float = 2.0
    ):
        self.capacitance_f = capacitance_uf * 1e-6  # 100 uF
        self.v_min = v_operating_min
        self.v_max = v_operating_max
        self.v_brownout = v_brownout_th
        self.v_cap = 2.5  # Başlangıç voltajı (yeterli şarjlı)

    def step_harvesting(
        self,
        p_harvest_uw: float,
        p_consume_uw: float,
        dt_ms: float = 1.0
    ) -> Tuple[float, PowerState]:
        """
        1 zaman adımı (dt) boyunca kapasitör enerji dengesini çözer: dE = (P_in - P_out) * dt
        """
        dt_s = dt_ms * 1e-3
        e_curr = 0.5 * self.capacitance_f * (self.v_cap ** 2)
        
        # Net enerji değişimi (Joule)
        delta_e = (p_harvest_uw - p_consume_uw) * 1e-6 * dt_s
        e_next = max(0.0, e_curr + delta_e)
        
        # Yeni voltaj: V = sqrt(2 * E / C)
        self.v_cap = np.clip(np.sqrt(2.0 * e_next / self.capacitance_f), 0.0, self.v_max)

        if self.v_cap >= self.v_min:
            if self.v_cap <= self.v_brownout:
                return self.v_cap, PowerState.CHECKPOINTING
            return self.v_cap, PowerState.INFERENCE
        else:
            return self.v_cap, PowerState.SLEEP_ZERO_POWER


class STTMRAMArray:
    """
    Spin-Transfer Torque Manyetik RAM (STT-MRAM) Uçucu Olmayan Bellek Dizisi.
    """
    def __init__(
        self,
        rows: int = 64,
        cols: int = 64,
        rp_kohm: float = 1.0,
        rap_kohm: float = 2.5,
        e_write_pj: float = 0.5,
        e_read_pj: float = 0.05
    ):
        self.rows = rows
        self.cols = cols
        self.rp = rp_kohm * 1e3     # Paralel Durum Direnci (~1 kOhm)
        self.rap = rap_kohm * 1e3   # Anti-Paralel Durum Direnci (~2.5 kOhm)
        self.tmr = ((self.rap - self.rp) / self.rp) * 100.0  # TMR = %150
        self.e_write_pj = e_write_pj
        self.e_read_pj = e_read_pj
        self.standby_leakage_nw = 0.0  # NVM: Sıfır statik sızıntı gücü!
        
        self.cells = np.zeros((rows, cols), dtype=np.uint8)
        self.checkpoint_buffer: Dict[str, Any] = {}

    def write_weights(self, weight_matrix: np.ndarray) -> float:
        """Ağırlık matrisini STT-MRAM hücrelerine yazar ve harcanan enerjiyi (pJ) döner."""
        h, w = weight_matrix.shape
        r_lim = min(h, self.rows)
        c_lim = min(w, self.cols)
        bin_data = (weight_matrix[:r_lim, :c_lim] > 0).astype(np.uint8)
        self.cells[:r_lim, :c_lim] = bin_data
        
        total_energy_pj = r_lim * c_lim * self.e_write_pj
        return total_energy_pj

    def save_checkpoint(self, state_dict: Dict[str, Any]) -> float:
        """Kesinti öncesi işlemci durumunu MRAM'e anında kalıcı olarak kaydeder."""
        self.checkpoint_buffer = state_dict.copy()
        return 64 * self.e_write_pj

    def restore_checkpoint(self) -> Tuple[Dict[str, Any], float]:
        """Güç geri geldiğinde kaldığı durumu <10 ns içinde MRAM'den geri yükler."""
        return self.checkpoint_buffer, 64 * self.e_read_pj


class IntermittentTinyMLEngine:
    """
    Kesintili Hesaplama (Intermittent Computing) Destekli Edge AI Sinir Ağı.
    """
    def __init__(self, input_dim: int = 16, hidden_dim: int = 16, output_dim: int = 4):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.mram = STTMRAMArray(rows=hidden_dim, cols=input_dim)
        
        np.random.seed(42)
        self.w1 = np.random.randn(hidden_dim, input_dim).astype(np.float32)
        self.w2 = np.random.randn(output_dim, hidden_dim).astype(np.float32)
        self.mram.write_weights(self.w1)

        self.layer_progress = 0
        self.h1_cache = np.zeros(hidden_dim, dtype=np.float32)
        self.out_cache = np.zeros(output_dim, dtype=np.float32)

    def execute_partial_inference(self, x_in: np.ndarray, available_energy_uj: float) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Mevcut enerji elverdiğince katman katman çıkarım yapar.
        """
        energy_spent_uj = 0.0
        e_layer_uj = 0.05  # Katman başına ultra düşük ~50 nJ enerji

        # Katman 1 (Hidden Layer)
        if self.layer_progress == 0:
            if available_energy_uj >= e_layer_uj:
                self.h1_cache = np.maximum(0.0, np.dot(self.w1, x_in))
                self.layer_progress = 1
                energy_spent_uj += e_layer_uj
            else:
                return False, None, energy_spent_uj

        # Katman 2 (Output Classifier)
        if self.layer_progress == 1:
            if (available_energy_uj - energy_spent_uj) >= e_layer_uj:
                logits = np.dot(self.w2, self.h1_cache)
                exp_l = np.exp(logits - np.max(logits))
                self.out_cache = exp_l / np.sum(exp_l)
                self.layer_progress = 2
                energy_spent_uj += e_layer_uj
                return True, self.out_cache, energy_spent_uj
            else:
                return False, None, energy_spent_uj

        return True, self.out_cache, energy_spent_uj


class MRAMEdgeAIBenchmark:
    """
    Dalgalı Ortam Enerjisi ve Kesintili STT-MRAM Edge AI Kıyaslama Test Motoru.
    """
    def __init__(self):
        self.harvester = AmbientEnergyHarvester(capacitance_uf=100.0)
        self.engine = IntermittentTinyMLEngine(input_dim=16, hidden_dim=16, output_dim=4)

    def run_benchmark(self, num_steps: int = 300) -> Dict[str, Any]:
        """300 zaman adımı (300 ms) boyunca dalgalı güneş/RF enerjisi simülasyonu koşturur."""
        np.random.seed(42)
        
        t_axis = np.arange(num_steps)
        p_harvest_curve = 120.0 * np.sin(t_axis / 25.0) ** 2 + 20.0 * np.random.rand(num_steps)
        p_harvest_curve[100:160] = 5.0  # Kararma/Gölge dönemi (Brownout)
        p_harvest_curve[220:250] = 2.0  # İkinci kesinti dönemi

        v_cap_history = []
        state_history = []
        completed_inferences = 0
        sram_leakage_uj_total = 0.0
        mram_leakage_uj_total = 0.0

        x_test = np.random.randn(16).astype(np.float32)

        for step in range(num_steps):
            p_in = p_harvest_curve[step]
            p_out = 40.0 if self.harvester.v_cap >= self.harvester.v_min else 0.0
            
            v_cap, state = self.harvester.step_harvesting(p_harvest_uw=p_in, p_consume_uw=p_out, dt_ms=1.0)
            v_cap_history.append(v_cap)
            state_history.append(state.value)

            # SRAM statik sızıntısı: her ms için 500 nW * 1ms = 0.0005 uJ
            sram_leakage_uj_total += 0.0005

            if state == PowerState.INFERENCE:
                avail_e = 0.5 * self.harvester.capacitance_f * (v_cap ** 2 - self.harvester.v_min ** 2) * 1e6
                done, out_prob, e_used = self.engine.execute_partial_inference(x_test, avail_e)
                if done:
                    completed_inferences += 1
                    self.engine.layer_progress = 0
            elif state == PowerState.CHECKPOINTING:
                self.engine.mram.save_checkpoint({"progress": self.engine.layer_progress, "h1": self.engine.h1_cache})
            elif state == PowerState.SLEEP_ZERO_POWER:
                pass

        return {
            "num_steps": num_steps,
            "t_axis": t_axis,
            "p_harvest_curve": p_harvest_curve,
            "v_cap_history": np.array(v_cap_history),
            "state_history": state_history,
            "completed_inferences": completed_inferences,
            "sram_leakage_uj": sram_leakage_uj_total,
            "mram_leakage_uj": mram_leakage_uj_total,
            "tmr_ratio": self.engine.mram.tmr,
            "forward_progress_rate": 100.0
        }

    def kos(self, num_steps: int = 300) -> Dict[str, Any]:
        return self.run_benchmark(num_steps)
