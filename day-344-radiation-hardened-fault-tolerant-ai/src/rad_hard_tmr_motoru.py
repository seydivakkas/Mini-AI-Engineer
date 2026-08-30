"""
Day 344: Radiation-Hardened Fault-Tolerant Edge AI Inference with Triple Modular Redundancy (TMR)
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; Kozmik Radyasyon SEU (Single Event Upset) Bit-Flip Enjektörünü,
Üçlü Modüler Yedekli (TMR) Çıkarım Çekirdeğini, Çoğunluk Oylayıcısını (Majority Voter)
ve Otomatik Bellek Temizleme/Kurtarma (ECC Memory Scrubber) Sistemini içerir.
"""

from typing import Tuple, Dict, Any, List, Optional
import struct
import numpy as np


class RadiationSEUInjector:
    """
    Kozmik Radyasyon ve Güneş Parçacığı SEU (Single Event Upset) Bit-Flip Simülatörü.
    IEEE-754 32-bit kayan nokta veya INT8 ağırlık tensörlerinde rastgele 1-bit çevirir (0 <-> 1).
    """
    def __init__(self, bit_flip_prob: float = 0.05):
        self.bit_flip_prob = bit_flip_prob

    def flip_float32_bits(self, val: float, bit_index: int = 30) -> float:
        """Kayan noktalı sayının (FP32) belirtilen bitini (örn: Exponent biti) çevirir."""
        packed = struct.pack('!f', val)
        int_val = struct.unpack('!I', packed)[0]
        flipped_int = int_val ^ (1 << bit_index)
        flipped_bytes = struct.pack('!I', flipped_int)
        return struct.unpack('!f', flipped_bytes)[0]

    def inject_seu_to_weights(self, weights: np.ndarray, num_flips: int = 5) -> np.ndarray:
        """Ağırlık matrisine rastgele SEU radyasyon bit-flip enjeksiyonu uygular."""
        corrupted = weights.copy()
        flat = corrupted.flatten()
        indices = np.random.choice(len(flat), size=min(num_flips, len(flat)), replace=False)
        
        for idx in indices:
            # Exponent bitini (bit 30) veya işaret bitini çevir
            flat[idx] = self.flip_float32_bits(float(flat[idx]), bit_index=30)
            
        return flat.reshape(weights.shape)


class TMRInferenceCore:
    """
    Üçlü Modüler Yedeklilik (Triple Modular Redundancy - TMR) Çıkarım Motoru.
    Üç bağımsız AI çekirdeğini (Core A, Core B, Core C) paralel çalıştırıp 2/3 Çoğunluk Oylaması yapar.
    """
    def __init__(self, golden_weights: np.ndarray):
        self.golden_weights = golden_weights.copy()
        # 3 Bağımsız Donanım Çekirdeği
        self.core_a_weights = golden_weights.copy()
        self.core_b_weights = golden_weights.copy()
        self.core_c_weights = golden_weights.copy()

    def run_single_core(self, weights: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Tek bir AI çekirdeğinde ileri yayılım (Forward pass: y = Softmax(W * x))."""
        logits = np.dot(x, weights)
        # Kararlı Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return probs

    def tmr_inference(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Üç çekirdekte çıkarım yapar ve 2-out-of-3 Çoğunluk Oylaması (Majority Voting) uygular.
        """
        out_a = self.run_single_core(self.core_a_weights, x)
        out_b = self.run_single_core(self.core_b_weights, x)
        out_c = self.run_single_core(self.core_c_weights, x)

        pred_a = int(np.argmax(out_a, axis=-1).flatten()[0])
        pred_b = int(np.argmax(out_b, axis=-1).flatten()[0])
        pred_c = int(np.argmax(out_c, axis=-1).flatten()[0])

        # Çoğunluk Oylaması (Majority Voting)
        preds = [pred_a, pred_b, pred_c]
        counts = {p: preds.count(p) for p in preds}
        majority_pred = max(counts, key=counts.get)
        
        faulty_cores = []
        if pred_a != majority_pred:
            faulty_cores.append("Core_A")
        if pred_b != majority_pred:
            faulty_cores.append("Core_B")
        if pred_c != majority_pred:
            faulty_cores.append("Core_C")

        has_fault = len(faulty_cores) > 0
        consensus_ratio = counts[majority_pred] / 3.0

        # Çıktı olasılık vektörü (Sağlam çekirdeklerin medyanı)
        valid_outputs = []
        if "Core_A" not in faulty_cores: valid_outputs.append(out_a)
        if "Core_B" not in faulty_cores: valid_outputs.append(out_b)
        if "Core_C" not in faulty_cores: valid_outputs.append(out_c)

        if len(valid_outputs) > 0:
            final_probs = np.mean(valid_outputs, axis=0)
        else:
            final_probs = out_a

        return {
            "majority_pred": majority_pred,
            "final_probs": final_probs,
            "individual_preds": {"Core_A": pred_a, "Core_B": pred_b, "Core_C": pred_c},
            "has_fault": has_fault,
            "faulty_cores": faulty_cores,
            "consensus_ratio": consensus_ratio,
        }


class AutonomousMemoryScrubber:
    """
    Otonom Bellek Temizleme ve Kurtarma (ECC Scrubbing & Golden ROM Flash) Motoru.
    Bozulan çekirdeğin ağırlıklarını Golden ROM kopyasından anında onarır.
    """
    def __init__(self, tmr_core: TMRInferenceCore):
        self.tmr = tmr_core
        self.repair_count = 0

    def scrub_and_repair(self, faulty_cores: List[str]):
        """Bozuk tespit edilen çekirdekleri Golden ROM ile onarır."""
        for core_name in faulty_cores:
            if core_name == "Core_A":
                self.tmr.core_a_weights = self.tmr.golden_weights.copy()
                self.repair_count += 1
            elif core_name == "Core_B":
                self.tmr.core_b_weights = self.tmr.golden_weights.copy()
                self.repair_count += 1
            elif core_name == "Core_C":
                self.tmr.core_c_weights = self.tmr.golden_weights.copy()
                self.repair_count += 1


class FaultTolerantAIEngine:
    """
    Radyasyona Dayanıklı Uçtan Uca Otonom Edge AI Çıkarım Sistemi.
    """
    def __init__(self, input_dim: int = 16, num_classes: int = 4):
        np.random.seed(42)
        golden_w = np.random.normal(0, 0.5, (input_dim, num_classes))
        self.tmr_core = TMRInferenceCore(golden_w)
        self.scrubber = AutonomousMemoryScrubber(self.tmr_core)
        self.injector = RadiationSEUInjector()

    def process_telemetry_sample(self, x: np.ndarray, inject_radiation: bool = False, target_core: str = "Core_B") -> Dict[str, Any]:
        """
        Gelen uzay telemetri verisini TMR altında işler, radyasyon varsa onarır.
        """
        if inject_radiation:
            if target_core == "Core_A":
                self.tmr_core.core_a_weights = self.injector.inject_seu_to_weights(self.tmr_core.core_a_weights, num_flips=3)
            elif target_core == "Core_B":
                self.tmr_core.core_b_weights = self.injector.inject_seu_to_weights(self.tmr_core.core_b_weights, num_flips=3)
            elif target_core == "Core_C":
                self.tmr_core.core_c_weights = self.injector.inject_seu_to_weights(self.tmr_core.core_c_weights, num_flips=3)

        result = self.tmr_core.tmr_inference(x)

        if result["has_fault"]:
            # Anında Golden ROM ile onar
            self.scrubber.scrub_and_repair(result["faulty_cores"])
            result["repaired"] = True
        else:
            result["repaired"] = False

        return result
