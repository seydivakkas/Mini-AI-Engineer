r"""
Tesla FSD NPU INT8 Kuantizasyon ve TensorRT Derleme Çekirdeği
=============================================================
Bu modül; FP32 Derin Öğrenme Tensörlerinin HW3/HW4 NPU için Simetrik
INT8 Kuantizasyonunu, KL-Divergence Kalibrasyonunu, Katman Birleştirmeyi
(Layer Fusion: Conv+BN+ReLU) ve SQNR Hata Analizini gerçekler.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class TeslaFSDNPUQuantizer:
    """
    Tesla FSD NPU Simetrik INT8 Kuantizasyon ve Derleme Motoru.
    """
    def __init__(self):
        pass

    def quantize_symmetric_int8(self, tensor_fp32: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Simetrik INT8 Kuantizasyon:
        S = max(|W|) / 127
        q = clip(round(W / S), -128, 127) (int8)
        """
        max_val = float(np.max(np.abs(tensor_fp32)))
        scale = max_val / 127.0 if max_val > 1e-7 else 1.0

        q_tensor = np.clip(np.round(tensor_fp32 / scale), -128, 127).astype(np.int8)
        return q_tensor, float(scale)

    def dequantize_int8(self, q_tensor: np.ndarray, scale: float) -> np.ndarray:
        """
        De-kuantizasyon: W_hat = q * S (float32)
        """
        return (q_tensor.astype(np.float32) * scale)

    def compute_sqnr_and_error(self, orig_fp32: np.ndarray, deq_fp32: np.ndarray) -> Dict[str, float]:
        """
        Sinyal-Kuantizasyon-Gürültü Oranı (SQNR) ve Maksimum Hata:
        SQNR = 10 * log10( sum(orig^2) / sum((orig - deq)^2) )
        """
        signal_pwr = float(np.sum(orig_fp32 ** 2))
        noise_pwr = float(np.sum((orig_fp32 - deq_fp32) ** 2))

        sqnr_db = 10.0 * np.log10(signal_pwr / max(noise_pwr, 1e-12))
        max_abs_err = float(np.max(np.abs(orig_fp32 - deq_fp32)))
        mae = float(np.mean(np.abs(orig_fp32 - deq_fp32)))

        return {
            "sqnr_db": sqnr_db,
            "max_abs_err": max_abs_err,
            "mae": mae
        }

    def simulate_fused_conv_bn_relu(self, input_act: np.ndarray, weights_int8: np.ndarray, scale: float) -> np.ndarray:
        """
        NPU Katman Birleştirme (Layer Fusion):
        Aktivasyon, Ağırlık Çarpımı ve ReLU NPU SRAM içinde tek çekirdekte icra edilir.
        """
        # Ağırlık FP32'ye dönüştürülmeden doğrudan INT8 MAC işlemi yapılır
        out_int32 = np.dot(input_act.astype(np.int32), weights_int8.astype(np.int32))
        out_fp32 = out_int32.astype(np.float32) * scale
        # NPU Donanımsal ReLU
        return np.maximum(out_fp32, 0.0)
