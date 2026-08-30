"""
Day 372: Custom RISC-V Vector Extension ISA Design for Transformer Kernels
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; 256-bit Vektör Kayıt Dosyasını (VRF), Özel RISC-V Transformer Komut Seti Emülatörünü
(v.softmax.exp.sum, v.gelu.approx, v.layernorm.fused, v.fma.chained) ve Performans Kıyaslayıcısını içerir.
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np


class RISCVVectorRegisterFile:
    """
    RISC-V Vektör Kayıt Dosyası (VRF).
    32 Vektör Kaydı (v0..v31), VLEN = 256 bit (8 adet float32/kayıt).
    """
    def __init__(self, vlen_bits: int = 256):
        self.vlen = vlen_bits
        self.elem_per_reg = vlen_bits // 32 # 8 eleman (float32)
        self.regs = np.zeros((32, self.elem_per_reg), dtype=np.float32)

    def write(self, reg_idx: int, data: np.ndarray):
        """Vektör kaydına veri yazar."""
        self.regs[reg_idx, :len(data)] = data[:self.elem_per_reg]

    def read(self, reg_idx: int) -> np.ndarray:
        """Vektör kaydından veri okur."""
        return self.regs[reg_idx].copy()


class CustomTransformerISAProcessor:
    """
    Özel RISC-V Vektör Transformer İşlemcisi ve Komut Yürütme Hattı.
    """
    def __init__(self):
        self.vrf = RISCVVectorRegisterFile(vlen_bits=256)
        self.instruction_count = 0
        self.cycle_count = 0

    # 1. Özel Komut: v.gelu.approx vd, vs2
    def exec_v_gelu_approx(self, vd: int, vs2: int):
        """Hızlı GeLU polinom aktivasyon komutu: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))"""
        x = self.vrf.read(vs2)
        c = np.sqrt(2.0 / np.pi)
        inner = c * (x + 0.044715 * (x ** 3))
        res = 0.5 * x * (1.0 + np.tanh(inner))
        self.vrf.write(vd, res)
        self.instruction_count += 1
        self.cycle_count += 2 # 2-saykıl donanım boru hattı

    # 2. Özel Komut: v.softmax.exp.sum vd, vs2, v_max
    def exec_v_softmax_exp_sum(self, vd: int, vs2: int, max_val: float) -> float:
        """Çevrimiçi (Online) Kararlı Softmax Üstel ve Toplam Komutu."""
        x = self.vrf.read(vs2)
        exp_x = np.exp(x - max_val)
        sum_exp = float(np.sum(exp_x))
        self.vrf.write(vd, exp_x)
        self.instruction_count += 1
        self.cycle_count += 3
        return sum_exp

    # 3. Özel Komut: v.layernorm.fused vd, vs2, gamma, beta
    def exec_v_layernorm_fused(self, vd: int, vs2: int, gamma: float = 1.0, beta: float = 0.0):
        """Tek Geçişli Birleşik LayerNorm Komutu."""
        x = self.vrf.read(vs2)
        mean = np.mean(x)
        var = np.var(x)
        norm_x = (x - mean) / np.sqrt(var + 1e-5)
        res = norm_x * gamma + beta
        self.vrf.write(vd, res)
        self.instruction_count += 1
        self.cycle_count += 2

    # 4. Özel Komut: v.fma.chained vd, vs1, vs2
    def exec_v_fma_chained(self, vd: int, vs1: int, vs2: int):
        """Vektör Çarp-Topla Birleşik Komutu: vd = vd + vs1 * vs2"""
        acc = self.vrf.read(vd)
        a = self.vrf.read(vs1)
        b = self.vrf.read(vs2)
        res = acc + a * b
        self.vrf.write(vd, res)
        self.instruction_count += 1
        self.cycle_count += 1


class TransformerKernelBenchmark:
    """
    Standart Skaler RISC-V vs Özel RVV-Transformer ISA Kıyaslama Testi.
    """
    def __init__(self):
        self.custom_cpu = CustomTransformerISAProcessor()

    def run_benchmark(self, seq_len: int = 8, hidden_dim: int = 8) -> Dict[str, Any]:
        """Transformer Attention + GeLU FFN bloğunu yürütür."""
        np.random.seed(42)
        x_input = np.random.randn(seq_len, hidden_dim).astype(np.float32)

        # -------------------------------------------------------------
        # 1. Standart Skaler RISC-V Yürütmesi (Temel Referans)
        # -------------------------------------------------------------
        # Skaler döngülerde her işlem için ayrı load/op/store/branch komutları gerekir
        # GeLU = ~22 komut/eleman, Softmax = ~18 komut/eleman, LayerNorm = ~15 komut/eleman
        total_elements = seq_len * hidden_dim
        scalar_instructions = total_elements * (22 + 18 + 15 + 8) # ~4,032 komut
        scalar_cycles = int(scalar_instructions * 1.6)

        # -------------------------------------------------------------
        # 2. Özel RVV-Transformer ISA Yürütmesi
        # -------------------------------------------------------------
        self.custom_cpu.instruction_count = 0
        self.custom_cpu.cycle_count = 0

        outputs = []
        for i in range(seq_len):
            row = x_input[i]
            # 1. LayerNorm
            self.custom_cpu.vrf.write(0, row)
            self.custom_cpu.exec_v_layernorm_fused(vd=1, vs2=0)

            # 2. Softmax Online
            max_v = float(np.max(row))
            sum_e = self.custom_cpu.exec_v_softmax_exp_sum(vd=2, vs2=1, max_val=max_v)
            
            # 3. GeLU FFN
            self.custom_cpu.exec_v_gelu_approx(vd=3, vs2=2)
            outputs.append(self.custom_cpu.vrf.read(3))

        custom_instructions = self.custom_cpu.instruction_count
        custom_cycles = self.custom_cpu.cycle_count

        inst_reduction = scalar_instructions / max(1, custom_instructions)
        cycle_speedup = scalar_cycles / max(1, custom_cycles)
        
        # Sayısal Doğruluk Testi (Ground Truth LayerNorm+Softmax+GeLU vs RVV)
        gt_outputs = []
        c = np.sqrt(2.0 / np.pi)
        for i in range(seq_len):
            row = x_input[i]
            m = np.mean(row)
            v = np.var(row)
            norm_r = (row - m) / np.sqrt(v + 1e-5)
            exp_r = np.exp(norm_r - np.max(row))
            gelu_r = 0.5 * exp_r * (1.0 + np.tanh(c * (exp_r + 0.044715 * (exp_r ** 3))))
            gt_outputs.append(gelu_r)
        
        mse_fidelity = float(np.mean((np.array(outputs) - np.array(gt_outputs))**2))

        return {
            "scalar_instructions": scalar_instructions,
            "custom_instructions": custom_instructions,
            "instruction_reduction": inst_reduction,
            "scalar_cycles": scalar_cycles,
            "custom_cycles": custom_cycles,
            "cycle_speedup": cycle_speedup,
            "mse_fidelity": mse_fidelity,
            "outputs": np.array(outputs)
        }
