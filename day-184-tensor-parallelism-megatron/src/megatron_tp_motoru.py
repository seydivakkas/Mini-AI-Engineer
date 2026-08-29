"""
Megatron-LM Tensor Parallelism (TP) Çekirdek Motoru (Day 184 - FAZ 10).
Shoeybi et al. (2019) Megatron-LM: ColumnParallelLinear, RowParallelLinear ve f/g Autograd Operatörleri.
"""

from typing import List, Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
from torch.autograd import Function


class _CopyToModelParallelRegion(Function):
    """
    f Operatörü:
    - İleri Geçiş (Forward): Giriş tensörünü tüm TP rank'lerine çoğaltır (Identity).
    - Geri Geçiş (Backward): Farklı TP rank'lerinden gelen gradyanları toplar (All-Reduce: sum(grad)).
    """

    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, world_size: int) -> torch.Tensor:
        ctx.world_size = world_size
        return input_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Gerçek çoklu GPU kümesinde torch.distributed.all_reduce(grad_output) çağrılır
        # Simülasyon ortamında grad_output doğrudan iletilir
        return grad_output, None


class _ReduceFromModelParallelRegion(Function):
    """
    g Operatörü:
    - İleri Geçiş (Forward): Farklı TP rank'lerinden gelen kısmi sonuçları toplar (All-Reduce: sum(outputs)).
    - Geri Geçiş (Backward): Çıktı gradyanını tüm TP rank'lerine kopyalar (Identity).
    """

    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, world_size: int) -> torch.Tensor:
        ctx.world_size = world_size
        return input_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


def copy_to_tensor_model_parallel_region(input_tensor: torch.Tensor, world_size: int = 1) -> torch.Tensor:
    return _CopyToModelParallelRegion.apply(input_tensor, world_size)


def reduce_from_tensor_model_parallel_region(input_tensor: torch.Tensor, world_size: int = 1) -> torch.Tensor:
    return _ReduceFromModelParallelRegion.apply(input_tensor, world_size)


class ColumnParallelLinear(nn.Module):
    """
    Sütun Paralel Doğrusal Katman (ColumnParallelLinear).
    Ağırlık matrisi sütun ekseninde K parçaya bölünür: W_i in R^{D_in x (D_out / K)}
    - İleri Geçiş: İletişim GEREKTİRMEZ (Y_i = X @ W_i). Aktivasyon fonksiyonu (GeLU) yerel hesaplanabilir.
    - Geri Geçiş: Giriş gradyanı dX için All-Reduce gerektirir (f operatörü ile).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_world_size: int = 2,
        tp_rank: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        assert out_features % tp_world_size == 0, f"out_features ({out_features}) tp_world_size'a ({tp_world_size}) tam bölünmelidir."
        self.in_features = in_features
        self.out_features = out_features
        self.tp_world_size = tp_world_size
        self.tp_rank = tp_rank

        self.split_out_features = out_features // tp_world_size

        # Bu rank'in 1/K'lık sütun ağırlık dilimi
        self.weight = nn.Parameter(torch.empty(self.split_out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.split_out_features))
        else:
            self.register_parameter("bias", None)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        1. f operatörü ile girişi kopyala
        2. Yerel matris çarpımı yap (Y_i = X @ W_i.T + b_i)
        """
        input_parallel = copy_to_tensor_model_parallel_region(x, self.tp_world_size)
        output_parallel = nn.functional.linear(input_parallel, self.weight, self.bias)
        return output_parallel


class RowParallelLinear(nn.Module):
    """
    Satır Paralel Doğrusal Katman (RowParallelLinear).
    Ağırlık matrisi satır ekseninde K parçaya bölünür: W_i in R^{(D_in / K) x D_out}
    - Giriş tensörü X_i in R^{B x (D_in / K)} dilimlidir.
    - İleri Geçiş: Çıktıları toplamak için All-Reduce GEREKTİRİR (Y = sum(X_i @ W_i) + b).
    - Geri Geçiş: İletişim GEREKTİRMEZ (g operatörü ile).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_world_size: int = 2,
        tp_rank: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        assert in_features % tp_world_size == 0, f"in_features ({in_features}) tp_world_size'a ({tp_world_size}) tam bölünmelidir."
        self.in_features = in_features
        self.out_features = out_features
        self.tp_world_size = tp_world_size
        self.tp_rank = tp_rank

        self.split_in_features = in_features // tp_world_size

        # Bu rank'in 1/K'lık satır ağırlık dilimi
        self.weight = nn.Parameter(torch.empty(out_features, self.split_in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x_parallel: torch.Tensor, all_rank_outputs: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        1. Yerel kısmi matris çarpımı yap (Y_i = X_i @ W_i.T)
        2. Tüm TP rank'lerinin çıktılarını All-Reduce ile topla
        3. Bias ekle
        """
        output_parallel = nn.functional.linear(x_parallel, self.weight, None)

        if all_rank_outputs is not None:
            # Simülasyon: N rank'in kısmi çıktısını topla (All-Reduce sum)
            reduced_output = torch.stack(all_rank_outputs).sum(dim=0)
        else:
            reduced_output = output_parallel

        reduced_output = reduce_from_tensor_model_parallel_region(reduced_output, self.tp_world_size)

        if self.bias is not None:
            reduced_output = reduced_output + self.bias

        return reduced_output
