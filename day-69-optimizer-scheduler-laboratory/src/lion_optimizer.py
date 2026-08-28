"""
Lion (EvoLved Sign Momentum) Optimizer
======================================
Google Brain tarafindan otomatik program arama (AutoML) ile kesfedilen,
ikinci moment (v_t) matrisini tutmayarak %50 bellek tasarrufu saglayan ve
sign() operatoru ile kararlı adimlar atan yuksek verimli optimizer.

Referans: Chen et al., 'Symbolic Discovery of Optimization Algorithms', 2023.
"""

from typing import Tuple, List, Optional, Callable, Dict, Any, Iterable
import torch
from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
    """
    Lion (EvoLved Sign Momentum) Optimizasyon Algoritmasi.

    Args:
        params (Iterable): Optimize edilecek model parametreleri
        lr (float): Ogrenme orani (AdamW'ye kiyasla genelde 3x-10x daha kucuk secilmelidir, or. 1e-4)
        betas (Tuple[float, float]): (beta1, beta2) momentum katsayilari (Varsayilan: 0.9, 0.99)
        weight_decay (float): Ayrıştırılmış ağırlık azaltma katsayisi (Decoupled Weight Decay)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Gecersiz ogrenme orani: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Gecersiz beta1 parametresi: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Gecersiz beta2 parametresi: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Gecersiz weight_decay parametresi: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Optimizasyon adımını gerçekleştirir."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Lion optimizer sparse gradyanlari desteklememektedir.")

                state = self.state[p]

                # Durum ilklendirmesi (Sadece 1. moment / momentum tensörü)
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]

                # 1. Adım: Ayrıştırılmış Ağırlık Azaltma (Decoupled Weight Decay)
                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                # 2. Adım: Sign Momentum ile Güncelleme Yönü (c_t) Hesabı
                # c_t = sign(beta1 * m_{t-1} + (1 - beta1) * g_t)
                update = exp_avg.mul(beta1).add(grad, alpha=1.0 - beta1).sign()

                # 3. Adım: Parametre Güncellemesi
                # theta_{t+1} = theta_t - lr * c_t
                p.data.add_(update, alpha=-lr)

                # 4. Adım: Momentum Durumunun Güncellenmesi
                # m_t = beta2 * m_{t-1} + (1 - beta2) * g_t
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)

        return loss
