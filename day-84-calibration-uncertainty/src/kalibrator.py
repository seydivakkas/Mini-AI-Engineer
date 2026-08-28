"""
Post-Hoc Sıcaklık Ölçekleme (Temperature Scaling) Kalibratörü
------------------------------------------------------------
Guo et al. (2017) "On Calibration of Modern Neural Networks" algoritması.
Model ağırlıklarını dondurarak tekil bir skaler sıcaklık parametresini (T > 0)
L-BFGS optimizasyonu ile Cross-Entropy (NLL) kaybını minimize ederek bulan kalibratör.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Tuple, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS


class SicaklikKalibratoru(nn.Module):
    """
    Sıcaklık Ölçekleme (Temperature Scaling) Modülü
    q_i = Softmax(z_i / T)
    """
    def __init__(self, baslangic_sicaklik: float = 1.5):
        super().__init__()
        self.sicaklik = nn.Parameter(torch.ones(1) * baslangic_sicaklik)

    def forward(self, logitler: torch.Tensor) -> torch.Tensor:
        """
        Logitleri sıcaklık katsayısına böler.
        """
        # Sıcaklığın her zaman pozitif ve güvenli bir tabanda kalmasını sağla
        t = torch.clamp(self.sicaklik, min=0.05, max=50.0)
        return logitler / t

    def kalibre_et(
        self,
        val_logitler: torch.Tensor,
        val_etiketler: torch.Tensor,
        max_iter: int = 50,
        lr: float = 0.05
    ) -> Dict[str, Any]:
        """
        Doğrulama kümesi logitleri üzerinde L-BFGS ile optimal sıcaklık T* parametresini öğrenir.
        """
        cihaz = val_logitler.device
        self.to(cihaz)

        nll_criterion = nn.CrossEntropyLoss()
        optimizer = LBFGS([self.sicaklik], lr=lr, max_iter=max_iter)

        kayip_gecmisi = []

        def eval_step():
            optimizer.zero_grad()
            olcekli_logitler = self.forward(val_logitler)
            loss = nll_criterion(olcekli_logitler, val_etiketler)
            loss.backward()
            kayip_gecmisi.append(loss.item())
            return loss

        optimizer.step(eval_step)

        optimal_t = torch.clamp(self.sicaklik, min=0.05, max=50.0).item()

        return {
            "optimal_sicaklik": optimal_t,
            "nll_gecmisi": kayip_gecmisi
        }
