"""
Triplet Metric Learning Eğitim Motoru
-------------------------------------
Online madencilik, mesafe takibi ve adaptif marjin ayrışmasını yöneten motor.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, List, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .triplet_ag import MetrikOznitelikAgi
from .triplet_loss import ModulerTripletMarginLoss


class TripletEgitimMotoru:
    """
    Triplet Metrik Öğrenim Döngüsü ve İstatistik Motoru.
    """
    def __init__(
        self,
        model: MetrikOznitelikAgi,
        marjin: float = 0.3,
        strateji: str = "batch_semi_hard",
        ogrenme_orani: float = 1e-3,
        agirlik_cezasi: float = 1e-4,
        cihaz: str = "cpu"
    ):
        self.model = model.to(cihaz)
        self.cihaz = cihaz
        self.loss_fn = ModulerTripletMarginLoss(marjin=marjin, strateji=strateji)
        
        self.optimizer = AdamW(self.model.parameters(), lr=ogrenme_orani, weight_decay=agirlik_cezasi)
        
        self.gecmis: Dict[str, List[float]] = {
            "epoch": [],
            "loss": [],
            "d_ap": [],
            "d_an": [],
            "marjin": [],
            "aktif_oran": [],
            "zor_oran": [],
            "yari_zor_oran": [],
            "kolay_oran": []
        }

    def egit(
        self,
        veri_yukleyici: DataLoader,
        toplam_epoch: int = 8
    ) -> Dict[str, List[float]]:
        """
        Modeli toplam_epoch boyunca eğitir ve zengin metrik geçmişini kaydeder.
        """
        scheduler = CosineAnnealingLR(self.optimizer, T_max=toplam_epoch, eta_min=1e-6)

        for ep in range(1, toplam_epoch + 1):
            self.model.train()
            ep_loss = 0.0
            ep_d_ap = 0.0
            ep_d_an = 0.0
            ep_aktif = 0.0
            ep_zor = 0.0
            ep_yari_zor = 0.0
            ep_kolay = 0.0
            adim = 0

            for batch_x, batch_y in veri_yukleyici:
                batch_x = batch_x.to(self.cihaz)
                batch_y = batch_y.to(self.cihaz)

                self.optimizer.zero_grad()
                gomulmeler = self.model(batch_x)
                
                kayip, istatistik = self.loss_fn(gomulmeler, batch_y)
                kayip.backward()
                self.optimizer.step()

                ep_loss += kayip.item()
                ep_d_ap += istatistik["d_ap_ort"]
                ep_d_an += istatistik["d_an_ort"]
                ep_aktif += istatistik["aktif_triplet_orani"]
                ep_zor += istatistik["zor_orani"]
                ep_yari_zor += istatistik["yari_zor_orani"]
                ep_kolay += istatistik["kolay_orani"]
                adim += 1

            scheduler.step()

            n = max(1, adim)
            loss_ort = ep_loss / n
            d_ap_ort = ep_d_ap / n
            d_an_ort = ep_d_an / n
            marjin_ort = d_an_ort - d_ap_ort

            self.gecmis["epoch"].append(ep)
            self.gecmis["loss"].append(loss_ort)
            self.gecmis["d_ap"].append(d_ap_ort)
            self.gecmis["d_an"].append(d_an_ort)
            self.gecmis["marjin"].append(marjin_ort)
            self.gecmis["aktif_oran"].append(ep_aktif / n)
            self.gecmis["zor_oran"].append(ep_zor / n)
            self.gecmis["yari_zor_oran"].append(ep_yari_zor / n)
            self.gecmis["kolay_oran"].append(ep_kolay / n)

        return self.gecmis

    def gomulmeleri_cikar(self, veri_yukleyici: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tüm veri seti için normalize edilmiş embedding vektörlerini ve etiketleri çıkarır."""
        self.model.eval()
        vektorler = []
        etiketler = []
        with torch.no_grad():
            for batch_x, batch_y in veri_yukleyici:
                batch_x = batch_x.to(self.cihaz)
                e = self.model(batch_x)
                vektorler.append(e.cpu())
                etiketler.append(batch_y)
        return torch.cat(vektorler, dim=0), torch.cat(etiketler, dim=0)
