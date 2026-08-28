"""
SimCLR Kendi Kendine Denetimli (Self-Supervised) Eğitim Motoru
-------------------------------------------------------------
Etiketsiz görüntüler üzerinden kontrastif temsil öğrenimini yürüten ve
hizalama (alignment) / düzgünlük (uniformity) dinamiklerini izleyen motor.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, List, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .simclr_model import SimCLRModeli
from .nt_xent_loss import NTXentLoss


class SimCLREgitimMotoru:
    """
    SimCLR modelini etiketsiz veri üzerinde eğiten ve temsil kalitesini izleyen motor.
    """
    def __init__(
        self,
        model: SimCLRModeli,
        sicaklik: float = 0.5,
        ogrenme_orani: float = 1e-3,
        agirlik_cezasi: float = 1e-4,
        toplam_epoch: int = 10,
        cihaz: str = "cpu"
    ):
        self.model = model.to(cihaz)
        self.cihaz = cihaz
        self.loss_fn = NTXentLoss(sicaklik=sicaklik)
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=ogrenme_orani,
            weight_decay=agirlik_cezasi
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=toplam_epoch, eta_min=1e-6)
        
        # Metrik Geçmişi
        self.gecmis: Dict[str, List[float]] = {
            "epoch": [],
            "loss": [],
            "lr": [],
            "alignment_loss": [],
            "uniformity_loss": [],
            "pozitif_kosinus": [],
            "negatif_kosinus": [],
            "kosinus_marjini": []
        }

    def bir_epoch_egit(self, veri_yukleyici: DataLoader) -> Dict[str, float]:
        """Tek bir epoch boyunca NT-Xent kaybı ile eğitim yapar."""
        self.model.train()
        toplam_kayip = 0.0
        toplam_hizalama = 0.0
        toplam_duzenlilik = 0.0
        toplam_poz_cos = 0.0
        toplam_neg_cos = 0.0
        adim_sayisi = 0

        for batch in veri_yukleyici:
            # Batch çifti: (v1, v2) veya (v1, v2, label)
            if isinstance(batch, (list, tuple)):
                v1, v2 = batch[0], batch[1]
            else:
                raise ValueError("Veri yükleyici (v1, v2) çifti dönmelidir.")

            v1, v2 = v1.to(self.cihaz), v2.to(self.cihaz)

            self.optimizer.zero_grad()
            _, z1 = self.model(v1)
            _, z2 = self.model(v2)

            kayip = self.loss_fn(z1, z2)
            kayip.backward()
            self.optimizer.step()

            # Metrikleri hesapla
            metrikler = self.loss_fn.hesapla_hizalama_ve_duzenlilik(z1, z2)
            
            toplam_kayip += kayip.item()
            toplam_hizalama += metrikler["alignment_loss"]
            toplam_duzenlilik += metrikler["uniformity_loss"]
            toplam_poz_cos += metrikler["pozitif_kosinus_ort"]
            toplam_neg_cos += metrikler["negatif_kosinus_ort"]
            adim_sayisi += 1

        self.scheduler.step()
        
        sonuclar = {
            "loss": toplam_kayip / max(1, adim_sayisi),
            "alignment_loss": toplam_hizalama / max(1, adim_sayisi),
            "uniformity_loss": toplam_duzenlilik / max(1, adim_sayisi),
            "pozitif_kosinus": toplam_poz_cos / max(1, adim_sayisi),
            "negatif_kosinus": toplam_neg_cos / max(1, adim_sayisi),
            "kosinus_marjini": (toplam_poz_cos - toplam_neg_cos) / max(1, adim_sayisi),
            "lr": float(self.optimizer.param_groups[0]["lr"])
        }
        return sonuclar

    def egit(self, veri_yukleyici: DataLoader, epoch_sayisi: int) -> Dict[str, List[float]]:
        """Tüm epoch'ları çalıştırır ve geçmişi kaydeder."""
        for ep in range(1, epoch_sayisi + 1):
            ep_sonuclar = self.bir_epoch_egit(veri_yukleyici)
            
            self.gecmis["epoch"].append(ep)
            self.gecmis["loss"].append(ep_sonuclar["loss"])
            self.gecmis["lr"].append(ep_sonuclar["lr"])
            self.gecmis["alignment_loss"].append(ep_sonuclar["alignment_loss"])
            self.gecmis["uniformity_loss"].append(ep_sonuclar["uniformity_loss"])
            self.gecmis["pozitif_kosinus"].append(ep_sonuclar["pozitif_kosinus"])
            self.gecmis["negatif_kosinus"].append(ep_sonuclar["negatif_kosinus"])
            self.gecmis["kosinus_marjini"].append(ep_sonuclar["kosinus_marjini"])
            
        return self.gecmis

    def temsilleri_cikar(self, veri_yukleyici: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tüm veri kümesi için temel temsil vektörlerini (h) ve etiketleri toplar."""
        self.model.eval()
        temsiller = []
        etiketler = []
        
        with torch.no_grad():
            for batch in veri_yukleyici:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                    y = batch[2] if len(batch) > 2 else torch.zeros(x.size(0))
                else:
                    x = batch
                    y = torch.zeros(x.size(0))
                    
                x = x.to(self.cihaz)
                h = self.model.temsil_cikar(x)
                temsiller.append(h.cpu())
                etiketler.append(y)
                
        return torch.cat(temsiller, dim=0), torch.cat(etiketler, dim=0)
