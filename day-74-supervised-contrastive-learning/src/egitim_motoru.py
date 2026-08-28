"""
SupCon İki Aşamalı Eğitim Motoru (Stage 1: Kontrastif + Stage 2: Doğrusal Sınıflandırma)
---------------------------------------------------------------------------------------
Supervised Contrastive temsil öğrenimi ve ardından dondurulmuş omurga üzerinde
doğrusal sınıflandırıcı ince ayarı (Linear Probing) protokolünü yöneten motor.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, List, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .supcon_model import SupConModeli
from .supcon_loss import SupConLoss


class SupConEgitimMotoru:
    """
    SupCon Stage 1 ve Stage 2 eğitim süreçlerini yöneten kurumsal motor.
    """
    def __init__(
        self,
        model: SupConModeli,
        sicaklik: float = 0.1,
        ogrenme_orani: float = 1e-3,
        agirlik_cezasi: float = 1e-4,
        cihaz: str = "cpu"
    ):
        self.model = model.to(cihaz)
        self.cihaz = cihaz
        self.supcon_loss_fn = SupConLoss(sicaklik=sicaklik)
        self.ce_loss_fn = nn.CrossEntropyLoss()
        
        self.ogrenme_orani = ogrenme_orani
        self.agirlik_cezasi = agirlik_cezasi
        
        # Stage 1 Geçmişi
        self.stage1_gecmis: Dict[str, List[float]] = {
            "epoch": [],
            "loss": [],
            "sinif_ici_kosinus": [],
            "siniflar_arasi_kosinus": [],
            "ayrisma_marjini": []
        }
        
        # Stage 2 Geçmişi
        self.stage2_gecmis: Dict[str, List[float]] = {
            "epoch": [],
            "loss": [],
            "dogruluk": []
        }

    def egit_stage1_kontrastif(
        self,
        veri_yukleyici: DataLoader,
        toplam_epoch: int = 8
    ) -> Dict[str, List[float]]:
        """
        Aşama 1: Kodlayıcı f(.) ve Projeksiyon Kafası g(.) SupCon kaybı ile eğitilir.
        """
        optimizer = AdamW(
            list(self.model.kodlayici.parameters()) + list(self.model.projeksiyon.parameters()),
            lr=self.ogrenme_orani,
            weight_decay=self.agirlik_cezasi
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=toplam_epoch, eta_min=1e-6)

        self.model.kodlayici.train()
        self.model.projeksiyon.train()

        for ep in range(1, toplam_epoch + 1):
            toplam_kayip = 0.0
            toplam_sinif_ici = 0.0
            toplam_siniflar_arasi = 0.0
            toplam_marjin = 0.0
            adim_sayisi = 0

            for batch in veri_yukleyici:
                v1, v2, labels = batch[0].to(self.cihaz), batch[1].to(self.cihaz), batch[2].to(self.cihaz)
                
                optimizer.zero_grad()
                _, z1 = self.model(v1)
                _, z2 = self.model(v2)
                
                # (B, 2, D) formatına getir
                z_cift = torch.stack([z1, z2], dim=1)
                
                kayip = self.supcon_loss_fn(z_cift, labels)
                kayip.backward()
                optimizer.step()

                geometri = self.supcon_loss_fn.hesapla_geometrik_ayrisma(z_cift, labels)
                
                toplam_kayip += kayip.item()
                toplam_sinif_ici += geometri["sinif_ici_kosinus"]
                toplam_siniflar_arasi += geometri["siniflar_arasi_kosinus"]
                toplam_marjin += geometri["ayrisma_marjini"]
                adim_sayisi += 1

            scheduler.step()
            
            ep_loss = toplam_kayip / max(1, adim_sayisi)
            ep_ici = toplam_sinif_ici / max(1, adim_sayisi)
            ep_arasi = toplam_siniflar_arasi / max(1, adim_sayisi)
            ep_marjin = toplam_marjin / max(1, adim_sayisi)

            self.stage1_gecmis["epoch"].append(ep)
            self.stage1_gecmis["loss"].append(ep_loss)
            self.stage1_gecmis["sinif_ici_kosinus"].append(ep_ici)
            self.stage1_gecmis["siniflar_arasi_kosinus"].append(ep_arasi)
            self.stage1_gecmis["ayrisma_marjini"].append(ep_marjin)

        return self.stage1_gecmis

    def egit_stage2_dogrusal_siniflandirici(
        self,
        egitim_yukleyici: DataLoader,
        dogrulama_yukleyici: DataLoader,
        toplam_epoch: int = 5
    ) -> Dict[str, List[float]]:
        """
        Aşama 2: Kodlayıcı f(.) DONDURULUR (Freeze). Yalnızca Doğrusal Sınıflandırıcı eğitilir (Linear Probing).
        """
        # Omurgayı dondur
        for param in self.model.kodlayici.parameters():
            param.requires_grad = False
            
        self.model.kodlayici.eval()
        self.model.siniflandirici.train()

        optimizer = AdamW(self.model.siniflandirici.parameters(), lr=1e-2, weight_decay=1e-4)

        for ep in range(1, toplam_epoch + 1):
            toplam_kayip = 0.0
            adim_sayisi = 0

            for batch in egitim_yukleyici:
                x, y = batch[0].to(self.cihaz), batch[2].to(self.cihaz)
                
                optimizer.zero_grad()
                with torch.no_grad():
                    h = self.model.temsil_cikar(x)
                logits = self.model.siniflandirici(h)
                
                kayip = self.ce_loss_fn(logits, y)
                kayip.backward()
                optimizer.step()

                toplam_kayip += kayip.item()
                adim_sayisi += 1

            # Doğrulama Doğruluğu Hesapla
            self.model.siniflandirici.eval()
            dogru = 0
            toplam = 0
            with torch.no_grad():
                for batch in dogrulama_yukleyici:
                    x, y = batch[0].to(self.cihaz), batch[2].to(self.cihaz)
                    h = self.model.temsil_cikar(x)
                    logits = self.model.siniflandirici(h)
                    tahminler = torch.argmax(logits, dim=1)
                    dogru += (tahminler == y).sum().item()
                    toplam += y.size(0)

            val_acc = (dogru / max(1, toplam)) * 100.0
            self.stage2_gecmis["epoch"].append(ep)
            self.stage2_gecmis["loss"].append(toplam_kayip / max(1, adim_sayisi))
            self.stage2_gecmis["dogruluk"].append(val_acc)

        return self.stage2_gecmis

    def temsilleri_cikar(self, veri_yukleyici: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tüm veri kümesi için temsil vektörlerini (h) ve etiketleri toplar."""
        self.model.kodlayici.eval()
        temsiller = []
        etiketler = []
        with torch.no_grad():
            for batch in veri_yukleyici:
                x = batch[0].to(self.cihaz)
                y = batch[2]
                h = self.model.temsil_cikar(x)
                temsiller.append(h.cpu())
                etiketler.append(y)
        return torch.cat(temsiller, dim=0), torch.cat(etiketler, dim=0)
