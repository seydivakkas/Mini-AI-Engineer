"""
Knowledge Distillation Eğitim ve Değerlendirme Motoru
-----------------------------------------------------
Öğretmen modelden öğrenciye bilgi aktaran, saf CE vs KD başarımını kıyaslayan ve
metrikleri anlık kaydeden yönetici motor.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .kayip_damitma import BilgiDamitmaKaybi


class BilgiDamiticiMotor:
    """
    Öğretmen-Öğrenci Bilgi Damıtma Eğitim ve Değerlendirme Yöneticisi
    """
    def __init__(
        self,
        ogrenci_modeli: nn.Module,
        ogretmen_modeli: Optional[nn.Module] = None,
        cihaz: str = "cpu",
        ogrenme_orani: float = 1e-3,
        sicaklik: float = 4.0,
        alfa: float = 0.7
    ):
        self.ogrenci = ogrenci_modeli.to(cihaz)
        self.ogretmen = ogretmen_modeli.to(cihaz) if ogretmen_modeli is not None else None
        self.cihaz = cihaz
        self.damitma_aktif = self.ogretmen is not None

        # Eğer öğretmen varsa dondur
        if self.ogretmen is not None:
            self.ogretmen.eval()
            for p in self.ogretmen.parameters():
                p.requires_grad = False

        self.optimizer = torch.optim.AdamW(self.ogrenci.parameters(), lr=ogrenme_orani, weight_decay=1e-4)
        self.damitma_kayip_fn = BilgiDamitmaKaybi(sicaklik=sicaklik, alfa=alfa)

        self.gecmis: Dict[str, List[float]] = {
            "toplam_kayip": [],
            "ce_kaybi": [],
            "kl_kaybi": [],
            "egitim_dogruluk": [],
            "dogrulama_dogruluk": []
        }

    def egitim_adimi(self, egitim_loader: DataLoader) -> Tuple[float, float, float, float]:
        """Tek bir epokluk eğitim adımı"""
        self.ogrenci.train()
        toplam_loss = 0.0
        toplam_ce = 0.0
        toplam_kl = 0.0
        toplam_dogru = 0
        toplam_ornek = 0

        for gorseller, etiketler in egitim_loader:
            gorseller = gorseller.to(self.cihaz)
            etiketler = etiketler.to(self.cihaz)

            self.optimizer.zero_grad()
            ogrenci_logitler = self.ogrenci(gorseller)

            if self.damitma_aktif:
                with torch.no_grad():
                    ogretmen_logitler = self.ogretmen(gorseller)
                loss, metrikler = self.damitma_kayip_fn(ogrenci_logitler, ogretmen_logitler, etiketler)
                ce_loss = metrikler["ce_kaybi"]
                kl_loss = metrikler["kl_kaybi"]
            else:
                loss = nn.functional.cross_entropy(ogrenci_logitler, etiketler)
                ce_loss = loss.item()
                kl_loss = 0.0

            loss.backward()
            self.optimizer.step()

            batch_size = gorseller.size(0)
            toplam_loss += loss.item() * batch_size
            toplam_ce += ce_loss * batch_size
            toplam_kl += kl_loss * batch_size
            toplam_dogru += (ogrenci_logitler.argmax(dim=-1) == etiketler).sum().item()
            toplam_ornek += batch_size

        return (
            toplam_loss / max(1, toplam_ornek),
            toplam_ce / max(1, toplam_ornek),
            toplam_kl / max(1, toplam_ornek),
            (toplam_dogru / max(1, toplam_ornek)) * 100.0
        )

    def dogrulama_adimi(self, val_loader: DataLoader) -> float:
        """Doğrulama adımı"""
        self.ogrenci.eval()
        toplam_dogru = 0
        toplam_ornek = 0

        with torch.no_grad():
            for gorseller, etiketler in val_loader:
                gorseller = gorseller.to(self.cihaz)
                etiketler = etiketler.to(self.cihaz)

                logitler = self.ogrenci(gorseller)
                toplam_dogru += (logitler.argmax(dim=-1) == etiketler).sum().item()
                toplam_ornek += gorseller.size(0)

        return (toplam_dogru / max(1, toplam_ornek)) * 100.0

    def egit(self, egitim_loader: DataLoader, val_loader: DataLoader, toplam_epok: int = 10) -> Dict[str, List[float]]:
        """Tüm epokları koşturan ana döngü"""
        for epok in range(toplam_epok):
            loss, ce_loss, kl_loss, tr_acc = self.egitim_adimi(egitim_loader)
            val_acc = self.dogrulama_adimi(val_loader)

            self.gecmis["toplam_kayip"].append(loss)
            self.gecmis["ce_kaybi"].append(ce_loss)
            self.gecmis["kl_kaybi"].append(kl_loss)
            self.gecmis["egitim_dogruluk"].append(tr_acc)
            self.gecmis["dogrulama_dogruluk"].append(val_acc)

        return self.gecmis
