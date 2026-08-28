"""
Mini Vision Transformer Eğitim ve Değerlendirme Motoru
------------------------------------------------------
AdamW Parametre Ayrımı (Decoupled Weight Decay), Linear Warmup + Cosine Annealing LR Zamanlayıcısı,
Gradyan Kırpma (Gradient Clipping), Top-1 & Top-5 Doğruluk ve Detaylı Eğitim Geçmişi Takipçisi.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, List, Tuple, Any, Optional
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .kayip_fonksiyonlari import YumusatilmisCrossEntropyKaybi
from .veri_artirma import MixupCutMixUygulayici


def ayristir_parametre_gruplari(
    model: nn.Module,
    agirlik_azaltma: float = 0.05
) -> List[Dict[str, Any]]:
    """
    Weight Decay uygulanacak (2D matrisler: Linear/Conv ağırlıkları) ve
    uygulanmayacak (1D biaslar, LayerNorm parametreleri, pos_embed) parametreleri ayırır.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or "pos_embed" in name or "cls_token" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": agirlik_azaltma},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]


def hesapla_dogruluk_top_k(
    cikti_logitleri: torch.Tensor,
    hedefler: torch.Tensor,
    top_k: Tuple[int, ...] = (1, 5)
) -> List[float]:
    """
    Top-1 ve Top-5 doğruluk oranlarını (%) hesaplar.
    """
    with torch.no_grad():
        max_k = max(top_k)
        batch_size = hedefler.size(0)

        # Eğer hedefler one-hot/soft ise en yüksek olasılıklı sınıfı al
        if hedefler.ndim == 2:
            targets = hedefler.argmax(dim=-1)
        else:
            targets = hedefler

        _, pred = cikti_logitleri.topk(max_k, dim=-1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        sonuclar = []
        for k in top_k:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            sonuclar.append(correct_k.mul_(100.0 / batch_size).item())
        return sonuclar


class MiniViTEgitici:
    """
    Mini Vision Transformer için Uçtan Uca Eğitim ve Değerlendirme Yöneticisi
    """
    def __init__(
        self,
        model: nn.Module,
        cihaz: str = "cpu",
        ogrenme_orani: float = 5e-4,
        min_ogrenme_orani: float = 1e-6,
        toplam_epok: int = 15,
        isinma_epok: int = 3,
        agirlik_azaltma: float = 0.05,
        gradyan_kirpma_normu: float = 1.0,
        etiket_yumusatma: float = 0.1,
        mixup_uygulayici: Optional[MixupCutMixUygulayici] = None
    ):
        self.model = model.to(cihaz)
        self.cihaz = cihaz
        self.ogrenme_orani = ogrenme_orani
        self.min_ogrenme_orani = min_ogrenme_orani
        self.toplam_epok = toplam_epok
        self.isinma_epok = isinma_epok
        self.gradyan_kirpma_normu = gradyan_kirpma_normu
        self.mixup_uygulayici = mixup_uygulayici

        # 1. Optimizer (Parametre Grupları ile AdamW)
        param_gruplari = ayristir_parametre_gruplari(model, agirlik_azaltma)
        self.optimizer = torch.optim.AdamW(param_gruplari, lr=ogrenme_orani, betas=(0.9, 0.999), eps=1e-8)

        # 2. Kayıp Fonksiyonu
        self.kayip_fonksiyonu = YumusatilmisCrossEntropyKaybi(etiket_yumusatma=etiket_yumusatma)

        # 3. Metrik Geçmişi
        self.gecmis: Dict[str, List[float]] = {
            "egitim_kaybi": [],
            "dogrulama_kaybi": [],
            "egitim_top1_acc": [],
            "egitim_top5_acc": [],
            "dogrulama_top1_acc": [],
            "dogrulama_top5_acc": [],
            "ogrenme_oranlari": [],
            "gradyan_normlari": []
        }

    def _ogrenme_orani_ayarla(self, mevcut_epok: int):
        """Warmup + Cosine Annealing LR Zamanlayıcısı"""
        if mevcut_epok < self.isinma_epok:
            # Linear Warmup
            lr = self.ogrenme_orani * (mevcut_epok + 1) / max(1, self.isinma_epok)
        else:
            # Cosine Decay
            ilerleme = (mevcut_epok - self.isinma_epok) / max(1, self.toplam_epok - self.isinma_epok)
            lr = self.min_ogrenme_orani + 0.5 * (self.ogrenme_orani - self.min_ogrenme_orani) * (1.0 + math.cos(math.pi * ilerleme))

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def egitim_adimi(self, egitim_loader: DataLoader, epok: int) -> Tuple[float, float, float, float]:
        """Tek bir epokluk eğitim adımı"""
        self.model.train()
        toplam_kayip = 0.0
        toplam_top1 = 0.0
        toplam_top5 = 0.0
        toplam_grad_norm = 0.0
        adim_sayisi = len(egitim_loader)

        mevcut_lr = self._ogrenme_orani_ayarla(epok)

        for gorseller, etiketler in egitim_loader:
            gorseller = gorseller.to(self.cihaz)
            etiketler = etiketler.to(self.cihaz)

            # Mixup / CutMix Uygula
            if self.mixup_uygulayici is not None:
                gorseller, yumusak_hedefler = self.mixup_uygulayici(gorseller, etiketler)
            else:
                yumusak_hedefler = etiketler

            self.optimizer.zero_grad()
            logitler = self.model(gorseller)
            kayip = self.kayip_fonksiyonu(logitler, yumusak_hedefler)
            kayip.backward()

            # Gradyan Kırpma (Gradient Clipping)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradyan_kirpma_normu)
            self.optimizer.step()

            # Metrikler
            top1, top5 = hesapla_dogruluk_top_k(logitler, yumusak_hedefler, top_k=(1, 5))
            toplam_kayip += kayip.item()
            toplam_top1 += top1
            toplam_top5 += top5
            toplam_grad_norm += grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

        ort_kayip = toplam_kayip / max(1, adim_sayisi)
        ort_top1 = toplam_top1 / max(1, adim_sayisi)
        ort_top5 = toplam_top5 / max(1, adim_sayisi)
        ort_grad_norm = toplam_grad_norm / max(1, adim_sayisi)

        return ort_kayip, ort_top1, ort_top5, ort_grad_norm

    def dogrulama_adimi(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        """Doğrulama adımı (Mixup olmadan temiz veri ile)"""
        self.model.eval()
        toplam_kayip = 0.0
        toplam_top1 = 0.0
        toplam_top5 = 0.0
        adim_sayisi = len(val_loader)

        with torch.no_grad():
            for gorseller, etiketler in val_loader:
                gorseller = gorseller.to(self.cihaz)
                etiketler = etiketler.to(self.cihaz)

                logitler = self.model(gorseller)
                kayip = self.kayip_fonksiyonu(logitler, etiketler)

                top1, top5 = hesapla_dogruluk_top_k(logitler, etiketler, top_k=(1, 5))
                toplam_kayip += kayip.item()
                toplam_top1 += top1
                toplam_top5 += top5

        return (
            toplam_kayip / max(1, adim_sayisi),
            toplam_top1 / max(1, adim_sayisi),
            toplam_top5 / max(1, adim_sayisi)
        )

    def egit(self, egitim_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        """Uçtan uca tüm epokları koşturan ana eğitim döngüsü"""
        for epok in range(self.toplam_epok):
            tr_loss, tr_top1, tr_top5, grad_norm = self.egitim_adimi(egitim_loader, epok)
            val_loss, val_top1, val_top5 = self.dogrulama_adimi(val_loader)
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.gecmis["egitim_kaybi"].append(tr_loss)
            self.gecmis["dogrulama_kaybi"].append(val_loss)
            self.gecmis["egitim_top1_acc"].append(tr_top1)
            self.gecmis["egitim_top5_acc"].append(tr_top5)
            self.gecmis["dogrulama_top1_acc"].append(val_top1)
            self.gecmis["dogrulama_top5_acc"].append(val_top5)
            self.gecmis["ogrenme_oranlari"].append(current_lr)
            self.gecmis["gradyan_normlari"].append(grad_norm)

            if (epok + 1) % 5 == 0 or epok == self.toplam_epok - 1:
                print(
                    f"  [Epok {epok+1:02d}/{self.toplam_epok:02d}] "
                    f"LR: {current_lr:.6f} | "
                    f"Tr Kayıp: {tr_loss:.4f} (Top-1: %{tr_top1:.1f}) | "
                    f"Val Kayıp: {val_loss:.4f} (Top-1: %{val_top1:.1f}, Top-5: %{val_top5:.1f})"
                )

        return self.gecmis
