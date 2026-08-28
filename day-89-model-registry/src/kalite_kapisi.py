"""
Otomatik Kalite Kapısı ve Staging Doğrulama Motoru (Quality Gate)
-----------------------------------------------------------------
Modelin Staging -> Production terfisi öncesi Doğruluk, Gecikme (Latency)
ve Kalibrasyon ECE eşiklerini denetleyen otomatik güvenlik kapısı.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, Tuple
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


class ModelKaliteKapisi:
    """
    Üretime aday modelleri otomatik güvenlik ve performans testlerinden geçiren sınıf.
    """
    def __init__(
        self,
        min_dogruluk: float = 90.0,
        max_gecikme_ms: float = 30.0,
        max_ece: float = 0.15
    ):
        self.min_dogruluk = min_dogruluk
        self.max_gecikme_ms = max_gecikme_ms
        self.max_ece = max_ece

    def degerlendir(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        cihaz: str = "cpu"
    ) -> Dict[str, Any]:
        """
        Modeli doğruluk, gecikme ve kalibrasyon açısından denetler.
        """
        model = model.to(cihaz).eval()

        # 1. Doğruluk ve Olasılık Toplama
        toplam_dogru = 0
        toplam_ornek = 0
        tum_olasiliklar = []
        tum_etiketler = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(cihaz), y.to(cihaz)
                logits = model(x)
                probs = torch.softmax(logits, dim=-1)
                preds = probs.argmax(dim=-1)

                toplam_dogru += (preds == y).sum().item()
                toplam_ornek += x.size(0)

                tum_olasiliklar.append(probs.cpu())
                tum_etiketler.append(y.cpu())

        val_acc = (toplam_dogru / max(1, toplam_ornek)) * 100.0

        # 2. Çıkarım Gecikmesi (Latency Benchmark)
        test_girdi = torch.randn(1, 3, 32, 32, device=cihaz)
        for _ in range(10):  # Isınma
            _ = model(test_girdi)

        tekrar = 50
        t0 = time.time()
        with torch.no_grad():
            for _ in range(tekrar):
                _ = model(test_girdi)
        if cihaz == "cuda":
            torch.cuda.synchronize()
        ortalama_gecikme_ms = ((time.time() - t0) / tekrar) * 1000.0

        # 3. Kalibrasyon Hatası (ECE - 10 Bin)
        probs_tensor = torch.cat(tum_olasiliklar, dim=0)
        labels_tensor = torch.cat(tum_etiketler, dim=0)
        confidences, preds = torch.max(probs_tensor, dim=1)
        accuracies = preds.eq(labels_tensor)

        ece = 0.0
        bin_sayisi = 10
        for i in range(bin_sayisi):
            bin_alt = i / bin_sayisi
            bin_ust = (i + 1) / bin_sayisi
            in_bin = (confidences > bin_alt) & (confidences <= bin_ust)
            prop_in_bin = in_bin.float().mean().item()

            if prop_in_bin > 0:
                acc_in_bin = accuracies[in_bin].float().mean().item()
                avg_conf_in_bin = confidences[in_bin].mean().item()
                ece += abs(avg_conf_in_bin - acc_in_bin) * prop_in_bin

        # 4. Kalite Kapısı Onay Kriterleri
        dogruluk_onay = val_acc >= self.min_dogruluk
        gecikme_onay = ortalama_gecikme_ms <= self.max_gecikme_ms
        kalibrasyon_onay = ece <= self.max_ece

        gecti_mi = dogruluk_onay and gecikme_onay and kalibrasyon_onay

        return {
            "gecti_mi": gecti_mi,
            "metrikler": {
                "val_acc": val_acc,
                "latency_ms": ortalama_gecikme_ms,
                "ece": ece
            },
            "kriterler": {
                "dogruluk_onay": dogruluk_onay,
                "gecikme_onay": gecikme_onay,
                "kalibrasyon_onay": kalibrasyon_onay
            },
            "esikler": {
                "min_dogruluk": self.min_dogruluk,
                "max_gecikme_ms": self.max_gecikme_ms,
                "max_ece": self.max_ece
            }
        }
