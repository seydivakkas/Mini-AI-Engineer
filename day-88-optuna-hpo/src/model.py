"""
Optuna Uyumlu Parametrik Görme Modeli ve Eğitim Döngüsü
-------------------------------------------------------
Optuna Trial raporlama ve erken budama (Early Pruning) mekanizmalarına sahip derin model.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import optuna


class ParametrikVisionModeli(nn.Module):
    """
    Hiperparametre optimizasyonu için dinamik genişlik ve düzenlileştirmeye sahip model.
    """
    def __init__(
        self,
        giris_kanali: int = 3,
        sinif_sayisi: int = 10,
        taban_kanal: int = 32,
        dropout_orani: float = 0.1
    ):
        super().__init__()
        self.giris_kanali = giris_kanali
        self.sinif_sayisi = sinif_sayisi
        self.taban_kanal = taban_kanal
        self.dropout_orani = dropout_orani

        self.omurga = nn.Sequential(
            nn.Conv2d(giris_kanali, taban_kanal, kernel_size=3, padding=1),
            nn.BatchNorm2d(taban_kanal),
            nn.ReLU(),
            nn.Dropout2d(dropout_orani),
            nn.Conv2d(taban_kanal, taban_kanal * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(taban_kanal * 2),
            nn.ReLU(),
            nn.Dropout2d(dropout_orani),
            nn.Conv2d(taban_kanal * 2, taban_kanal * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(taban_kanal * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.kafa = nn.Linear(taban_kanal * 4, sinif_sayisi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.omurga(x).flatten(1)
        return self.kafa(x)

    @classmethod
    def egit_ve_degerlendir(
        cls,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-3,
        optimizator_tipi: str = "adamw",
        weight_decay: float = 1e-4,
        epok_sayisi: int = 8,
        trial: Optional[optuna.Trial] = None,
        cihaz: str = "cpu"
    ) -> float:
        """
        Modeli eğitir, her epokta Optuna'ya ara metrik raporlar ve gerekiyorsa budar.
        Dönüş: En son veya en iyi Validation Loss (Minimizasyon hedefi).
        """
        model = model.to(cihaz)

        if optimizator_tipi.lower() == "adamw":
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizator_tipi.lower() == "adam":
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizator_tipi.lower() == "sgd":
            opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        son_val_loss = float("inf")

        for epok in range(1, epok_sayisi + 1):
            model.train()
            for x, y in train_loader:
                x, y = x.to(cihaz), y.to(cihaz)
                opt.zero_grad()
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                opt.step()

            # Doğrulama Kaybı
            model.eval()
            val_loss_toplam = 0.0
            val_ornek = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(cihaz), y.to(cihaz)
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
                    val_loss_toplam += loss.item() * x.size(0)
                    val_ornek += x.size(0)

            val_loss = val_loss_toplam / max(1, val_ornek)
            son_val_loss = val_loss

            # Optuna Entegrasyonu: Raporla ve Budanmayı Kontrol Et
            if trial is not None:
                trial.report(val_loss, step=epok)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        return son_val_loss
