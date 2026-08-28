"""
Parametrik Deney Görme Modeli ve Takip Entegrasyonlu Eğitim Motoru
-----------------------------------------------------------------
Farklı hiperparametre kombinasyonlarını (Öğrenme Oranı, Optimizer, Kanal Boyutu, Dropout)
otomatik olarak loglayan ve model kontrol noktalarını kaydeden yapı.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, List, Tuple, Optional
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .takip_motoru import DeneyKosusu


class DeneyVisionModeli(nn.Module):
    """
    Parametrik Derin Görme Modeli
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

    def parametre_sayisi(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def egit_ve_kaydet(
        cls,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        kosu: DeneyKosusu,
        epok_sayisi: int = 10,
        lr: float = 1e-3,
        optimizator_tipi: str = "adamw",
        weight_decay: float = 1e-4,
        cihaz: str = "cpu"
    ) -> Dict[str, Any]:
        """
        Modeli eğitir ve her epok metriklerini merkezi deney koşusuna kaydeder.
        """
        model = model.to(cihaz)
        
        # Optimizer seçimi
        if optimizator_tipi.lower() == "adamw":
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizator_tipi.lower() == "adam":
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizator_tipi.lower() == "sgd":
            opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Parametreleri logla
        kosu.log_params({
            "learning_rate": lr,
            "optimizer": optimizator_tipi,
            "weight_decay": weight_decay,
            "epochs": epok_sayisi,
            "taban_kanal": getattr(model, "taban_kanal", 32),
            "dropout": getattr(model, "dropout_orani", 0.1),
            "param_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "batch_size": train_loader.batch_size
        })

        en_iyi_val_acc = 0.0

        for epok in range(1, epok_sayisi + 1):
            t_baslangic = time.time()
            model.train()
            tr_loss_toplam = 0.0
            tr_ornek = 0

            for x, y in train_loader:
                x, y = x.to(cihaz), y.to(cihaz)
                opt.zero_grad()
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                opt.step()

                tr_loss_toplam += loss.item() * x.size(0)
                tr_ornek += x.size(0)

            tr_loss = tr_loss_toplam / max(1, tr_ornek)

            # Doğrulama (Validation)
            model.eval()
            val_loss_toplam = 0.0
            val_dogru = 0
            val_ornek = 0

            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(cihaz), y.to(cihaz)
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
                    val_loss_toplam += loss.item() * x.size(0)
                    val_dogru += (logits.argmax(dim=-1) == y).sum().item()
                    val_ornek += x.size(0)

            val_loss = val_loss_toplam / max(1, val_ornek)
            val_acc = (val_dogru / max(1, val_ornek)) * 100.0
            epok_suresi = time.time() - t_baslangic

            # Metrikleri logla
            kosu.log_metrics({
                "train_loss": tr_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch_duration_sec": epok_suresi
            }, step=epok)

            if val_acc > en_iyi_val_acc:
                en_iyi_val_acc = val_acc
                kosu.log_model(model, "best_model.pt")

        kosu.set_tag("best_val_acc", f"{en_iyi_val_acc:.2f}")
        return {"best_val_acc": en_iyi_val_acc}
