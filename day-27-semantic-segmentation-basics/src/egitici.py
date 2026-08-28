"""U-Net Modeli Eğitim ve Değerlendirme Döngüsü Modülü.

Bu modül; U-Net modelini AdamW optimizasyonu, Combo Loss (CE + Dice) ve
validasyon mIoU takibi ile eğiten üretim seviyesi bir eğitim yöneticisidir.
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.kayip_ve_metrikler import BolutlemeMetrikleri, ComboLoss


class BolutlemeEgitici:
    """Anlamsal bölütleme modellerini eğiten ve değerlendiren sınıf."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        lr: float = 1e-3,
        alpha: float = 0.5,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = model.to(self.device)
        self.criterion = ComboLoss(alpha=alpha).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

    def bir_epok_egit(self, dataloader: DataLoader) -> float:
        """Modeli tek bir epok boyunca eğitir ve ortalama kaybı döndürür."""
        self.model.train()
        toplam_kayip = 0.0

        for images, masks in dataloader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, masks)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            toplam_kayip += loss.item() * images.size(0)

        return float(toplam_kayip / len(dataloader.dataset))

    def dogrula(
        self, dataloader: DataLoader, sinif_isimleri: List[str]
    ) -> Tuple[float, Dict]:
        """Doğrulama kümesinde modelin kaybını ve mIoU/Dice metriklerini hesaplar."""
        self.model.eval()
        toplam_kayip = 0.0

        tum_tahminler = []
        tum_gercekler = []

        with torch.no_grad():
            for images, masks in dataloader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, masks)

                toplam_kayip += loss.item() * images.size(0)

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                targets = masks.cpu().numpy()

                tum_tahminler.extend(list(preds))
                tum_gercekler.extend(list(targets))

        ort_kayip = float(toplam_kayip / len(dataloader.dataset))
        metrik_raporu = BolutlemeMetrikleri.kapsamli_rapor(
            np.array(tum_tahminler), np.array(tum_gercekler), sinif_isimleri
        )

        return ort_kayip, metrik_raporu

    def tam_egitim(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        sinif_isimleri: List[str],
        epochs: int = 5,
    ) -> Dict:
        """Belirtilen epok sayısı boyunca eğitimi ve validasyonu yürütür."""
        tarihce = {
            "train_loss": [],
            "val_loss": [],
            "val_miou": [],
            "val_pixel_acc": [],
            "val_mean_dice": [],
        }

        for epoch in range(1, epochs + 1):
            train_loss = self.bir_epok_egit(train_loader)
            val_loss, metrikler = self.dogrula(val_loader, sinif_isimleri)

            tarihce["train_loss"].append(train_loss)
            tarihce["val_loss"].append(val_loss)
            tarihce["val_miou"].append(metrikler["miou"])
            tarihce["val_pixel_acc"].append(metrikler["pixel_accuracy"])
            tarihce["val_mean_dice"].append(metrikler["mean_dice"])

            print(f"[*] Epok [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: %{metrikler['miou']*100:.2f} | Pix Acc: %{metrikler['pixel_accuracy']*100:.2f}")

        return tarihce
