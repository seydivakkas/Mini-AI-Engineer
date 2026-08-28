"""
Devam Edebilir ve Çökmeye Dayanıklı Eğitim Motoru
=================================================
Simüle edilmiş donanım kesintisi / SIGKILL çöküş senaryosunda,
tam durum restorasyonu yaparak kesintisiz eğitim devamlılığını sağlayan motor.
"""

from typing import Dict, Any, Tuple, Optional, List
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.model import KompaktVisionNet
from src.checkpoint_yoneticisi import GuvenliCheckpointYoneticisi


class DevamEdebilirEgitimMotoru:
    """
    Çökmelere karşı tam durum korumalı eğitim motoru.
    """

    def __init__(
        self,
        kayit_dizini: str = "checkpoints",
        lr: float = 1e-3,
        weight_decay: float = 0.01
    ) -> None:
        self.cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_yoneticisi = GuvenliCheckpointYoneticisi(kayit_dizini=kayit_dizini, maks_saklanan=3)

        self.model = KompaktVisionNet(girdi_kanali=3, sinif_sayisi=5, taban_kanal=32).to(self.cihaz)
        self.kriter = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=1e-5)

        self.baslangic_epoch: int = 1
        self.gecmis: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "lr": []
        }

    @classmethod
    def sentetik_veri_uret(
        cls,
        ornek_sayisi: int = 1000,
        girdi_sekli: Tuple[int, int, int] = (3, 32, 32),
        sinif_sayisi: int = 5,
        batch_size: int = 32
    ) -> Tuple[DataLoader, DataLoader]:
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        C, H, W = girdi_sekli
        X = torch.randn(ornek_sayisi, C, H, W)
        y = torch.randint(0, sinif_sayisi, (ornek_sayisi,))

        n_train = int(ornek_sayisi * 0.8)
        train_ds = TensorDataset(X[:n_train], y[:n_train])
        val_ds = TensorDataset(X[n_train:], y[n_train:])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

    def egit(
        self,
        hedef_epoch: int,
        cokus_epochu: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Eğitimi baslangic_epoch'tan hedef_epoch'a kadar koşturur.
        Eğer cokus_epochu verilmişse o epoch'ta kontrollü RuntimeError (Çökme) üretir.
        """
        train_loader, val_loader = self.sentetik_veri_uret()

        for epoch in range(self.baslangic_epoch, hedef_epoch + 1):
            # 1. Eğitim Fazı
            self.model.train()
            toplam_loss = 0.0

            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(self.cihaz), y_b.to(self.cihaz)
                self.optimizer.zero_grad()

                ciktilar = self.model(X_b)
                loss = self.kriter(ciktilar, y_b)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
                toplam_loss += loss.item() * X_b.size(0)

            self.scheduler.step()

            ort_train_loss = toplam_loss / len(train_loader.dataset)
            guncel_lr = self.optimizer.param_groups[0]["lr"]

            # 2. Doğrulama Fazı
            self.model.eval()
            val_loss = 0.0
            dogru = 0
            with torch.no_grad():
                for X_v, y_v in val_loader:
                    X_v, y_v = X_v.to(self.cihaz), y_v.to(self.cihaz)
                    out_v = self.model(X_v)
                    v_l = self.kriter(out_v, y_v)
                    val_loss += v_l.item() * X_v.size(0)
                    tahmin = torch.argmax(out_v, dim=1)
                    dogru += (tahmin == y_v).sum().item()

            ort_val_loss = val_loss / len(val_loader.dataset)
            val_acc = (dogru / len(val_loader.dataset)) * 100.0

            # 3. Metrik Kaydı
            self.gecmis["epoch"].append(epoch)
            self.gecmis["train_loss"].append(round(ort_train_loss, 5))
            self.gecmis["val_loss"].append(round(ort_val_loss, 5))
            self.gecmis["val_accuracy"].append(round(val_acc, 2))
            self.gecmis["lr"].append(guncel_lr)

            # 4. Atomik Checkpoint Kaydı
            self.checkpoint_yoneticisi.kaydet_atomik(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                val_loss=ort_val_loss,
                val_acc=val_acc
            )

            # 5. Simüle Edilmiş Çökme / Donanım Kesintisi Kontrolü
            if cokus_epochu is not None and epoch == cokus_epochu:
                raise RuntimeError(
                    f"SIMULE EDILMIS DONANIM COKMESI / SIGKILL! (Epoch: {epoch} tamamlandi ve kaydedildi, süreç sonlandı.)"
                )

        return {
            "son_epoch": hedef_epoch,
            "gecmis": self.gecmis,
            "en_iyi_val_loss": self.checkpoint_yoneticisi.en_iyi_metrik
        }

    def checkpointten_devam_et(self, checkpoint_yolu: str) -> int:
        """
        Kayıtlı bir checkpoint'ten tüm sistem durumunu geri yükler ve başlangıç epoch'unu ayarlar.
        """
        durum = self.checkpoint_yoneticisi.yukle_ve_geri_yukle(
            dosya_yolu=checkpoint_yolu,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            cihaz=self.cihaz
        )
        self.baslangic_epoch = durum["epoch"] + 1
        return self.baslangic_epoch
