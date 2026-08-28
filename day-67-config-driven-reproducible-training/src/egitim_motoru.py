"""
Tekrarlanabilir Egitim Motoru (Reproducible Training Engine)
===========================================================
Konfigurasyon odakli egitim dongusu, deterministik DataLoader,
ogrenme orani zamanlayicisi (Scheduler) ve metrik izleme yoneticisi.
"""

from typing import Dict, List, Any, Tuple
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.konfigurasyon_semasi import KokKonfigurasyon
from src.determinizm_motoru import DeterminizmYoneticisi
from src.model_mimari import ModulerVisionNet


class TekrarlanabilirEgitici:
    """
    KokKonfigurasyon nesnesinden beslenen, %100 tekrarlanabilir egitim yurutucusu.
    """

    def __init__(self, config: KokKonfigurasyon) -> None:
        self.cfg = config
        self.cihaz = torch.device("cuda" if torch.cuda.is_available() and config.egitim.deterministik_mod else "cpu")

        # 1. Tohumlari Kilitle
        DeterminizmYoneticisi.tohum_sabitle(
            tohum=self.cfg.egitim.tohum,
            deterministik_mod=self.cfg.egitim.deterministik_mod
        )

        # 2. Veri Yukleyicileri Olustur
        self.train_loader, self.val_loader = self._veri_yukleyicileri_olustur()

        # 3. Modeli Kur
        self.model = ModulerVisionNet(self.cfg.model).to(self.cihaz)

        # 4. Kayip Fonksiyonu ve Optimizer
        self.kriter = nn.CrossEntropyLoss()
        self.optimizer = self._optimizer_olustur()
        self.scheduler = self._scheduler_olustur()

        # 5. Gecmis Metrikleri
        self.gecmis: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "lr": []
        }
        self.epoch_agirlik_hashleri: List[str] = []

    def _veri_yukleyicileri_olustur(self) -> Tuple[DataLoader, DataLoader]:
        """Konfigürasyondaki tohumla deterministik sentetik veri seti oluşturur."""
        v_cfg = self.cfg.veri
        N = v_cfg.ornek_sayisi
        C, H, W = v_cfg.girdi_boyutu
        num_classes = self.cfg.model.sinif_sayisi

        # Deterministik tensör üretimi
        X = torch.randn(N, C, H, W)
        y = torch.randint(0, num_classes, (N,))

        n_train = int(N * v_cfg.egitim_orani)
        X_train, y_train = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train:], y[n_train:]

        train_ds = TensorDataset(X_train, y_train)
        val_ds = TensorDataset(X_val, y_val)

        g = DeterminizmYoneticisi.generator_al(self.cfg.egitim.tohum)

        train_loader = DataLoader(
            train_ds,
            batch_size=v_cfg.batch_size,
            shuffle=True,
            generator=g,
            worker_init_fn=DeterminizmYoneticisi.worker_init_fn,
            num_workers=v_cfg.num_workers,
            pin_memory=v_cfg.pin_memory
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=v_cfg.batch_size,
            shuffle=False,
            num_workers=v_cfg.num_workers
        )

        return train_loader, val_loader

    def _optimizer_olustur(self) -> torch.optim.Optimizer:
        """Konfigurasyona uygun Optimizer nesnesi dondurur."""
        opt_cfg = self.cfg.optimizer
        if opt_cfg.tur == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=opt_cfg.lr,
                weight_decay=opt_cfg.weight_decay,
                betas=opt_cfg.betas
            )
        elif opt_cfg.tur == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=opt_cfg.lr,
                weight_decay=opt_cfg.weight_decay,
                betas=opt_cfg.betas
            )
        elif opt_cfg.tur == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=opt_cfg.lr,
                momentum=opt_cfg.momentum,
                weight_decay=opt_cfg.weight_decay
            )
        else:
            raise ValueError(f"Desteklenmeyen optimizer: {opt_cfg.tur}")

    def _scheduler_olustur(self) -> Any:
        """Konfigurasyona uygun Scheduler nesnesi dondurur."""
        sch_cfg = self.cfg.scheduler
        if sch_cfg.tur == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sch_cfg.t_max,
                eta_min=sch_cfg.eta_min
            )
        elif sch_cfg.tur == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=max(1, sch_cfg.t_max // 3),
                gamma=0.5
            )
        elif sch_cfg.tur == "none":
            return None
        return None

    def egit(self) -> Dict[str, Any]:
        """
        Modeli konfigürasyonda belirtilen epoch sayısı boyunca deterministik olarak eğitir.
        """
        toplam_epoch = self.cfg.egitim.epoch_sayisi

        for epoch in range(1, toplam_epoch + 1):
            # 1. Eğitim Adımı
            self.model.train()
            toplam_train_kayip = 0.0

            for X_b, y_b in self.train_loader:
                X_b, y_b = X_b.to(self.cihaz), y_b.to(self.cihaz)
                self.optimizer.zero_grad()

                ciktilar = self.model(X_b)
                loss = self.kriter(ciktilar, y_b)
                loss.backward()

                if self.cfg.egitim.grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.egitim.grad_clip_norm)

                self.optimizer.step()
                toplam_train_kayip += loss.item() * X_b.size(0)

            if self.scheduler:
                self.scheduler.step()

            ort_train_loss = toplam_train_kayip / len(self.train_loader.dataset)

            # 2. Doğrulama (Validation) Adımı
            self.model.eval()
            toplam_val_kayip = 0.0
            dogru_tahmin = 0

            with torch.no_grad():
                for X_v, y_v in self.val_loader:
                    X_v, y_v = X_v.to(self.cihaz), y_v.to(self.cihaz)
                    val_out = self.model(X_v)
                    v_loss = self.kriter(val_out, y_v)
                    toplam_val_kayip += v_loss.item() * X_v.size(0)

                    tahminler = torch.argmax(val_out, dim=1)
                    dogru_tahmin += (tahminler == y_v).sum().item()

            ort_val_loss = toplam_val_kayip / len(self.val_loader.dataset)
            val_acc = (dogru_tahmin / len(self.val_loader.dataset)) * 100.0
            mevcut_lr = self.optimizer.param_groups[0]["lr"]

            # Kayıt
            self.gecmis["train_loss"].append(round(ort_train_loss, 6))
            self.gecmis["val_loss"].append(round(ort_val_loss, 6))
            self.gecmis["val_accuracy"].append(round(val_acc, 2))
            self.gecmis["lr"].append(mevcut_lr)
            self.epoch_agirlik_hashleri.append(self.model.agirlik_hashi_al())

        return {
            "gecmis": self.gecmis,
            "son_train_loss": self.gecmis["train_loss"][-1],
            "son_val_loss": self.gecmis["val_loss"][-1],
            "son_val_accuracy": self.gecmis["val_accuracy"][-1],
            "son_agirlik_hashi": self.epoch_agirlik_hashleri[-1],
            "toplam_epoch": toplam_epoch
        }
