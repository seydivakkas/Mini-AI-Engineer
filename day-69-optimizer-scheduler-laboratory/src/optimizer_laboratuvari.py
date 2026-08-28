"""
Optimizer ve Scheduler Karsilastirma Laboratuvari
=================================================
AdamW + StepLR vs AdamW + WarmupCosine vs Lion + WarmupCosine kombinasyonlarini
ayni tohum ve mimaride eszamanli egiterek kayip, dogruluk, gradyan normu ve
bellek kullanimini olcen deneysel test platformu.
"""

from typing import Dict, Any, List, Tuple
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.lion_optimizer import Lion
from src.zamanlayicilar import LinearWarmupCosineScheduler
from src.laboratuvar_modeli import DeneySinirAgi, parametre_gruplari_ayristir


class OptimizerLaboratuvari:
    """
    Farklı optimizasyon ve zamanlama stratejilerini kıyaslayan laboratuvar motoru.
    """

    @classmethod
    def tohum_sabitle(cls, tohum: int = 42) -> None:
        random.seed(tohum)
        np.random.seed(tohum)
        torch.manual_seed(tohum)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(tohum)

    @classmethod
    def veri_olustur(
        cls,
        ornek_sayisi: int = 1000,
        girdi_sekli: Tuple[int, int, int] = (3, 32, 32),
        sinif_sayisi: int = 5,
        batch_size: int = 32
    ) -> Tuple[DataLoader, DataLoader]:
        cls.tohum_sabitle(42)
        C, H, W = girdi_sekli
        X = torch.randn(ornek_sayisi, C, H, W)
        y = torch.randint(0, sinif_sayisi, (ornek_sayisi,))

        n_train = int(ornek_sayisi * 0.8)
        train_ds = TensorDataset(X[:n_train], y[:n_train])
        val_ds = TensorDataset(X[n_train:], y[n_train:])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

    @classmethod
    def tek_deney_kos(
        cls,
        deney_adi: str,
        optimizer_turu: str,
        scheduler_turu: str,
        lr: float,
        weight_decay: float,
        toplam_epoch: int = 10,
        warmup_epoch: int = 2
    ) -> Dict[str, Any]:
        """Tekil bir optimizer + scheduler konfigürasyonunu eğitir ve metrikleri toplar."""
        cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.tohum_sabitle(42)

        train_loader, val_loader = cls.veri_olustur()
        model = DeneySinirAgi(girdi_kanali=3, sinif_sayisi=5, taban_kanal=32).to(cihaz)
        kriter = nn.CrossEntropyLoss()

        param_gruplari = parametre_gruplari_ayristir(model, weight_decay=weight_decay)

        # 1. Optimizer Seçimi
        if optimizer_turu.lower() == "adamw":
            optimizer = torch.optim.AdamW(param_gruplari, lr=lr, betas=(0.9, 0.999))
            # AdamW parametre başına 2 tensör tutar (exp_avg, exp_avg_sq)
            durum_tensors_per_param = 2
        elif optimizer_turu.lower() == "lion":
            optimizer = Lion(param_gruplari, lr=lr, betas=(0.9, 0.99), weight_decay=weight_decay)
            # Lion parametre başına sadece 1 tensör tutar (exp_avg)
            durum_tensors_per_param = 1
        else:
            raise ValueError(f"Bilinmeyen optimizer: {optimizer_turu}")

        # 2. Scheduler Seçimi
        if scheduler_turu.lower() == "warmup_cosine":
            scheduler = LinearWarmupCosineScheduler(
                optimizer,
                warmup_epochs=warmup_epoch,
                max_epochs=toplam_epoch,
                eta_min=lr * 0.01
            )
        elif scheduler_turu.lower() == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=max(1, toplam_epoch // 3),
                gamma=0.5
            )
        else:
            raise ValueError(f"Bilinmeyen scheduler: {scheduler_turu}")

        toplam_param_sayisi = sum(p.numel() for p in model.parameters())
        tahmini_opt_bellek_kb = (toplam_param_sayisi * 4 * durum_tensors_per_param) / 1024.0

        gecmis = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "lr": [],
            "grad_norm": []
        }

        for epoch in range(1, toplam_epoch + 1):
            # Eğitim
            model.train()
            toplam_loss = 0.0
            epoch_grad_normlar = []

            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(cihaz), y_b.to(cihaz)
                optimizer.zero_grad()

                ciktilar = model(X_b)
                loss = kriter(ciktilar, y_b)
                loss.backward()

                # Gradyan Normu Ölçümü ve Kırpma
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                epoch_grad_normlar.append(float(total_norm.item()))

                optimizer.step()
                toplam_loss += loss.item() * X_b.size(0)

            scheduler.step()

            ort_train_loss = toplam_loss / len(train_loader.dataset)
            ort_grad_norm = float(np.mean(epoch_grad_normlar))
            guncel_lr = optimizer.param_groups[0]["lr"]

            # Doğrulama
            model.eval()
            val_loss = 0.0
            dogru = 0
            with torch.no_grad():
                for X_v, y_v in val_loader:
                    X_v, y_v = X_v.to(cihaz), y_v.to(cihaz)
                    val_out = model(X_v)
                    v_l = kriter(val_out, y_v)
                    val_loss += v_l.item() * X_v.size(0)
                    tahmin = torch.argmax(val_out, dim=1)
                    dogru += (tahmin == y_v).sum().item()

            ort_val_loss = val_loss / len(val_loader.dataset)
            val_acc = (dogru / len(val_loader.dataset)) * 100.0

            gecmis["train_loss"].append(round(ort_train_loss, 5))
            gecmis["val_loss"].append(round(ort_val_loss, 5))
            gecmis["val_accuracy"].append(round(val_acc, 2))
            gecmis["lr"].append(guncel_lr)
            gecmis["grad_norm"].append(round(ort_grad_norm, 4))

        return {
            "deney_adi": deney_adi,
            "optimizer": optimizer_turu.upper(),
            "scheduler": scheduler_turu.upper(),
            "lr": lr,
            "weight_decay": weight_decay,
            "tahmini_opt_bellek_kb": round(tahmini_opt_bellek_kb, 1),
            "son_train_loss": gecmis["train_loss"][-1],
            "son_val_loss": gecmis["val_loss"][-1],
            "son_val_accuracy": gecmis["val_accuracy"][-1],
            "gecmis": gecmis
        }

    @classmethod
    def tum_laboratuvari_kos(cls, toplam_epoch: int = 10) -> Dict[str, Any]:
        """Tüm deney kombinasyonlarını koşturur."""
        # 1. AdamW + StepLR (Klasik Baseline)
        deney_1 = cls.tek_deney_kos(
            deney_adi="AdamW + StepLR",
            optimizer_turu="adamw",
            scheduler_turu="step",
            lr=1e-3,
            weight_decay=0.01,
            toplam_epoch=toplam_epoch
        )

        # 2. AdamW + Warmup Cosine (Modern Standart)
        deney_2 = cls.tek_deney_kos(
            deney_adi="AdamW + WarmupCosine",
            optimizer_turu="adamw",
            scheduler_turu="warmup_cosine",
            lr=1e-3,
            weight_decay=0.01,
            toplam_epoch=toplam_epoch,
            warmup_epoch=2
        )

        # 3. Lion + Warmup Cosine (Yeni Nesil Sign Momentum)
        deney_3 = cls.tek_deney_kos(
            deney_adi="Lion + WarmupCosine",
            optimizer_turu="lion",
            scheduler_turu="warmup_cosine",
            lr=1e-4,
            weight_decay=0.05,
            toplam_epoch=toplam_epoch,
            warmup_epoch=2
        )

        return {
            "deney_1": deney_1,
            "deney_2": deney_2,
            "deney_3": deney_3,
            "toplam_epoch": toplam_epoch
        }
