"""
Modüler Olay Tabanlı Eğitim Geri Çağırımları (Training Callbacks: Checkpoint, Early Stopping, Metrics).
"""

from typing import Dict, Any, Optional
import os
import shutil
import numpy as np
import torch
import torch.nn as nn


class EgitimCallback:
    """Tüm eğitim geri çağırımları için temel soyut sınıf (Callback Base Protocol)."""

    def on_train_begin(self, motor: Any) -> None:
        pass

    def on_epoch_begin(self, motor: Any, epoch: int) -> None:
        pass

    def on_batch_end(self, motor: Any, batch_idx: int, metrikler: Dict[str, float]) -> None:
        pass

    def on_epoch_end(self, motor: Any, epoch: int, metrikler: Dict[str, float]) -> None:
        pass

    def on_train_end(self, motor: Any) -> None:
        pass


class ModelCheckpointCallback(EgitimCallback):
    """Doğrulama metriğine göre en iyi modeli atomik olarak kaydeden kontrol noktası yöneticisi."""

    def __init__(
        self,
        kayit_dizini: str = "checkpoints",
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True
    ):
        self.kayit_dizini = kayit_dizini
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.en_iyi_skor = float("inf") if mode == "min" else float("-inf")
        self.en_iyi_epoch = -1
        self.en_iyi_dosya_yolu = ""
        os.makedirs(kayit_dizini, exist_ok=True)

    def _skor_daha_iyi_mi(self, mevcut_skor: float) -> bool:
        if self.mode == "min":
            return mevcut_skor < self.en_iyi_skor
        return mevcut_skor > self.en_iyi_skor

    def on_epoch_end(self, motor: Any, epoch: int, metrikler: Dict[str, float]) -> None:
        mevcut_skor = metrikler.get(self.monitor, None)
        if mevcut_skor is None:
            return

        checkpoint_bilgisi = {
            "epoch": epoch,
            "model_state_dict": motor.model.state_dict(),
            "optimizer_state_dict": motor.optimizer.state_dict(),
            "scheduler_state_dict": motor.scheduler.state_dict() if motor.scheduler else None,
            "metrikler": metrikler,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state()
        }

        # 1. En son durum her epoch atomik kaydedilir
        son_hedef = os.path.join(self.kayit_dizini, "son_checkpoint.pt")
        gecici_hedef = son_hedef + ".tmp"
        torch.save(checkpoint_bilgisi, gecici_hedef)
        if os.path.exists(son_hedef):
            os.remove(son_hedef)
        os.rename(gecici_hedef, son_hedef)

        # 2. En iyi model kontrolü
        if self._skor_daha_iyi_mi(mevcut_skor):
            self.en_iyi_skor = mevcut_skor
            self.en_iyi_epoch = epoch
            en_iyi_hedef = os.path.join(self.kayit_dizini, "en_iyi_model.pt")
            gecici_en_iyi = en_iyi_hedef + ".tmp"
            torch.save(checkpoint_bilgisi, gecici_en_iyi)
            if os.path.exists(en_iyi_hedef):
                os.remove(en_iyi_hedef)
            os.rename(gecici_en_iyi, en_iyi_hedef)
            self.en_iyi_dosya_yolu = en_iyi_hedef
            motor.logger(f"    [*] Checkpoint: Yeni en iyi model kaydedildi! ({self.monitor}: {mevcut_skor:.4f} @ Epoch {epoch})")


class EarlyStoppingCallback(EgitimCallback):
    """Doğrulama metriğinde gelişme durduğunda eğitimi kontrollü olarak sonlandıran erken durdurucu."""

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        patience: int = 4,
        min_delta: float = 1e-4
    ):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.sabir_sayaci = 0
        self.en_iyi_skor = float("inf") if mode == "min" else float("-inf")
        self.tetiklendi = False

    def _gelisme_var_mi(self, mevcut_skor: float) -> bool:
        if self.mode == "min":
            return (self.en_iyi_skor - mevcut_skor) > self.min_delta
        return (mevcut_skor - self.en_iyi_skor) > self.min_delta

    def on_epoch_end(self, motor: Any, epoch: int, metrikler: Dict[str, float]) -> None:
        mevcut_skor = metrikler.get(self.monitor, None)
        if mevcut_skor is None:
            return

        if self._gelisme_var_mi(mevcut_skor):
            self.en_iyi_skor = mevcut_skor
            self.sabir_sayaci = 0
        else:
            self.sabir_sayaci += 1
            motor.logger(f"    [!] EarlyStopping: Gelisme yok. Sabir Sayaci: {self.sabir_sayaci}/{self.patience}")
            if self.sabir_sayaci >= self.patience:
                self.tetiklendi = True
                motor.erken_durdur = True
                motor.logger(f"    [!] EarlyStopping: Sabir esigi ({self.patience}) asildi! Egitim guvenle sonlandiriliyor.")


class MetrikKayitCallback(EgitimCallback):
    """Eğitim boyunca tüm metriklerin geçmişini kayıt altına alan ve analiz eden geri çağırım."""

    def __init__(self):
        self.gecmis: Dict[str, list] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "learning_rate": [],
            "grad_norm": [],
            "patience_sayaci": []
        }

    def on_epoch_end(self, motor: Any, epoch: int, metrikler: Dict[str, float]) -> None:
        self.gecmis["epoch"].append(epoch)
        self.gecmis["train_loss"].append(metrikler.get("train_loss", 0.0))
        self.gecmis["val_loss"].append(metrikler.get("val_loss", 0.0))
        self.gecmis["train_acc"].append(metrikler.get("train_acc", 0.0))
        self.gecmis["val_acc"].append(metrikler.get("val_acc", 0.0))
        self.gecmis["learning_rate"].append(metrikler.get("learning_rate", 0.0))
        self.gecmis["grad_norm"].append(metrikler.get("grad_norm", 0.0))
        self.gecmis["patience_sayaci"].append(metrikler.get("patience_sayaci", 0))
