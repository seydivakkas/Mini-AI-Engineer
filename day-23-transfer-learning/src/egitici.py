"""Transfer Öğrenme Eğitim ve Değerlendirme Motoru.

Bu modül; dondurulmuş (Frozen) ve kısmi açılmış (Fine-Tuning) omurgalarda
ayrıştırılmış öğrenme oranlarını (Discriminative Learning Rates), erken durdurmayı
ve kapsamlı metrik değerlendirmesini yönetir.
"""

import copy
from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class TransferEgitimSonucu:
    """Transfer öğrenme eğitim ve test çıktılarının saklandığı veri yapısı."""

    model_adi: str
    strateji: str
    train_kayiplari: List[float]
    val_kayiplari: List[float]
    train_dogruluklari: List[float]
    val_dogruluklari: List[float]
    test_kayip: float
    test_dogruluk: float
    f1_macro: float
    precision_macro: float
    recall_macro: float
    karisiklik_matrisi: np.ndarray
    y_test_gercek: np.ndarray
    y_test_tahmin: np.ndarray
    egitim_suresi_sn: float
    ornek_basina_gecikme_ms: float
    egitilebilir_parametre: int
    toplam_parametre: int


class TransferEgitici:
    """Transfer öğrenme eğitim sınıfı."""

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def egit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        param_gruplari: Optional[List[Dict]] = None,
        lr_varsayilan: float = 0.001,
        weight_decay: float = 1e-4,
        epochs: int = 25,
        patience: int = 8,
    ) -> Tuple[Dict[str, List[float]], nn.Module]:
        """Modeli belirtilen parametre grupları ve ayrıştırılmış LR ile eğitir."""
        if param_gruplari is not None and len(param_gruplari) > 0:
            optimizer = torch.optim.AdamW(param_gruplari)
        else:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr=lr_varsayilan, weight_decay=weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        train_kayiplari, val_kayiplari = [], []
        train_dogruluklari, val_dogruluklari = [], []

        en_iyi_val_loss = float("inf")
        en_iyi_agirliklar = copy.deepcopy(self.model.state_dict())
        sabir = 0

        for epoch in range(epochs):
            self.model.train()
            toplam_loss, dogru, toplam = 0.0, 0, 0

            for bx, by in train_loader:
                bx = bx.to(self.device)
                by = by.to(self.device)

                optimizer.zero_grad()
                out = self.model(bx)
                loss = self.criterion(out, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                toplam_loss += loss.item() * bx.size(0)
                _, preds = torch.max(out, 1)
                dogru += torch.sum(preds == by).item()
                toplam += bx.size(0)

            ep_tr_loss = toplam_loss / max(1, toplam)
            ep_tr_acc = dogru / max(1, toplam)

            # Doğrulama (Validation)
            self.model.eval()
            val_loss, val_dogru, val_toplam = 0.0, 0, 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx = bx.to(self.device)
                    by = by.to(self.device)
                    out = self.model(bx)
                    loss = self.criterion(out, by)
                    val_loss += loss.item() * bx.size(0)
                    _, preds = torch.max(out, 1)
                    val_dogru += torch.sum(preds == by).item()
                    val_toplam += bx.size(0)

            ep_val_loss = val_loss / max(1, val_toplam)
            ep_val_acc = val_dogru / max(1, val_toplam)

            scheduler.step()

            train_kayiplari.append(ep_tr_loss)
            val_kayiplari.append(ep_val_loss)
            train_dogruluklari.append(ep_tr_acc)
            val_dogruluklari.append(ep_val_acc)

            if ep_val_loss < en_iyi_val_loss:
                en_iyi_val_loss = ep_val_loss
                en_iyi_agirliklar = copy.deepcopy(self.model.state_dict())
                sabir = 0
            else:
                sabir += 1
                if sabir >= patience:
                    self.model.load_state_dict(en_iyi_agirliklar)
                    break

        self.model.load_state_dict(en_iyi_agirliklar)

        tarihce = {
            "train_loss": train_kayiplari,
            "val_loss": val_kayiplari,
            "train_acc": train_dogruluklari,
            "val_acc": val_dogruluklari,
        }
        return tarihce, self.model

    def degerlendir(
        self,
        test_loader: DataLoader,
        tarihce: Dict[str, List[float]],
        model_adi: str,
        strateji: str,
        egitim_suresi_sn: float,
    ) -> TransferEgitimSonucu:
        """Test kümesinde değerlendirir."""
        self.model.eval()
        toplam_loss, toplam = 0.0, 0
        tum_tahminler, tum_gercekler = [], []

        t0 = time.perf_counter()
        with torch.no_grad():
            for bx, by in test_loader:
                bx = bx.to(self.device)
                by = by.to(self.device)
                out = self.model(bx)
                loss = self.criterion(out, by)
                toplam_loss += loss.item() * bx.size(0)
                _, preds = torch.max(out, 1)

                tum_tahminler.extend(preds.cpu().numpy())
                tum_gercekler.extend(by.cpu().numpy())
                toplam += bx.size(0)

        gecikme_ms = ((time.perf_counter() - t0) / max(1, toplam)) * 1000.0
        y_true = np.array(tum_gercekler)
        y_pred = np.array(tum_tahminler)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        total_p = sum(p.numel() for p in self.model.parameters())
        trainable_p = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return TransferEgitimSonucu(
            model_adi=model_adi,
            strateji=strateji,
            train_kayiplari=tarihce["train_loss"],
            val_kayiplari=tarihce["val_loss"],
            train_dogruluklari=tarihce["train_acc"],
            val_dogruluklari=tarihce["val_acc"],
            test_kayip=toplam_loss / max(1, toplam),
            test_dogruluk=acc,
            f1_macro=f1,
            precision_macro=prec,
            recall_macro=rec,
            karisiklik_matrisi=cm,
            y_test_gercek=y_true,
            y_test_tahmin=y_pred,
            egitim_suresi_sn=egitim_suresi_sn,
            ornek_basina_gecikme_ms=gecikme_ms,
            egitilebilir_parametre=trainable_p,
            toplam_parametre=total_p,
        )
