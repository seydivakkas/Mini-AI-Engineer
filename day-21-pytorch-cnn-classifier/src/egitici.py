"""PyTorch Eğitim ve Değerlendirme Motoru (Trainer & Evaluator).

Bu modül; PyTorch modelleri için tam donanımlı eğitim döngüsü (Training Loop),
erken durdurma (Early Stopping), en iyi ağırlıkları geri yükleme (Model Checkpointing),
öğrenme oranı zamanlayıcısı (LR Scheduler) ve test kümesinde kapsamlı metrik değerlendirmesini yönetir.
"""

import copy
from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class EgitimSonucu:
    """Eğitim ve değerlendirme çıktılarının saklandığı veri yapısı."""

    train_kayiplari: List[float]
    val_kayiplari: List[float]
    train_dogruluklari: List[float]
    val_dogruluklari: List[float]
    lr_tarihcesi: List[float]
    test_kayip: float
    test_dogruluk: float
    f1_macro: float
    precision_macro: float
    recall_macro: float
    karisiklik_matrisi: np.ndarray
    y_test_gercek: np.ndarray
    y_test_tahmin: np.ndarray
    y_test_olasiliklar: np.ndarray
    egitim_suresi_sn: float
    ornek_basina_gecikme_ms: float

    def ozet(self) -> str:
        """Kısa metrik özet metnini döndürür."""
        return (
            f"Test Doğruluğu: %{self.test_dogruluk * 100:.2f} | "
            f"F1-Macro: {self.f1_macro:.4f} | "
            f"Precision: {self.precision_macro:.4f} | "
            f"Recall: {self.recall_macro:.4f} | "
            f"Eğitim Süresi: {self.egitim_suresi_sn:.2f}s | "
            f"Gecikme: {self.ornek_basina_gecikme_ms:.2f}ms/örnek"
        )


class PyTorchEgitici:
    """PyTorch eğitim ve değerlendirme sınıfı."""

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
    ) -> None:
        """Eğiticiyi ilklendirir."""
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def egit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 30,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        patience: int = 8,
        grad_clip: float = 1.0,
    ) -> Tuple[Dict[str, List[float]], nn.Module]:
        """Modeli belirtilen epoch sayısı ve erken durdurma ile eğitir.

        Returns:
            Tuple[Dict[str, List[float]], nn.Module]:
                - Tarihçe sözlüğü (train_loss, val_loss, train_acc, val_acc, lr)
                - En iyi ağırlıklara sahip model nesnesi
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )

        train_kayiplari = []
        val_kayiplari = []
        train_dogruluklari = []
        val_dogruluklari = []
        lr_tarihcesi = []

        en_iyi_val_loss = float("inf")
        en_iyi_agirliklar = copy.deepcopy(self.model.state_dict())
        sabir_sayaci = 0

        for epoch in range(epochs):
            # --- EĞİTİM ADIMI ---
            self.model.train()
            toplam_train_loss = 0.0
            dogru_train = 0
            toplam_train = 0

            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()

                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

                optimizer.step()

                toplam_train_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                dogru_train += torch.sum(preds == targets.data).item()
                toplam_train += inputs.size(0)

            epoch_train_loss = toplam_train_loss / toplam_train
            epoch_train_acc = dogru_train / toplam_train

            # --- DOĞRULAMA (VALIDATION) ADIMI ---
            self.model.eval()
            toplam_val_loss = 0.0
            dogru_val = 0
            toplam_val = 0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                    toplam_val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    dogru_val += torch.sum(preds == targets.data).item()
                    toplam_val += inputs.size(0)

            epoch_val_loss = toplam_val_loss / toplam_val
            epoch_val_acc = dogru_val / toplam_val

            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            train_kayiplari.append(epoch_train_loss)
            val_kayiplari.append(epoch_val_loss)
            train_dogruluklari.append(epoch_train_acc)
            val_dogruluklari.append(epoch_val_acc)
            lr_tarihcesi.append(current_lr)

            # Erken Durdurma ve Checkpointing
            if epoch_val_loss < en_iyi_val_loss:
                en_iyi_val_loss = epoch_val_loss
                en_iyi_agirliklar = copy.deepcopy(self.model.state_dict())
                sabir_sayaci = 0
            else:
                sabir_sayaci += 1
                if sabir_sayaci >= patience:
                    # En iyi ağırlıkları yükle ve döngüyü sonlandır
                    self.model.load_state_dict(en_iyi_agirliklar)
                    break

        # En iyi ağırlıkları garanti yükle
        self.model.load_state_dict(en_iyi_agirliklar)

        tarihce = {
            "train_loss": train_kayiplari,
            "val_loss": val_kayiplari,
            "train_acc": train_dogruluklari,
            "val_acc": val_dogruluklari,
            "lr": lr_tarihcesi,
        }
        return tarihce, self.model

    def degerlendir(
        self,
        test_loader: DataLoader,
        tarihce: Dict[str, List[float]],
        egitim_suresi_sn: float,
    ) -> EgitimSonucu:
        """Test veri seti üzerinde modeli değerlendirir ve EgitimSonucu döner."""
        self.model.eval()
        toplam_test_loss = 0.0
        toplam_test = 0

        tum_tahminler = []
        tum_olasiliklar = []
        tum_gercekler = []

        t0 = time.perf_counter()
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                probs = torch.softmax(outputs, dim=1)

                toplam_test_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)

                tum_tahminler.extend(preds.cpu().numpy())
                tum_olasiliklar.extend(probs.cpu().numpy())
                tum_gercekler.extend(targets.cpu().numpy())
                toplam_test += inputs.size(0)

        inference_suresi = time.perf_counter() - t0
        gecikme_ms = (inference_suresi / max(1, toplam_test)) * 1000.0

        y_true = np.array(tum_gercekler)
        y_pred = np.array(tum_tahminler)
        y_prob = np.array(tum_olasiliklar)

        test_loss = toplam_test_loss / max(1, toplam_test)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        return EgitimSonucu(
            train_kayiplari=tarihce["train_loss"],
            val_kayiplari=tarihce["val_loss"],
            train_dogruluklari=tarihce["train_acc"],
            val_dogruluklari=tarihce["val_acc"],
            lr_tarihcesi=tarihce["lr"],
            test_kayip=test_loss,
            test_dogruluk=acc,
            f1_macro=f1,
            precision_macro=prec,
            recall_macro=rec,
            karisiklik_matrisi=cm,
            y_test_gercek=y_true,
            y_test_tahmin=y_pred,
            y_test_olasiliklar=y_prob,
            egitim_suresi_sn=egitim_suresi_sn,
            ornek_basina_gecikme_ms=gecikme_ms,
        )
