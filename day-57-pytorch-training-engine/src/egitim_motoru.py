"""
Modüler PyTorch Eğitim Motoru (EgitimMotoru with Gradient Clipping, Callbacks & Resume).
"""

from typing import List, Dict, Any, Optional, Tuple
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .geri_cagirimlar import EgitimCallback, ModelCheckpointCallback, EarlyStoppingCallback, MetrikKayitCallback


class EgitimMotoru:
    """Üretime hazır, olay tabanlı geri çağırımlara ve hata toleransına sahip PyTorch eğitim motoru."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[Any] = None,
        max_grad_norm: float = 1.0,
        callbacks: Optional[List[EgitimCallback]] = None,
        device: Optional[str] = None,
        sessiz: bool = False
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.sessiz = sessiz
        self.erken_durdur = False

        # Varsayılan geri çağırımlar
        if callbacks is None:
            self.metrik_kaydedici = MetrikKayitCallback()
            self.checkpoint_yoneticisi = ModelCheckpointCallback()
            self.erken_durdurucu = EarlyStoppingCallback()
            self.callbacks = [self.metrik_kaydedici, self.checkpoint_yoneticisi, self.erken_durdurucu]
        else:
            self.callbacks = callbacks
            self.metrik_kaydedici = next((c for c in callbacks if isinstance(c, MetrikKayitCallback)), MetrikKayitCallback())
            self.checkpoint_yoneticisi = next((c for c in callbacks if isinstance(c, ModelCheckpointCallback)), None)
            self.erken_durdurucu = next((c for c in callbacks if isinstance(c, EarlyStoppingCallback)), None)

    def logger(self, mesaj: str) -> None:
        if not self.sessiz:
            print(mesaj, flush=True)

    def egitim_adimi(self, train_loader: DataLoader) -> Tuple[float, float, float]:
        """Tek bir epoch için ileri/geri geçiş, gradient clipping ve parametre güncellemesi yapar."""
        self.model.train()
        toplam_kayip = 0.0
        dogru_sayisi = 0
        toplam_ornek = 0
        grad_norm_listesi: List[float] = []

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            loss.backward()

            # Gradient Kırpma (Gradient Clipping)
            if self.max_grad_norm > 0:
                toplam_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.max_grad_norm
                )
                grad_norm_listesi.append(float(toplam_grad_norm.item() if isinstance(toplam_grad_norm, torch.Tensor) else toplam_grad_norm))
            else:
                grad_norm_listesi.append(0.0)

            self.optimizer.step()

            # Metrik biriktirme
            toplam_kayip += loss.detach().item() * len(targets)
            preds = outputs.argmax(dim=-1)
            dogru_sayisi += (preds == targets).sum().item()
            toplam_ornek += len(targets)

            # Batch seviyesi geri çağırımlar
            for cb in self.callbacks:
                cb.on_batch_end(self, batch_idx, {"batch_loss": loss.item()})

        ort_kayip = float(toplam_kayip / max(toplam_ornek, 1))
        dogruluk = float((dogru_sayisi / max(toplam_ornek, 1)) * 100.0)
        ort_grad_norm = float(sum(grad_norm_listesi) / max(len(grad_norm_listesi), 1))

        return ort_kayip, dogruluk, ort_grad_norm

    def dogrulama_adimi(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Doğrulama veri seti üzerinde model performansını değerlendirir."""
        self.model.eval()
        toplam_kayip = 0.0
        dogru_sayisi = 0
        toplam_ornek = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                toplam_kayip += loss.item() * len(targets)
                preds = outputs.argmax(dim=-1)
                dogru_sayisi += (preds == targets).sum().item()
                toplam_ornek += len(targets)

        ort_kayip = float(toplam_kayip / max(toplam_ornek, 1))
        dogruluk = float((dogru_sayisi / max(toplam_ornek, 1)) * 100.0)
        return ort_kayip, dogruluk

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 15,
        baslangic_epoch: int = 1
    ) -> Dict[str, Any]:
        """Eğitim ve doğrulama döngüsünü geri çağırımlarla birlikte uçtan uca yürütür."""
        self.logger(f"\n[+] Eğitim Başlatılıyor | Cihaz: {self.device.upper()} | Toplam Epoch: {epochs} | Max Grad Norm: {self.max_grad_norm}")
        self.logger("=" * 85)

        for cb in self.callbacks:
            cb.on_train_begin(self)

        for epoch in range(baslangic_epoch, epochs + 1):
            if self.erken_durdur:
                self.logger(f"\n[!] Erken Durdurma Sinyali Alındı. Epoch {epoch}'da döngü kırıldı.")
                break

            for cb in self.callbacks:
                cb.on_epoch_begin(self, epoch)

            train_loss, train_acc, grad_norm = self.egitim_adimi(train_loader)
            val_loss, val_acc = self.dogrulama_adimi(val_loader)

            mevcut_lr = self.optimizer.param_groups[0]["lr"]

            # Öğrenme Oranı Zamanlayıcısı (LR Scheduler) Güncellemesi
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            sabir_sayaci = self.erken_durdurucu.sabir_sayaci if self.erken_durdurucu else 0
            metrikler = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "learning_rate": mevcut_lr,
                "grad_norm": grad_norm,
                "patience_sayaci": sabir_sayaci
            }

            self.logger(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"| Train Loss: {train_loss:.4f} - Acc: %{train_acc:5.1f} "
                f"| Val Loss: {val_loss:.4f} - Acc: %{val_acc:5.1f} "
                f"| Grad Norm: {grad_norm:.2f} | LR: {mevcut_lr:.1e}"
            )

            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch, metrikler)

        for cb in self.callbacks:
            cb.on_train_end(self)

        self.logger("=" * 85)
        self.logger("[+] Egitim Dongusu Basariyla Tamamlandi!")

        return self.metrik_kaydedici.gecmis

    def resume(self, checkpoint_path: str) -> Dict[str, Any]:
        """Kaydedilmiş bir kontrol noktasından (checkpoint) modeli ve optimizasyonu geri yükler."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint dosyası bulunamadı: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "torch_rng_state" in checkpoint:
            try:
                rng_s = checkpoint["torch_rng_state"]
                if isinstance(rng_s, torch.Tensor):
                    torch.set_rng_state(rng_s.cpu().to(torch.uint8))
            except Exception:
                pass

        self.logger(f"[+] Model durumu basariyla geri yuklendi: {checkpoint_path} (Kayitli Epoch: {checkpoint['epoch']})")
        return checkpoint
