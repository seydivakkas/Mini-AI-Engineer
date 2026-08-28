"""
Eğitim Süreci ve Yakınsama İzleyicisi (Training History & Convergence Tracker).
"""

from typing import List, Dict, Any, Optional
import numpy as np


class EgitimGecmisi:
    """Model eğitim sürecindeki Loss, Accuracy ve Learning Rate seyrini kaydeder ve analiz eder."""

    def __init__(self, model_adi: str = "ResNet-Classifier-v2", sabir_patience: int = 5):
        self.model_adi = model_adi
        self.sabir_patience = sabir_patience
        self.epochlar: List[int] = []
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.train_acc: List[float] = []
        self.val_acc: List[float] = []
        self.ogrenme_orani: List[float] = []

    def epoch_ekle(
        self,
        epoch: int,
        t_loss: float,
        v_loss: float,
        t_acc: float,
        v_acc: float,
        lr: float = 1e-3
    ):
        self.epochlar.append(epoch)
        self.train_loss.append(float(t_loss))
        self.val_loss.append(float(v_loss))
        self.train_acc.append(float(t_acc))
        self.val_acc.append(float(v_acc))
        self.ogrenme_orani.append(float(lr))

    def analiz_et(self) -> Dict[str, Any]:
        """Eğitim yakınsamasını, en iyi epoch'u ve overfitting farkını hesaplar."""
        if not self.epochlar:
            return {"durum": "VERI_YOK"}

        en_iyi_epoch_idx = int(np.argmin(self.val_loss))
        en_iyi_epoch = self.epochlar[en_iyi_epoch_idx]
        en_iyi_val_loss = self.val_loss[en_iyi_epoch_idx]
        en_iyi_val_acc = self.val_acc[en_iyi_epoch_idx]

        son_val_loss = self.val_loss[-1]
        overfitting_gap = float(max(0.0, son_val_loss - en_iyi_val_loss))

        # Erken durdurma kontrolü
        gecen_epoch_sayisi = len(self.epochlar) - 1 - en_iyi_epoch_idx
        erken_durdurma_tetiklendi = (gecen_epoch_sayisi >= self.sabir_patience)

        return {
            "model_adi": self.model_adi,
            "toplam_epoch": len(self.epochlar),
            "en_iyi_epoch": en_iyi_epoch,
            "en_iyi_val_loss": float(round(en_iyi_val_loss, 4)),
            "en_iyi_val_acc": float(round(en_iyi_val_acc * 100.0, 2)),
            "son_train_loss": float(round(self.train_loss[-1], 4)),
            "son_val_loss": float(round(son_val_loss, 4)),
            "overfitting_gap": float(round(overfitting_gap, 4)),
            "erken_durdurma_tetiklendi": erken_durdurma_tetiklendi,
            "overfitting_riski": "YUKSEK" if overfitting_gap > 0.15 else "NORMAL"
        }
