"""
Regülerizasyon Karşılaştırıcı ve Laboratuvar Motoru
===================================================
Baseline vs Mixup + Label Smoothing vs CutMix + Label Smoothing
deneylerini yürüten ve model kalibrasyonu / aşırı güven (overconfidence) metriklerini toplayan modül.
"""

from typing import Dict, Any, List, Tuple
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from src.mixup_cutmix import ModernArtirici
from src.kayip_fonksiyonlari import YumusatilmisCrossEntropyLoss
from src.deney_modeli import ModernRegulerVisionNet


class RegulerizasyonLaboratuvari:
    """
    Modern düzenlileştirme stratejilerini kıyaslayan laboratuvar sınıfı.
    """

    @classmethod
    def tohum_sabitle(cls, tohum: int = 42) -> None:
        random.seed(tohum)
        np.random.seed(tohum)
        torch.manual_seed(tohum)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(tohum)

    @classmethod
    def sentetik_veri_uret(
        cls,
        ornek_sayisi: int = 1000,
        girdi_sekli: Tuple[int, int, int] = (3, 32, 32),
        sinif_sayisi: int = 5,
        batch_size: int = 32
    ) -> Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
        cls.tohum_sabitle(42)
        C, H, W = girdi_sekli

        # Farklı sınıflara ait ayırt edilebilir desenler üret
        X_list = []
        y_list = []
        for c in range(sinif_sayisi):
            n_c = ornek_sayisi // sinif_sayisi
            x_c = torch.randn(n_c, C, H, W) * 0.5 + (c * 0.3)
            y_c = torch.full((n_c,), c, dtype=torch.long)
            X_list.append(x_c)
            y_list.append(y_c)

        X = torch.cat(X_list, dim=0)
        y = torch.cat(y_list, dim=0)

        # Karıştır
        perm = torch.randperm(ornek_sayisi)
        X = X[perm]
        y = y[perm]

        n_train = int(ornek_sayisi * 0.8)
        train_ds = TensorDataset(X[:n_train], y[:n_train])
        val_ds = TensorDataset(X[n_train:], y[n_train:])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, X[:4], y[:4]

    @classmethod
    def tek_deney_kos(
        cls,
        deney_adi: str,
        artirma_turu: str,  # 'none', 'mixup', 'cutmix'
        label_smoothing: float = 0.0,
        toplam_epoch: int = 10,
        lr: float = 1e-3
    ) -> Dict[str, Any]:
        cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.tohum_sabitle(42)

        train_loader, val_loader, ornek_x, ornek_y = cls.sentetik_veri_uret()
        model = ModernRegulerVisionNet(girdi_kanali=3, sinif_sayisi=5, taban_kanal=32).to(cihaz)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        kriter = YumusatilmisCrossEntropyLoss(smoothing=label_smoothing)

        gecmis = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "mean_confidence": []
        }

        for epoch in range(1, toplam_epoch + 1):
            model.train()
            toplam_loss = 0.0

            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(cihaz), y_b.to(cihaz)
                optimizer.zero_grad()

                if artirma_turu == "mixup":
                    X_aug, y_a, y_b_target, lam = ModernArtirici.uygula_mixup(X_b, y_b, alpha=0.8)
                    ciktilar = model(X_aug)
                    loss = kriter(ciktilar, y_a, y_b_target, lam=lam)
                elif artirma_turu == "cutmix":
                    X_aug, y_a, y_b_target, lam = ModernArtirici.uygula_cutmix(X_b, y_b, alpha=1.0)
                    ciktilar = model(X_aug)
                    loss = kriter(ciktilar, y_a, y_b_target, lam=lam)
                else:  # none
                    ciktilar = model(X_b)
                    loss = kriter(ciktilar, y_b)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                toplam_loss += loss.item() * X_b.size(0)

            ort_train_loss = toplam_loss / len(train_loader.dataset)

            # Doğrulama ve Kalibrasyon / Güven Testi
            model.eval()
            val_loss = 0.0
            dogru = 0
            en_yuksek_olasiliklar = []

            with torch.no_grad():
                for X_v, y_v in val_loader:
                    X_v, y_v = X_v.to(cihaz), y_v.to(cihaz)
                    val_out = model(X_v)
                    v_l = F.cross_entropy(val_out, y_v)
                    val_loss += v_l.item() * X_v.size(0)

                    olasiliklar = F.softmax(val_out, dim=-1)
                    max_prob, tahmin = torch.max(olasiliklar, dim=-1)
                    dogru += (tahmin == y_v).sum().item()
                    en_yuksek_olasiliklar.extend(max_prob.cpu().numpy().tolist())

            ort_val_loss = val_loss / len(val_loader.dataset)
            val_acc = (dogru / len(val_loader.dataset)) * 100.0
            ort_guven = float(np.mean(en_yuksek_olasiliklar))

            gecmis["train_loss"].append(round(ort_train_loss, 5))
            gecmis["val_loss"].append(round(ort_val_loss, 5))
            gecmis["val_accuracy"].append(round(val_acc, 2))
            gecmis["mean_confidence"].append(round(ort_guven, 4))

        return {
            "deney_adi": deney_adi,
            "artirma_turu": artirma_turu.upper(),
            "label_smoothing": label_smoothing,
            "son_train_loss": gecmis["train_loss"][-1],
            "son_val_loss": gecmis["val_loss"][-1],
            "son_val_accuracy": gecmis["val_accuracy"][-1],
            "son_ort_guven": gecmis["mean_confidence"][-1],
            "gecmis": gecmis
        }

    @classmethod
    def tum_laboratuvari_kos(cls, toplam_epoch: int = 10) -> Dict[str, Any]:
        """Tüm deney kombinasyonlarını ve görsel örnekleri üretir."""
        # 1. Standart Eğitim (Baseline - No Reg)
        deney_1 = cls.tek_deney_kos(
            deney_adi="1. Standart (No Reg)",
            artirma_turu="none",
            label_smoothing=0.0,
            toplam_epoch=toplam_epoch
        )

        # 2. Mixup + Label Smoothing
        deney_2 = cls.tek_deney_kos(
            deney_adi="2. Mixup + LabelSmooth",
            artirma_turu="mixup",
            label_smoothing=0.1,
            toplam_epoch=toplam_epoch
        )

        # 3. CutMix + Label Smoothing
        deney_3 = cls.tek_deney_kos(
            deney_adi="3. CutMix + LabelSmooth",
            artirma_turu="cutmix",
            label_smoothing=0.1,
            toplam_epoch=toplam_epoch
        )

        # Görsel Örnekler Üret
        _, _, ornekler_x, ornekler_y = cls.sentetik_veri_uret()
        mixup_x, _, _, mix_lam = ModernArtirici.uygula_mixup(ornekler_x, ornekler_y, alpha=0.8)
        cutmix_x, _, _, cut_lam = ModernArtirici.uygula_cutmix(ornekler_x, ornekler_y, alpha=1.0)

        return {
            "deney_1": deney_1,
            "deney_2": deney_2,
            "deney_3": deney_3,
            "toplam_epoch": toplam_epoch,
            "ornekler": {
                "orijinal": ornekler_x[0].numpy(),
                "mixup": mixup_x[0].numpy(),
                "cutmix": cutmix_x[0].numpy(),
                "mix_lam": round(mix_lam, 2),
                "cut_lam": round(cut_lam, 2)
            }
        }
