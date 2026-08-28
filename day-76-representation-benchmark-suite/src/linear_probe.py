"""
Lineer Yoklama (Linear Probing) Değerlendirme Protokolü
------------------------------------------------------
Dondurulmuş öznitelik temsilleri üzerine tek katmanlı lineer sınıflandırıcı
eğiterek temsil uzayının doğrusal ayrışabilirliğini ölçen protokol.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW


class DogrusalYoklayici(nn.Module):
    """
    Lineer Sınıflandırıcı Kafası: f(h) = W * h + b
    """
    def __init__(self, temsil_boyutu: int, sinif_sayisi: int):
        super().__init__()
        self.fc = nn.Linear(temsil_boyutu, sinif_sayisi)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)


class LinearProbeProtokolu:
    """
    Linear Probing eğitim ve değerlendirme yöneticisi.
    """
    def __init__(
        self,
        temsil_boyutu: int,
        sinif_sayisi: int,
        ogrenme_orani: float = 1e-2,
        agirlik_cezasi: float = 1e-4,
        cihaz: str = "cpu"
    ):
        self.temsil_boyutu = temsil_boyutu
        self.sinif_sayisi = sinif_sayisi
        self.ogrenme_orani = ogrenme_orani
        self.agirlik_cezasi = agirlik_cezasi
        self.cihaz = cihaz

    def egit_ve_degerlendir(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
        etiket_orani: float = 1.0,
        epoch_sayisi: int = 15,
        batch_boyutu: int = 64
    ) -> Dict[str, Any]:
        """
        Belirtilen etiket oranıyla (%1, %10, %100) lineer sınıflandırıcıyı eğitir ve doğruluğu ölçer.
        """
        N_train = x_train.size(0)
        
        # Few-shot örnekleme
        if etiket_orani < 1.0:
            kullanilacak_n = max(self.sinif_sayisi, int(N_train * etiket_orani))
            perm = torch.randperm(N_train)
            secilen_idx = perm[:kullanilacak_n]
            x_train_sub = x_train[secilen_idx]
            y_train_sub = y_train[secilen_idx]
        else:
            x_train_sub = x_train
            y_train_sub = y_train

        train_ds = TensorDataset(x_train_sub, y_train_sub)
        val_ds = TensorDataset(x_val, y_val)
        
        train_loader = DataLoader(train_ds, batch_size=batch_boyutu, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_boyutu, shuffle=False)

        model = DogrusalYoklayici(self.temsil_boyutu, self.sinif_sayisi).to(self.cihaz)
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=self.ogrenme_orani, weight_decay=self.agirlik_cezasi)

        # Eğitim
        model.train()
        for ep in range(epoch_sayisi):
            for bx, by in train_loader:
                bx, by = bx.to(self.cihaz), by.to(self.cihaz)
                optimizer.zero_grad()
                logits = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

        # Doğrulama Değerlendirmesi
        model.eval()
        dogru = 0
        toplam = 0
        val_loss = 0.0
        adim = 0

        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(self.cihaz), by.to(self.cihaz)
                logits = model(bx)
                loss = criterion(logits, by)
                val_loss += loss.item()
                adim += 1
                
                tahminler = torch.argmax(logits, dim=1)
                dogru += (tahminler == by).sum().item()
                toplam += by.size(0)

        acc = (dogru / max(1, toplam)) * 100.0
        ortalama_val_loss = val_loss / max(1, adim)

        return {
            "etiket_orani": etiket_orani,
            "kullanilan_ornek_sayisi": x_train_sub.size(0),
            "dogruluk_yuzdesi": acc,
            "dogrulama_kaybi": ortalama_val_loss
        }
