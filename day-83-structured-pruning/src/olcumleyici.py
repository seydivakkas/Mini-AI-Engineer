"""
Model Performans, Gecikme ve İnce Ayar Ölçümleyicisi
---------------------------------------------------
Budanmış modellerin doğruluk, parametre, bellek boyutu (MB), çıkarım gecikmesi (Latency ms)
ve FPS değerlerini ölçen ve ince ayar (Fine-Tuning) ile doğruluk toparlayan yönetici.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Dict, Tuple, List, Any
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class PerformansOlcumleyici:
    """
    Model performans ve kaynak kullanımını ölçümleyen analiz kütüphanesi.
    """
    @staticmethod
    def parametre_sayisi(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())

    @staticmethod
    def model_boyutu_mb(model: nn.Module) -> float:
        toplam_bayt = sum(p.numel() * p.element_size() for p in model.parameters())
        toplam_bayt += sum(b.numel() * b.element_size() for b in model.buffers())
        return toplam_bayt / (1024 * 1024)

    @classmethod
    def cikarim_gecikmesi_ve_fps(
        cls,
        model: nn.Module,
        girdi_sekli: Tuple[int, int, int, int] = (1, 3, 32, 32),
        cihaz: str = "cpu",
        isinma: int = 15,
        tekrar: int = 80
    ) -> Dict[str, float]:
        """
        Modelin tekil çıkarım gecikmesini (ms) ve saniye başına kare sayısını (FPS) ölçer.
        """
        model = model.to(cihaz).eval()
        dummy_input = torch.randn(*girdi_sekli, device=cihaz)

        # Isınma turları
        with torch.no_grad():
            for _ in range(isinma):
                _ = model(dummy_input)

        if cihaz == "cuda":
            torch.cuda.synchronize()

        baslangic = time.perf_counter()
        with torch.no_grad():
            for _ in range(tekrar):
                _ = model(dummy_input)

        if cihaz == "cuda":
            torch.cuda.synchronize()

        toplam_sure = time.perf_counter() - baslangic
        ortalama_gecikme_ms = (toplam_sure / tekrar) * 1000.0
        fps = tekrar / toplam_sure

        return {
            "gecikme_ms": ortalama_gecikme_ms,
            "fps": fps
        }

    @classmethod
    def dogruluk_olc(cls, model: nn.Module, data_loader: DataLoader, cihaz: str = "cpu") -> float:
        model = model.to(cihaz).eval()
        toplam_dogru = 0
        toplam_ornek = 0

        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(cihaz), y.to(cihaz)
                tahminler = model(x).argmax(dim=-1)
                toplam_dogru += (tahminler == y).sum().item()
                toplam_ornek += x.size(0)

        return (toplam_dogru / max(1, toplam_ornek)) * 100.0

    @classmethod
    def ince_ayar_yap(
        cls,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epok_sayisi: int = 3,
        lr: float = 3e-4,
        cihaz: str = "cpu"
    ) -> List[float]:
        """
        Budama sonrası doğruluk toparlanması için kısa ince ayar (Fine-Tuning) yapar.
        """
        model = model.to(cihaz).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        val_gecmisi = []

        for _ in range(epok_sayisi):
            model.train()
            for x, y in train_loader:
                x, y = x.to(cihaz), y.to(cihaz)
                optimizer.zero_grad()
                loss = nn.functional.cross_entropy(model(x), y)
                loss.backward()
                optimizer.step()

            val_acc = cls.dogruluk_olc(model, val_loader, cihaz)
            val_gecmisi.append(val_acc)

        return val_gecmisi
