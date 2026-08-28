"""
Çökmeye Dayanıklı Güvenli Checkpoint Yöneticisi
==============================================
Atomik disk yazma (Atomic I/O), tam durum restorasyonu (Model, Optimizer, Scheduler, RNG, Scaler)
ve Top-K checkpoint saklama politikalarını yöneten kurumsal sınıf modül.
"""

from typing import Dict, Any, Optional, List, Tuple
import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class GuvenliCheckpointYoneticisi:
    """
    Atomik ve çökmeye dayanıklı Checkpoint yönetim sınıfı.
    """

    def __init__(self, kayit_dizini: str = "checkpoints", maks_saklanan: int = 3) -> None:
        self.kayit_dizini = os.path.abspath(kayit_dizini)
        self.maks_saklanan = maks_saklanan
        os.makedirs(self.kayit_dizini, exist_ok=True)
        self.en_iyi_metrik: float = float("inf")
        self.gecmis_checkpointler: List[Tuple[float, str]] = []  # (val_loss, dosya_yolu)

    @staticmethod
    def rng_durumu_topla() -> Dict[str, Any]:
        """Tüm rastgele sayı üreteçlerinin (RNG) anlık durumlarını yakalar."""
        rng_durum = {
            "torch_cpu": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python_random": random.getstate()
        }
        if torch.cuda.is_available():
            rng_durum["torch_cuda"] = torch.cuda.get_rng_state_all()
        return rng_durum

    @staticmethod
    def rng_durumu_geri_yukle(rng_durum: Dict[str, Any]) -> None:
        """Saklanan RNG durumlarını geri yükleyerek deterministik devamlılık sağlar."""
        try:
            if "torch_cpu" in rng_durum and rng_durum["torch_cpu"] is not None:
                cpu_state = rng_durum["torch_cpu"]
                if isinstance(cpu_state, torch.Tensor):
                    cpu_state = cpu_state.cpu().to(torch.uint8)
                torch.set_rng_state(cpu_state)
            if "numpy" in rng_durum and rng_durum["numpy"] is not None:
                np.random.set_state(rng_durum["numpy"])
            if "python_random" in rng_durum and rng_durum["python_random"] is not None:
                random.setstate(rng_durum["python_random"])
            if torch.cuda.is_available() and "torch_cuda" in rng_durum and rng_durum["torch_cuda"] is not None:
                cuda_states = rng_durum["torch_cuda"]
                if isinstance(cuda_states, list):
                    cuda_states = [s.cpu().to(torch.uint8) if isinstance(s, torch.Tensor) else s for s in cuda_states]
                torch.cuda.set_rng_state_all(cuda_states)
        except Exception as e:
            # RNG yükleme uyarısı ver ama eğitimi durdurma
            print(f"[!] RNG durumu geri yuklenirken ikincil uyari (ihmal edilebilir): {e}")


    def kaydet_atomik(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        val_loss: float,
        val_acc: float,
        ek_meta: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Modeli ve tüm eğitim bileşenlerini atomik olarak diske yazar.
        Doğrudan hedef dosyaya yazmak yerine geçici (.tmp) dosyaya yazıp os.replace() ile taşır.
        """
        hedef_dosya = os.path.join(self.kayit_dizini, f"checkpoint_epoch_{epoch:03d}.pt")
        gecici_dosya = f"{hedef_dosya}.tmp"

        durum_paketi = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "rng_state": self.rng_durumu_topla(),
            "ek_meta": ek_meta or {}
        }

        # 1. Adım: Geçici dosyaya güvenle kaydet
        torch.save(durum_paketi, gecici_dosya)

        # 2. Adım: Atomik dosya taşıma / isim değiştirme (Atomic Replace)
        os.replace(gecici_dosya, hedef_dosya)

        # 3. Adım: Son durum dosyasını güncelle (last.pt)
        son_durum_dosyasi = os.path.join(self.kayit_dizini, "last.pt")
        son_durum_tmp = f"{son_durum_dosyasi}.tmp"
        torch.save(durum_paketi, son_durum_tmp)
        os.replace(son_durum_tmp, son_durum_dosyasi)

        # 4. Adım: En iyi model kontrolü (best.pt)
        if val_loss < self.en_iyi_metrik:
            self.en_iyi_metrik = val_loss
            en_iyi_dosya = os.path.join(self.kayit_dizini, "best.pt")
            en_iyi_tmp = f"{en_iyi_dosya}.tmp"
            torch.save(durum_paketi, en_iyi_tmp)
            os.replace(en_iyi_tmp, en_iyi_dosya)

        # 5. Adım: Top-K Budama (Disk dolmasını önleme)
        self.gecmis_checkpointler.append((val_loss, hedef_dosya))
        self._top_k_budama()

        return hedef_dosya

    def _top_k_budama(self) -> None:
        """Yalnızca en iyi ve en son checkpointleri saklayıp eski dosyaları siler."""
        if len(self.gecmis_checkpointler) <= self.maks_saklanan:
            return

        # val_loss'a göre sırala (en düşük en başta)
        self.gecmis_checkpointler.sort(key=lambda x: x[0])

        # En iyi maks_saklanan haricindekileri diskten temizle
        korunanlar = set(path for _, path in self.gecmis_checkpointler[:self.maks_saklanan])
        yeni_liste = []

        for loss_val, path in self.gecmis_checkpointler:
            if path in korunanlar:
                yeni_liste.append((loss_val, path))
            else:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

        self.gecmis_checkpointler = yeni_liste

    def yukle_ve_geri_yukle(
        self,
        dosya_yolu: str,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        cihaz: torch.device
    ) -> Dict[str, Any]:
        """
        Checkpoint dosyasından Model, Optimizer, Scheduler ve RNG durumlarını eksiksiz geri yükler.
        """
        if not os.path.exists(dosya_yolu):
            raise FileNotFoundError(f"Checkpoint dosyasi bulunamadi: {dosya_yolu}")

        paket = torch.load(dosya_yolu, map_location=cihaz, weights_only=False)

        # 1. Model Ağırlıklarını Geri Yükle
        model.load_state_dict(paket["model_state_dict"])

        # 2. Optimizer Durumunu Geri Yükle (Momentum ve İkinci Moment vektörleri)
        optimizer.load_state_dict(paket["optimizer_state_dict"])

        # 3. Scheduler Durumunu Geri Yükle (Öğrenme Oranı eğrisi adımı)
        scheduler.load_state_dict(paket["scheduler_state_dict"])

        # 4. RNG Durumunu Geri Yükle (Determinizm devamlılığı)
        if "rng_state" in paket:
            self.rng_durumu_geri_yukle(paket["rng_state"])

        return {
            "epoch": paket["epoch"],
            "val_loss": paket["val_loss"],
            "val_acc": paket["val_acc"],
            "ek_meta": paket.get("ek_meta", {})
        }
