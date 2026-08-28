"""
Veri Boru Hatti Performans Karsilastiricisi (Pipeline Benchmark Suite)
====================================================================
1. Standart torchvision boru hatti,
2. Albumentations C++ tabanli hizli boru hatti,
3. Albumentations + CUDA Stream Asenkron Prefetcher boru hatlarini
throughput (FPS), gecikme (ms) ve hizlanma katsayilari acisindan kiyaslar.
"""

from typing import Dict, Any, List, Tuple
import time
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from src.veri_donusturucu import YuksekPerformansArtirici
from src.veri_seti import SentetikGorselVeriSeti
from src.cuda_prefetcher import CUDAPrefetcher


class TorchvisionSentetikSet(Dataset):
    """Torchvision için PIL tabanlı veri seti."""

    def __init__(self, veriler_np: np.ndarray, etiketler_np: np.ndarray, donusum: T.Compose) -> None:
        self.veriler_np = veriler_np
        self.etiketler_np = etiketler_np
        self.donusum = donusum

    def __len__(self) -> int:
        return len(self.veriler_np)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        gorsel_pil = Image.fromarray(self.veriler_np[idx])
        gorsel_t = self.donusum(gorsel_pil)
        return gorsel_t, int(self.etiketler_np[idx])


class BoruHattiKarsilastirici:
    """
    Üç farklı veri boru hattının hız ve gecikme metriklerini ölçen test paketi.
    """

    @classmethod
    def benchmark_kos(
        cls,
        ornek_sayisi: int = 1200,
        batch_size: int = 64,
        gorsel_boyutu: Tuple[int, int] = (64, 64),
        tekrar_sayisi: int = 2
    ) -> Dict[str, Any]:
        """
        Her 3 boru hattını aynı sentetik veri üzerinde koşturur ve sonuçları özetler.
        """
        cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        H, W = gorsel_boyutu

        # Ortak sentetik ham veri
        rng = np.random.RandomState(42)
        ham_veriler = rng.randint(0, 256, size=(ornek_sayisi, H, W, 3), dtype=np.uint8)
        ham_etiketler = rng.randint(0, 10, size=(ornek_sayisi,), dtype=np.int64)

        # -----------------------------------------------------------------
        # 1. Boru Hattı: Standart Torchvision (PIL)
        # -----------------------------------------------------------------
        tv_transforms = T.Compose([
            T.RandomResizedCrop((H, W), scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        tv_dataset = TorchvisionSentetikSet(ham_veriler, ham_etiketler, tv_transforms)
        tv_loader = DataLoader(tv_dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())

        # Isınma (Warmup)
        for _ in range(1):
            for x, y in tv_loader:
                _ = x.to(cihaz)

        tv_sureler = []
        for _ in range(tekrar_sayisi):
            t0 = time.perf_counter()
            for x, y in tv_loader:
                x_dev = x.to(cihaz)
                _ = x_dev.mean()
            t1 = time.perf_counter()
            tv_sureler.append(t1 - t0)

        tv_ort_sure = float(np.mean(tv_sureler))
        tv_fps = ornek_sayisi / tv_ort_sure
        tv_batch_ms = (tv_ort_sure / len(tv_loader)) * 1000.0

        # -----------------------------------------------------------------
        # 2. Boru Hattı: Albumentations (C++ OpenCV)
        # -----------------------------------------------------------------
        albu_engine = YuksekPerformansArtirici(hedef_boyut=gorsel_boyutu)
        albu_dataset = SentetikGorselVeriSeti(
            ornek_sayisi=ornek_sayisi,
            gorsel_boyutu=gorsel_boyutu,
            donusum=albu_engine.egitim_donustur
        )
        albu_dataset.veriler = ham_veriler
        albu_dataset.etiketler = ham_etiketler
        albu_loader = DataLoader(albu_dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())

        # Isınma
        for _ in range(1):
            for x, y in albu_loader:
                _ = x.to(cihaz)

        albu_sureler = []
        for _ in range(tekrar_sayisi):
            t0 = time.perf_counter()
            for x, y in albu_loader:
                x_dev = x.to(cihaz)
                _ = x_dev.mean()
            t1 = time.perf_counter()
            albu_sureler.append(t1 - t0)

        albu_ort_sure = float(np.mean(albu_sureler))
        albu_fps = ornek_sayisi / albu_ort_sure
        albu_batch_ms = (albu_ort_sure / len(albu_loader)) * 1000.0

        # -----------------------------------------------------------------
        # 3. Boru Hattı: Albumentations + CUDA Prefetcher
        # -----------------------------------------------------------------
        prefetch_sureler = []
        for _ in range(tekrar_sayisi):
            prefetcher = CUDAPrefetcher(albu_loader, cihaz=cihaz)
            t0 = time.perf_counter()
            for x_dev, y_dev in prefetcher:
                _ = x_dev.mean()
            t1 = time.perf_counter()
            prefetch_sureler.append(t1 - t0)

        pref_ort_sure = float(np.mean(prefetch_sureler))
        pref_fps = ornek_sayisi / pref_ort_sure
        pref_batch_ms = (pref_ort_sure / len(albu_loader)) * 1000.0

        hizlanma_albu = albu_fps / tv_fps
        hizlanma_prefetch = pref_fps / tv_fps

        return {
            "ornek_sayisi": ornek_sayisi,
            "batch_size": batch_size,
            "gorsel_boyutu": gorsel_boyutu,
            "cihaz": str(cihaz),
            "torchvision": {
                "toplam_sure_sn": round(tv_ort_sure, 4),
                "fps": round(tv_fps, 1),
                "batch_gecikmesi_ms": round(tv_batch_ms, 2),
                "hizlanma_kat": 1.00
            },
            "albumentations_cpu": {
                "toplam_sure_sn": round(albu_ort_sure, 4),
                "fps": round(albu_fps, 1),
                "batch_gecikmesi_ms": round(albu_batch_ms, 2),
                "hizlanma_kat": round(hizlanma_albu, 2)
            },
            "albumentations_prefetcher": {
                "toplam_sure_sn": round(pref_ort_sure, 4),
                "fps": round(pref_fps, 1),
                "batch_gecikmesi_ms": round(pref_batch_ms, 2),
                "hizlanma_kat": round(hizlanma_prefetch, 2)
            }
        }
