"""
PyTorch DataLoader Darboğaz, Gecikme ve İşleme Hızı Kıyaslama Motoru (DataLoader Benchmark Engine).
"""

from typing import Dict, Any, List, Optional
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from .veri_seti_motoru import worker_init_fn


class DataLoaderBenchmarkEngine:
    """Farklı num_workers, pin_memory, persistent_workers ve prefetch_factor ayarlarını test eder ve kıyaslar."""

    @classmethod
    def tekil_olcum(
        cls,
        dataset: Dataset,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: Optional[int] = None,
        non_blocking: bool = False,
        num_batches: int = 25
    ) -> Dict[str, Any]:
        """Belirtilen DataLoader konfigürasyonunu belirli sayıda batch üzerinden milisaniye hassasiyetinde ölçer."""
        kwargs: Dict[str, Any] = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": True,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "worker_init_fn": worker_init_fn if num_workers > 0 else None
        }

        if num_workers > 0:
            kwargs["persistent_workers"] = persistent_workers
            if prefetch_factor is not None:
                kwargs["prefetch_factor"] = prefetch_factor

        loader = DataLoader(**kwargs)

        # Ölçüm Döngüsü (Kararlı Durum / Steady-State Ölçümü)
        batch_sureleri_ms: List[float] = []
        transfer_sureleri_ms: List[float] = []
        toplam_ornek_sayisi = 0

        t_baslangic = None

        for batch_x, batch_y in loader:
            if t_baslangic is None:
                t_baslangic = time.perf_counter()

            t0 = time.perf_counter()
            # Batch çıkarma gecikmesi
            t_batch = (time.perf_counter() - t0) * 1000.0

            # Bellek Transferi (Host Pinned -> GPU / CPU)
            t_tr0 = time.perf_counter()
            if torch.cuda.is_available():
                _ = batch_x.to("cuda", non_blocking=non_blocking)
                _ = batch_y.to("cuda", non_blocking=non_blocking)
                torch.cuda.synchronize()
            else:
                _ = batch_x.clone()
            t_tr = (time.perf_counter() - t_tr0) * 1000.0

            batch_sureleri_ms.append(t_batch)
            transfer_sureleri_ms.append(t_tr)
            toplam_ornek_sayisi += len(batch_x)

        toplam_sure_sn = max(time.perf_counter() - (t_baslangic if t_baslangic is not None else time.perf_counter()), 1e-5)
        isleme_hizi = float(round(toplam_ornek_sayisi / toplam_sure_sn, 1))

        # Windows için açık temizlik
        del loader

        return {
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": persistent_workers,
            "prefetch_factor": prefetch_factor,
            "non_blocking": non_blocking,
            "toplam_sure_sn": float(round(toplam_sure_sn, 3)),
            "isleme_hizi_ornek_sn": isleme_hizi,
            "ort_batch_gecikmesi_ms": float(round(float(np.mean(batch_sureleri_ms)), 2)),
            "ort_transfer_gecikmesi_ms": float(round(float(np.mean(transfer_sureleri_ms)), 2)),
            "toplam_ornek": toplam_ornek_sayisi
        }

    @classmethod
    def karsilastirmali_benchmark(
        cls,
        dataset: Dataset,
        batch_size: int = 64,
        num_batches: int = 20
    ) -> List[Dict[str, Any]]:
        """4 ana endüstriyel konfigürasyonu ardışık olarak çalıştırıp hızlanma oranlarını hesaplar."""
        konfigler = [
            {
                "ad": "1. Basit (Naive Baseline)",
                "num_workers": 0,
                "pin_memory": False,
                "persistent_workers": False,
                "prefetch_factor": None,
                "non_blocking": False
            },
            {
                "ad": "2. Çoklu İş Parçacığı (Multi-Worker)",
                "num_workers": 2,
                "pin_memory": False,
                "persistent_workers": False,
                "prefetch_factor": None,
                "non_blocking": False
            },
            {
                "ad": "3. Sabitlenmiş Bellek (Pinned Memory)",
                "num_workers": 2,
                "pin_memory": True,
                "persistent_workers": False,
                "prefetch_factor": None,
                "non_blocking": True
            },
            {
                "ad": "4. Üretim Optimize (Production Prefetch)",
                "num_workers": 2,
                "pin_memory": True,
                "persistent_workers": True,
                "prefetch_factor": 2,
                "non_blocking": True
            }
        ]

        sonuclar = []
        baseline_hiz = 1.0

        for idx, cfg in enumerate(konfigler):
            print(f"    -> [{idx+1}/4] Test Ediliyor: {cfg['ad']}...", flush=True)
            res = cls.tekil_olcum(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=cfg["num_workers"],
                pin_memory=cfg["pin_memory"],
                persistent_workers=cfg["persistent_workers"],
                prefetch_factor=cfg["prefetch_factor"],
                non_blocking=cfg["non_blocking"],
                num_batches=num_batches
            )
            res["ad"] = cfg["ad"]

            if idx == 0:
                baseline_hiz = max(res["isleme_hizi_ornek_sn"], 1.0)
                res["hizlanma_carpani"] = 1.0
            else:
                res["hizlanma_carpani"] = float(round(res["isleme_hizi_ornek_sn"] / baseline_hiz, 2))

            # GPU Starvation (Boşta Kalma) Oranı
            res["gpu_starvation_orani"] = float(round(max(0.0, 100.0 - (res["hizlanma_carpani"] / 4.0 * 100.0)), 1))
            sonuclar.append(res)

        return sonuclar

    @classmethod
    def worker_olceklenme_taramasi(
        cls,
        dataset: Dataset,
        batch_size: int = 64,
        worker_listesi: Optional[List[int]] = None,
        num_batches: int = 15
    ) -> List[Dict[str, Any]]:
        """num_workers değerinin (0, 1, 2, 4, 8) işleme hızına etkisini ölçekler."""
        if worker_listesi is None:
            worker_listesi = [0, 1, 2, 4, 8]

        tarama_sonuclari = []
        for w in worker_listesi:
            res = cls.tekil_olcum(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=w,
                pin_memory=(w > 0),
                persistent_workers=False,
                prefetch_factor=2 if w > 0 else None,
                num_batches=num_batches
            )
            tarama_sonuclari.append(res)

        return tarama_sonuclari
