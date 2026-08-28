"""
AMP Benchmark Motoru (FP32 vs FP16+GradScaler vs BF16 Throughput, Memory & Loss).
"""

from typing import Dict, Any, List, Tuple
import time
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class KapsamliGorusModeli(nn.Module):
    """AMP ve bellek yükünü net ölçmek için derin konvolüsyonel vizyon modeli."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.ozellikler = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.siniflandirici = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ozellikler(x)
        return self.siniflandirici(x)


class AMPBenchmarkMotoru:
    """FP32, AMP-FP16 ve BF16 eğitim modlarını throughput, VRAM ve sayısal kararlılık açısından kıyaslar."""

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def sentetik_veri_olustur(
        self,
        num_samples: int = 1000,
        img_size: int = 32,
        batch_size: int = 64
    ) -> DataLoader:
        torch.manual_seed(42)
        X = torch.randn(num_samples, 3, img_size, img_size)
        y = torch.randint(0, 10, (num_samples,))
        return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

    def calistir_kiyaslama(
        self,
        loader: DataLoader,
        epochs: int = 5,
        isinma_adimlari: int = 2
    ) -> Dict[str, Dict[str, Any]]:
        modlar = ["FP32 (Standart)", "AMP-FP16 (GradScaler)", "AMP-BF16"]
        sonuclar: Dict[str, Dict[str, Any]] = {}

        for mod in modlar:
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()

            model = KapsamliGorusModeli().to(self.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()

            use_amp = "AMP" in mod
            dtype = torch.float16 if "FP16" in mod else torch.bfloat16
            use_scaler = "FP16" in mod and self.device == "cuda"
            scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

            kayip_gecmisi: List[float] = []
            olcek_gecmisi: List[float] = []
            batch_sureleri: List[float] = []

            # Isınma (Warmup)
            for i, (inputs, targets) in enumerate(loader):
                if i >= isinma_adimlari:
                    break
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    with torch.amp.autocast(device_type=self.device, dtype=dtype):
                        out = model(inputs)
                        loss = criterion(out, targets)
                    if use_scaler:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                else:
                    out = model(inputs)
                    loss = criterion(out, targets)
                    loss.backward()
                    optimizer.step()

            if self.device == "cuda":
                torch.cuda.synchronize()

            baslangic_zamani = time.perf_counter()
            toplam_ornek = 0

            for epoch in range(1, epochs + 1):
                epoch_kayip = 0.0
                epoch_ornek = 0

                for inputs, targets in loader:
                    b_start = time.perf_counter()
                    inputs = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)

                    if use_amp:
                        with torch.amp.autocast(device_type=self.device, dtype=dtype):
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)

                        if use_scaler:
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            scaler.step(optimizer)
                            scaler.update()
                            olcek_gecmisi.append(scaler.get_scale())
                        else:
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                            olcek_gecmisi.append(1.0)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        olcek_gecmisi.append(1.0)

                    if self.device == "cuda":
                        torch.cuda.synchronize()

                    b_end = time.perf_counter()
                    batch_sureleri.append((b_end - b_start) * 1000.0)

                    epoch_kayip += loss.item() * len(targets)
                    epoch_ornek += len(targets)

                kayip_gecmisi.append(epoch_kayip / max(epoch_ornek, 1))
                toplam_ornek += epoch_ornek

            toplam_sure = time.perf_counter() - baslangic_zamani
            ort_batch_ms = float(sum(batch_sureleri) / len(batch_sureleri))
            throughput = float(toplam_ornek / max(toplam_sure, 1e-6))

            peak_vram_mb = 0.0
            if self.device == "cuda":
                peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
            else:
                param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
                peak_vram_mb = float(param_bytes / (1024 * 1024) * 4.0)

            sonuclar[mod] = {
                "toplam_sure_s": toplam_sure,
                "ort_batch_ms": ort_batch_ms,
                "throughput_ornek_s": throughput,
                "peak_vram_mb": peak_vram_mb,
                "kayip_gecmisi": kayip_gecmisi,
                "olcek_gecmisi": olcek_gecmisi,
                "nihai_loss": kayip_gecmisi[-1] if kayip_gecmisi else 0.0
            }

        return sonuclar
