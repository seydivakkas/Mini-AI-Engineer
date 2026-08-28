"""
Determinizm ve Tekrarlanabilirlik Yönetici Modülü (Day 97).
PyTorch çıkarım sürecinde bit-level determinizmi yapılandırır ve tensör hash kontrolleri yapar.
"""

import os
import random
import hashlib
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

from .model import MiniViTForImageClassification


class DeterminizmOrtami:
    """
    Tüm rastlantısallık kaynaklarını donduran ve PyTorch deterministik algoritmalarını
    etkinleştiren Context Manager / Yönetici sınıfı.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._eski_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        self._eski_deterministic = torch.are_deterministic_algorithms_enabled()
        self._eski_cudnn_det = torch.backends.cudnn.deterministic if torch.cuda.is_available() else False
        self._eski_cudnn_bench = torch.backends.cudnn.benchmark if torch.cuda.is_available() else False

    def etkinlestir(self):
        """Determinizm bayraklarını etkinleştirir ve seed'leri kilitler."""
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    def devre_disi_birak(self):
        """Eski determinizm ayarlarına geri döner."""
        if self._eski_cublas is not None:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = self._eski_cublas
        elif "CUBLAS_WORKSPACE_CONFIG" in os.environ:
            del os.environ["CUBLAS_WORKSPACE_CONFIG"]

        try:
            torch.use_deterministic_algorithms(self._eski_deterministic)
        except Exception:
            pass

        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = self._eski_cudnn_det
            torch.backends.cudnn.benchmark = self._eski_cudnn_bench

    def __enter__(self):
        self.etkinlestir()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.devre_disi_birak()


class BitHashHesaplayici:
    """Tensörlerin bellek baytları üzerinden SHA-256 karma özeti hesaplayan sınıf."""

    @staticmethod
    def tensor_bit_hash(tensor: torch.Tensor) -> str:
        """Tensörün ham baytları üzerinden deterministik SHA-256 hash üretir."""
        # CPU'ya al, contiguous float32 formatında bayt dizisini çıkar
        arr = tensor.detach().cpu().contiguous().numpy()
        return hashlib.sha256(arr.tobytes()).hexdigest()


class DeterminizmDenetleyicisi:
    """Aynı model ve girdi üzerinde ardışık çıkarımların determinizmini test eden sınıf."""

    def __init__(self, model: MiniViTForImageClassification):
        self.model = model
        self.cihaz = next(model.parameters()).device
        self.model.eval()

    def ardil_cikarim_testi(
        self,
        girdi: torch.Tensor,
        tekrar_sayisi: int = 100,
    ) -> Dict[str, Any]:
        """
        Aynı girdi tensörü ile belirtilen sayıda çıkarım yapar ve
        tüm adımlarda bit-level eşitliği doğrular.
        """
        girdi = girdi.to(self.cihaz)
        bit_hashler = []
        ciktilar = []

        with torch.no_grad():
            for _ in range(tekrar_sayisi):
                out = self.model(girdi).logits
                h = BitHashHesaplayici.tensor_bit_hash(out)
                bit_hashler.append(h)
                ciktilar.append(out.detach().cpu())

        # İlk çıktıya göre maksimum sayısal sapmalar
        ref_out = ciktilar[0]
        maks_sapmalar = [torch.max(torch.abs(out - ref_out)).item() for out in ciktilar]
        global_maks_sapma = max(maks_sapmalar)
        benzersiz_hashler = set(bit_hashler)

        tam_deterministik = (len(benzersiz_hashler) == 1) and (global_maks_sapma == 0.0)

        return {
            "tam_deterministik": tam_deterministik,
            "tekrar_sayisi": tekrar_sayisi,
            "benzersiz_hash_sayisi": len(benzersiz_hashler),
            "ornek_hash": bit_hashler[0],
            "global_maks_sapma": global_maks_sapma,
            "sapmalar_listesi": maks_sapmalar,
        }
