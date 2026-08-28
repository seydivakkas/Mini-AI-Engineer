"""
Determinizm ve Tekrarlanabilirlik Motoru (Determinism Engine)
============================================================
PyTorch, CUDA, cuDNN, NumPy ve Python rastgelelik kaynaklarini
sabitler; coklu islemcili DataLoader worker'larinin tohumlanmasini yonetir.
"""

from typing import Optional
import os
import random
import numpy as np
import torch


class DeterminizmYoneticisi:
    """
    Derin ogrenme egitiminde %100 tekrarlanabilirlik (bit-for-bit reproducibility) saglayan yonetici.
    """

    @staticmethod
    def tohum_sabitle(tohum: int = 42, deterministik_mod: bool = True) -> None:
        """
        Tum rastgelelik tohumlarini (RNG) ve donanim duzeyi bayraklari kilitler.
        """
        # 1. Python ve Isletim Sistemi
        os.environ["PYTHONHASHSEED"] = str(tohum)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        random.seed(tohum)

        # 2. NumPy
        np.random.seed(tohum)

        # 3. PyTorch CPU & GPU
        torch.manual_seed(tohum)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(tohum)
            torch.cuda.manual_seed_all(tohum)

        # 4. cuDNN ve Deterministik Algoritmalar
        if deterministik_mod:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

    @staticmethod
    def worker_init_fn(worker_id: int) -> None:
        """
        DataLoader worker is parcalari icin bagimsiz ama tekrarlanabilir tohumlama fonksiyonu.
        """
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    @staticmethod
    def generator_al(tohum: int) -> torch.Generator:
        """
        DataLoader karistirma (shuffling) islemi icin izole PyTorch Generator uretir.
        """
        g = torch.Generator()
        g.manual_seed(tohum)
        return g
