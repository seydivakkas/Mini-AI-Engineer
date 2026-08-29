"""
Self-Instruct Sentetik Akıl Yürütme ve Damıtma Verisi Üreteci (Day 119).
Büyük Öğretmen Modelden CoT (Düşünce Zinciri) ve nihai çözüm sentetik veri seti oluşturur.
"""

from typing import List, Dict, Any, Tuple
import torch
import random


class SelfInstructUreteci:
    """Öğretmen modelden öğrenciye aktarılacak sentetik muhakeme veri seti üreteci."""

    GOREVLER = [
        "Matematiksel İspat: Asal sayıların sonsuzluğunu Euclid yöntemiyle kanıtlayın.",
        "Algoritma Tasarımı: O(N) zaman ve O(1) bellek ile iki bağlı listeyi birleştirin.",
        "Sistem Mimarisi: 100k QPS kaldıran distributed cache mimarisi tasarlayın.",
        "Kriptografi: Diffie-Hellman anahtar değişim protokolünün matematiğini açıklayın.",
    ]

    def __init__(self, vocab_size: int = 1000, max_seq_len: int = 32, seed: int = 42):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        random.seed(seed)

    def sentetik_batch_uret(self, batch_size: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Öğrenci eğitimi için sentetik girdi (x) ve hedef (y) tensörleri üretir.
        Dönüş: (input_ids: [B, S], target_ids: [B, S])
        """
        inputs = torch.randint(1, self.vocab_size - 1, (batch_size, self.max_seq_len))
        # Otoregresif bir sonraki token hedefi (shifted targets)
        targets = torch.roll(inputs, shifts=-1, dims=-1)
        targets[:, -1] = 0  # Son token pad/end
        return inputs, targets
