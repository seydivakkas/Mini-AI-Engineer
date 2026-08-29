"""
Kirchenbauer LLM Filigran (Watermarking) Ekleme Modülü (Day 118).
Önceki token özetine (Hash) bağlı olarak sözlüğü Yeşil/Kırmızı listeye böler ve Yeşil logitlere delta yanlılığı ekler.
"""

from typing import List, Set, Tuple
import torch
import torch.nn.functional as F
import hashlib


class KirchenbauerWatermarker:
    """Kirchenbauer et al. (ICML 2023) algoritmasıyla çalışan filigran enjektörü."""

    def __init__(
        self,
        vocab_size: int = 1000,
        gamma: float = 0.5,
        delta: float = 2.5,
        gizli_anahtar: int = 15485863,
    ):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.delta = delta
        self.gizli_anahtar = gizli_anahtar
        self.green_size = int(vocab_size * gamma)

    def _yesil_listeyi_uret(self, onceki_token: int) -> Set[int]:
        """Önceki token ve gizli anahtardan deterministik yeşil liste üretir."""
        tohum_verisi = f"{onceki_token}_{self.gizli_anahtar}".encode("utf-8")
        hash_degeri = int(hashlib.sha256(tohum_verisi).hexdigest(), 16) % (2**32)

        gen = torch.Generator().manual_seed(hash_degeri)
        permutasyon = torch.randperm(self.vocab_size, generator=gen).tolist()
        return set(permutasyon[: self.green_size])

    def filigranli_logits(self, logits: torch.Tensor, onceki_token: int) -> torch.Tensor:
        """Yeşil listedeki tokenların logit değerlerini delta kadar artırır."""
        yesil_liste = self._yesil_listeyi_uret(onceki_token)
        yesil_indeksler = torch.tensor(list(yesil_liste), dtype=torch.long, device=logits.device)

        yenilenmis_logits = logits.clone()
        yenilenmis_logits[yesil_indeksler] += self.delta
        return yenilenmis_logits

    def token_dizisi_uret(
        self,
        baslangic_token: int,
        uzunluk: int = 50,
        filigran_aktif: bool = True,
        sicaklik: float = 0.8,
    ) -> List[int]:
        """Simüle edilmiş model dağılımından filigranlı veya filigransız token dizisi üretir."""
        ureten_dizi = [baslangic_token]

        for _ in range(uzunluk):
            onceki = ureten_dizi[-1]
            # Standart sentetik model logit dağılımı
            temel_logits = torch.randn(self.vocab_size) * 1.5

            if filigran_aktif:
                islenmis_logits = self.filigranli_logits(temel_logits, onceki)
            else:
                islenmis_logits = temel_logits

            olasiliklar = F.softmax(islenmis_logits / max(0.1, sicaklik), dim=-1)
            sonraki_token = int(torch.multinomial(olasiliklar, num_samples=1).item())
            ureten_dizi.append(sonraki_token)

        return ureten_dizi
