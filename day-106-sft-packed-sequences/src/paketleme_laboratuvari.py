"""
SFT Token Paketleme Kıyaslama ve Karşılaştırma Laboratuvarı (Day 106).
Standart Padding vs Token Packing israf analizi, throughput ve eğitim hızlanması.
"""

import time
import random
from typing import Dict, Any, List, Tuple
import numpy as np
import torch

from .token_paketleyici import Ornek, TokenPaketleyici, PaketlenmisDizi
from .sft_egitim_motoru import SFTEgitimMotoru


class PaketlemeLaboratuvari:
    """Standart Paddingli SFT ile Token Packed SFT yöntemlerini karşılaştıran benchmark motoru."""

    def __init__(
        self,
        max_seq_len: int = 1024,
        vocab_size: int = 1000,
        cihaz: torch.device = torch.device("cpu"),
    ):
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.cihaz = cihaz

    def sentetik_sft_veri_seti_uret(self, ornek_sayisi: int = 300) -> List[Ornek]:
        """Gerçekçi değişken uzunluklu (log-normal dağılım) SFT sohbet verisi üretir."""
        random.seed(42)
        torch.manual_seed(42)

        ornekler = []
        for _ in range(ornek_sayisi):
            # Prompt: 20 ile 120 token arası
            p_len = int(np.clip(np.random.lognormal(mean=3.5, sigma=0.5), 20, 150))
            # Response: 30 ile 350 token arası
            r_len = int(np.clip(np.random.lognormal(mean=4.5, sigma=0.6), 30, 400))

            p_ids = [random.randint(1, self.vocab_size - 1) for _ in range(p_len)]
            r_ids = [random.randint(1, self.vocab_size - 1) for _ in range(r_len)]
            ornekler.append(Ornek(prompt_ids=p_ids, response_ids=r_ids))

        return ornekler

    def padding_israf_analizi(self, ornekler: List[Ornek], batch_size: int = 4) -> Dict[str, Any]:
        """
        Standart Paddingli Batchleme ile Token Packing arasındaki token israfını hesaplar.
        """
        toplam_gercek_token = sum(o.toplam_uzunluk for o in ornekler)

        # 1. Standart Padding Yaklaşımı
        standart_batch_sayisi = (len(ornekler) + batch_size - 1) // batch_size
        standart_toplam_islenen_token = 0

        for i in range(0, len(ornekler), batch_size):
            batch = ornekler[i : i + batch_size]
            max_len_in_batch = max(o.toplam_uzunluk for o in batch)
            standart_toplam_islenen_token += len(batch) * max_len_in_batch

        standart_pad_token = standart_toplam_islenen_token - toplam_gercek_token
        standart_israf_orani = (standart_pad_token / standart_toplam_islenen_token) * 100.0

        # 2. Token Packing (FFD) Yaklaşımı
        paketleyici = TokenPaketleyici(max_seq_len=self.max_seq_len)
        paketlenmis_diziler = paketleyici.paketle(ornekler)

        packed_toplam_islenen_token = len(paketlenmis_diziler) * self.max_seq_len
        packed_pad_token = packed_toplam_islenen_token - toplam_gercek_token
        packed_israf_orani = (packed_pad_token / packed_toplam_islenen_token) * 100.0
        ortalama_doluluk = float(np.mean([p.doluluk_orani for p in paketlenmis_diziler])) * 100.0

        return {
            "toplam_ornek_sayisi": len(ornekler),
            "toplam_gercek_token": toplam_gercek_token,
            "standart": {
                "toplam_islenen_token": standart_toplam_islenen_token,
                "pad_token_sayisi": standart_pad_token,
                "israf_orani_yuzde": round(standart_israf_orani, 2),
                "adim_sayisi": standart_batch_sayisi,
            },
            "token_packing": {
                "toplam_islenen_token": packed_toplam_islenen_token,
                "pad_token_sayisi": packed_pad_token,
                "israf_orani_yuzde": round(packed_israf_orani, 2),
                "adim_sayisi": len(paketlenmis_diziler),
                "ortalama_doluluk_yuzde": round(ortalama_doluluk, 2),
            },
            "adim_tasarrufu_kat": round(standart_batch_sayisi / max(len(paketlenmis_diziler), 1), 2),
            "token_tasarrufu_kat": round(standart_toplam_islenen_token / packed_toplam_islenen_token, 2),
        }

    def hiz_ve_throughput_karsilastir(
        self,
        ornekler: List[Ornek],
        model: SFTEgitimMotoru,
        iterasyon: int = 15,
    ) -> Dict[str, Dict[str, Any]]:
        """Eğitim adımı süresini ve efektif örnek/saniye hızını ölçer."""
        model.to(self.cihaz).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        paketleyici = TokenPaketleyici(max_seq_len=self.max_seq_len)
        paketlenmis = paketleyici.paketle(ornekler)

        # Token Packing Throughput Ölçümü
        t0 = time.perf_counter()
        toplam_paket_ornek = 0
        for i in range(min(iterasyon, len(paketlenmis))):
            p = paketlenmis[i]
            loss = model.egitim_adimi_paketlenmis(p)
            optimizer.zero_grad()
            toplam_paket_ornek += len(p.ornek_uzunluklari)

        t1 = time.perf_counter()
        packed_sure = t1 - t0
        packed_ornek_sn = toplam_paket_ornek / max(packed_sure, 1e-4)

        # Standart Padding Throughput Simülasyonu
        # Standartta her adım 1 örnek (batch=1) üzerinden simüle edilir
        t2 = time.perf_counter()
        for i in range(min(iterasyon, len(ornekler))):
            o = ornekler[i]
            inp = torch.tensor([o.prompt_ids + o.response_ids], dtype=torch.long, device=self.cihaz)
            lbl = torch.tensor([[-100]*len(o.prompt_ids) + o.response_ids], dtype=torch.long, device=self.cihaz)
            _, loss = model(inp, labels=lbl)
            if loss is not None:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        t3 = time.perf_counter()
        standart_sure = t3 - t2
        standart_ornek_sn = min(iterasyon, len(ornekler)) / max(standart_sure, 1e-4)

        hizlanma_kati = packed_ornek_sn / max(standart_ornek_sn, 1e-4)

        return {
            "Standart Paddingli SFT": {
                "ornek_saniye": round(standart_ornek_sn, 1),
                "sure_s": round(standart_sure, 3),
            },
            "Token Packed SFT (FFD)": {
                "ornek_saniye": round(packed_ornek_sn, 1),
                "sure_s": round(packed_sure, 3),
                "hizlanma_orani": f"{hizlanma_kati:.2f}x Daha Hızlı",
            },
        }
