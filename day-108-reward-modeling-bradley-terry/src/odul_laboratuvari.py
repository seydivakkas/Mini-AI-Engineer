"""
Bradley-Terry Ödül Modeli Eğitim ve Değerlendirme Laboratuvarı (Day 108).
Çiftli tercih veri seti üretimi, ödül eğitimi, marjin ayrışması ve Reward Hacking analizi.
"""

import time
import random
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn

from .bradley_terry_kaybi import BradleyTerryLoss, tercih_olasiligi, tercih_dogrulugu
from .odul_modeli import OdulModeli


class OdulLaboratuvari:
    """Reward Model eğitim ve tercih ayrışma laboratuvarı."""

    def __init__(
        self,
        vocab_size: int = 1000,
        dim: int = 256,
        cihaz: torch.device = torch.device("cpu"),
    ):
        self.vocab_size = vocab_size
        self.dim = dim
        self.cihaz = cihaz

    def sentetik_tercih_verisi_uret(
        self,
        cift_sayisi: int = 250,
        seq_len: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tercih edilen (chosen - yüksek kaliteli) ve reddedilen (rejected - düşük kaliteli)
        çiftli veri seti üretir.
        """
        torch.manual_seed(42)
        random.seed(42)

        chosen_list = []
        rejected_list = []

        p_max = max(self.vocab_size // 10, 2)
        c_min = p_max + 1
        c_max = max(self.vocab_size // 2, c_min + 1)
        r_min = c_max + 1
        r_max = self.vocab_size - 1

        for _ in range(cift_sayisi):
            # Prompt: 15 token
            prompt = [random.randint(1, p_max) for _ in range(min(15, seq_len // 2))]
            kalan = seq_len - len(prompt)
            # Chosen: Yüksek kaliteli token aralığı
            chosen_resp = [random.randint(c_min, c_max) for _ in range(kalan)]
            # Rejected: Düşük kaliteli token aralığı
            rejected_resp = [random.randint(r_min, r_max) for _ in range(kalan)]

            chosen_list.append(prompt + chosen_resp)
            rejected_list.append(prompt + rejected_resp)

        chosen_tensor = torch.tensor(chosen_list, dtype=torch.long, device=self.cihaz)
        rejected_tensor = torch.tensor(rejected_list, dtype=torch.long, device=self.cihaz)
        return chosen_tensor, rejected_tensor

    def odul_modeli_egit(
        self,
        model: OdulModeli,
        chosen: torch.Tensor,
        rejected: torch.Tensor,
        epok_sayisi: int = 20,
        batch_size: int = 32,
        lr: float = 1e-3,
        margin: float = 0.5,
    ) -> Dict[str, List[float]]:
        """Bradley-Terry kaybı ile ödül modelini eğitir ve gelişim metriklerini kaydeder."""
        model.to(self.cihaz).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        bt_kayip_fn = BradleyTerryLoss(margin=margin, reg_lambda=0.01)

        N = chosen.shape[0]
        kayiplar, dogruluklar, marjinler, r_w_ortalamalar, r_l_ortalamalar = [], [], [], [], []

        for epok in range(epok_sayisi):
            perm = torch.randperm(N, device=self.cihaz)
            epok_kayip = 0.0
            epok_dogruluk = 0.0
            epok_rw, epok_rl = 0.0, 0.0
            adim = 0

            for i in range(0, N, batch_size):
                idx = perm[i : i + batch_size]
                b_chosen = chosen[idx]
                b_rejected = rejected[idx]

                r_w, r_l = model.ciftli_odul_hesapla(b_chosen, b_rejected)
                loss, acc = bt_kayip_fn(r_w, r_l)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epok_kayip += loss.item()
                epok_dogruluk += acc.item()
                epok_rw += r_w.mean().item()
                epok_rl += r_l.mean().item()
                adim += 1

            kayiplar.append(round(epok_kayip / adim, 4))
            dogruluklar.append(round((epok_dogruluk / adim) * 100.0, 2))
            r_w_ort = epok_rw / adim
            r_l_ort = epok_rl / adim
            r_w_ortalamalar.append(round(r_w_ort, 3))
            r_l_ortalamalar.append(round(r_l_ort, 3))
            marjinler.append(round(r_w_ort - r_l_ort, 3))

        return {
            "kayiplar": kayiplar,
            "dogruluklar": dogruluklar,
            "marjinler": marjinler,
            "r_w_ort": r_w_ortalamalar,
            "r_l_ort": r_l_ortalamalar,
        }

    def reward_hacking_analizi(
        self,
        model: OdulModeli,
        ornek_sayisi: int = 50,
    ) -> Dict[str, float]:
        """
        Reward Hacking (Ödül İstismarı / Goodhart Yasası) testi:
        Modelin uzunluk veya tekrara sahte yüksek ödül verip vermediğini ölçer.
        """
        model.eval()
        with torch.no_grad():
            # 1. Normal Kaliteli Yanıt
            norm_inp = torch.tensor([[1]*15 + [200]*40], dtype=torch.long, device=self.cihaz)
            r_normal = float(model(norm_inp).item())

            # 2. Aşırı Uzun / Tekrarlayan Yanıt (Sahte Hacking)
            hack_inp = torch.tensor([[1]*15 + [200]*100], dtype=torch.long, device=self.cihaz)
            r_uzun = float(model(hack_inp).item())

            # 3. Anlamsız / Toksik Yanıt
            bad_inp = torch.tensor([[1]*15 + [800]*40], dtype=torch.long, device=self.cihaz)
            r_bad = float(model(bad_inp).item())

        return {
            "odul_kaliteli": round(r_normal, 3),
            "odul_uzun_tekrar": round(r_uzun, 3),
            "odul_kotu": round(r_bad, 3),
            "ayrisma_guvenilirligi": round(r_normal - r_bad, 3),
        }
