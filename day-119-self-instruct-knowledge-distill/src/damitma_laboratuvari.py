"""
Knowledge Distillation ve Model Kıyaslama Laboratuvarı (Day 119).
Öğretmen vs SFT Öğrenci vs KD Öğrenci eğitimi, doğruluk, çıkarım gecikmesi ve bellek tasarrufu.
"""

from typing import Dict, Any, List, Tuple
import time
import torch
import torch.optim as optim

from .ogretmen_ogrenci_modeller import ogretmen_model_uret, ogrenci_model_uret
from .damitma_kaybi import KnowledgeDistillationLoss
from .self_instruct_ureteci import SelfInstructUreteci


class DamitmaLaboratuvari:
    """Öğretmen ve Öğrenci modellerin damıtma eğitimini ve kıyaslamasını yöneten laboratuvar."""

    def __init__(
        self,
        vocab_size: int = 1000,
        seq_len: int = 32,
        sicaklik: float = 2.5,
        alpha: float = 0.3,
        lr: float = 1e-3,
        seed: int = 42,
    ):
        torch.manual_seed(seed)
        self.vocab_size = vocab_size
        self.seq_len = seq_len

        self.ogretmen = ogretmen_model_uret(vocab_size=vocab_size)
        self.ogrenci_sft = ogrenci_model_uret(vocab_size=vocab_size)
        self.ogrenci_kd = ogrenci_model_uret(vocab_size=vocab_size)

        self.kd_loss_fn = KnowledgeDistillationLoss(sicaklik=sicaklik, alpha=alpha)
        self.ce_loss_fn = torch.nn.CrossEntropyLoss()

        self.uretec = SelfInstructUreteci(vocab_size=vocab_size, max_seq_len=seq_len, seed=seed)
        self.lr = lr

    def egitim_ve_kiyaslama_kostur(
        self,
        adim_sayisi: int = 30,
        batch_size: int = 16,
    ) -> Dict[str, Any]:
        """SFT ve Knowledge Distillation eğitim döngülerini koşturur ve performans metriklerini toplar."""
        opt_sft = optim.AdamW(self.ogrenci_sft.parameters(), lr=self.lr)
        opt_kd = optim.AdamW(self.ogrenci_kd.parameters(), lr=self.lr)

        sft_kayiplar = []
        kd_kayiplar = []
        kl_kayiplar = []

        self.ogretmen.eval()

        for adim in range(adim_sayisi):
            x, y = self.uretec.sentetik_batch_uret(batch_size=batch_size)

            # 1. Öğretmen İleri Yayılım (Dondurulmuş / No Grad)
            with torch.no_grad():
                ogretmen_logits = self.ogretmen(x)

            # 2. Standart SFT Öğrenci Eğitimi (Yalnızca Hard CE)
            opt_sft.zero_grad()
            sft_logits = self.ogrenci_sft(x)
            l_sft = self.ce_loss_fn(sft_logits.view(-1, self.vocab_size), y.view(-1))
            l_sft.backward()
            opt_sft.step()
            sft_kayiplar.append(float(l_sft.item()))

            # 3. Knowledge Distillation (KD) Öğrenci Eğitimi (Hard CE + Soft KL)
            opt_kd.zero_grad()
            kd_logits = self.ogrenci_kd(x)
            l_kd, metrik = self.kd_loss_fn(kd_logits, ogretmen_logits, y)
            l_kd.backward()
            opt_kd.step()
            kd_kayiplar.append(metrik["toplam_kayip"])
            kl_kayiplar.append(metrik["kl_kaybi"])

        # 4. Çıkarım Hızı ve Gecikme Testi
        test_x, _ = self.uretec.sentetik_batch_uret(batch_size=32)

        # Öğretmen Gecikmesi
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(50):
                _ = self.ogretmen(test_x)
        t_ogretmen = (time.perf_counter() - t0) * 1000.0 / 50.0  # ms / batch

        # Öğrenci Gecikmesi
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(50):
                _ = self.ogrenci_kd(test_x)
        t_ogrenci = (time.perf_counter() - t0) * 1000.0 / 50.0  # ms / batch

        ogretmen_param = self.ogretmen.toplam_parametre()
        ogrenci_param = self.ogrenci_kd.toplam_parametre()
        hizlanma_orani = t_ogretmen / max(1e-5, t_ogrenci)
        parametre_tasarrufu = ((ogretmen_param - ogrenci_param) / ogretmen_param) * 100.0

        return {
            "sft_kayiplar": sft_kayiplar,
            "kd_kayiplar": kd_kayiplar,
            "kl_kayiplar": kl_kayiplar,
            "son_sft_kayip": sft_kayiplar[-1],
            "son_kd_kayip": kd_kayiplar[-1],
            "ogretmen_parametre": ogretmen_param,
            "ogrenci_parametre": ogrenci_param,
            "parametre_tasarrufu": parametre_tasarrufu,
            "ogretmen_gecikme_ms": t_ogretmen,
            "ogrenci_gecikme_ms": t_ogrenci,
            "hizlanma_orani": hizlanma_orani,
        }
