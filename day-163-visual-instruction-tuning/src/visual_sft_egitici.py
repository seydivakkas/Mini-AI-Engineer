"""
Görsel SFT Eğitim Motoru (Visual SFT Trainer) Modülü (Day 163 - FAZ 9).
LLaVA tarzı VLM üzerinde görsel komut ince ayar döngüsünü yürütür.
"""

from typing import Dict, Any, List
import torch
import torch.nn as nn
import torch.optim as optim

from .kayip_maskeleyici import VisualLossMaskeleyici


class VisualSFTEgitici:
    """Görsel SFT Eğitim ve Doğrulama Simülatörü."""

    @classmethod
    def egitim_dongusu_yurut(
        cls,
        vlm_model: nn.Module,
        adim_sayisi: int = 5,
        ogrenme_orani: float = 1e-3,
    ) -> Dict[str, Any]:
        """Görsel SFT adımlarını simüle eder ve kayıp eğrisini döner."""
        optimizer = optim.AdamW(vlm_model.parameters(), lr=ogrenme_orani, weight_decay=0.01)
        kayiplar = []

        torch.manual_seed(42)
        # Sabit mini-parti üzerinde gradient descent
        dummy_img = torch.randn(2, 3, 224, 224)
        prompt_ids = torch.randint(10, 500, (2, 12))
        response_ids = torch.randint(10, 500, (2, 20))

        input_text_ids, labels = VisualLossMaskeleyici.hedef_maskeli_etiket_olustur(
            visual_token_count=256,
            prompt_token_ids=prompt_ids,
            response_token_ids=response_ids,
        )

        for adim in range(1, adim_sayisi + 1):
            optimizer.zero_grad()
            logits = vlm_model(dummy_img, input_text_ids)
            loss = VisualLossMaskeleyici.maskeli_cross_entropy_kaybi_hesapla(logits, labels)

            loss.backward()
            optimizer.step()

            kayiplar.append(float(loss.item()))

        return {
            "baslangic_kaybi": round(kayiplar[0], 4),
            "bitis_kaybi": round(kayiplar[-1], 4),
            "kayip_dususu_yuzdesi": round((1.0 - kayiplar[-1] / kayiplar[0]) * 100.0, 2),
            "kayip_gecmisi": kayiplar,
        }
