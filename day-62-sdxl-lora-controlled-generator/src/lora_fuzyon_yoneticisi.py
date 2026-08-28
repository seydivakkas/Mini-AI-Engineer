"""
Çoklu LoRA Adaptör Füzyonu, Dinamik Ağırlıklandırma ve Bellek/Verim Yöneticisi.
"""

from typing import Dict, Any, List, Tuple
import time
import torch
import numpy as np
from .sdxl_lora_motoru import SDXLLoRAMotoru, LatentDenoisingSampler


class LoRAFuzyonYoneticisi:
    """Çoklu LoRA adaptörlerini birleştiren, dinamik ölçekleme ve parametre verimliliği analizi yapan yönetici."""

    @classmethod
    def parametre_verimlilik_analizi(cls, model: SDXLLoRAMotoru) -> Dict[str, Any]:
        """Taban model parametreleri ile LoRA adaptör parametrelerini karşılaştırır."""
        taban_param = sum(p.numel() for n, p in model.named_parameters() if "lora" not in n)
        lora_param = sum(p.numel() for n, p in model.named_parameters() if "lora" in n)
        toplam_param = taban_param + lora_param

        return {
            "taban_parametre_sayisi": taban_param,
            "lora_parametre_sayisi": lora_param,
            "toplam_parametre_sayisi": toplam_param,
            "lora_oran_yuzde": (lora_param / max(taban_param, 1)) * 100.0,
            "tasarruf_orani_yuzde": (1.0 - (lora_param / max(taban_param, 1))) * 100.0
        }

    @classmethod
    def calistir_fuzyon_deneyi(
        cls,
        model: SDXLLoRAMotoru,
        skala_degerleri: List[float] = [0.0, 0.4, 0.8, 1.2],
        cfg_degerleri: List[float] = [3.0, 7.5, 12.0]
    ) -> Dict[str, Any]:
        """Farklı LoRA ölçekleri (lambda) ve CFG değerleri için çıktı latent analizi yapar."""
        d_model = model.d_model
        d_text = model.d_text

        # Sentetik metin koşulları
        kosul_metin = torch.randn(1, 16, d_text)
        kosulsuz_metin = torch.zeros(1, 16, d_text)

        # Eğitilmiş stil adaptörü ağırlıklarını simüle et
        if "stil_lora" in model.adaptorler:
            model.simule_et_egitilmis_lora("stil_lora", std=0.04)

        # 1. Taban Model Çıktısı (LoRA Scale = 0.0)
        model.adaptor_agirligi_ayarla("stil_lora", 0.0)
        sampler = LatentDenoisingSampler(adim_sayisi=15, cfg_skalasi=7.5)
        z_taban, _ = sampler.ornekle_latent(model, kosul_metin, kosulsuz_metin)

        skala_sonuclari = {}
        for skala in skala_degerleri:
            model.adaptor_agirligi_ayarla("stil_lora", skala)
            start = time.perf_counter()
            z_out, enerjiler = sampler.ornekle_latent(model, kosul_metin, kosulsuz_metin)
            gecikme_ms = (time.perf_counter() - start) * 1000.0

            # Taban ile LoRA çıktısı arasındaki L2 mesafesi ve Kosinüs Benzerliği
            delta_norm = float(torch.norm(z_out - z_taban).item())
            cos_sim = float(torch.cosine_similarity(z_out.flatten(), z_taban.flatten(), dim=0).item())

            skala_sonuclari[f"LoRA_Skala_{skala}"] = {
                "skala": skala,
                "delta_l2_norm": delta_norm,
                "kosinus_benzerlik": cos_sim,
                "gecikme_ms": gecikme_ms,
                "adim_enerjileri": enerjiler
            }

        # 2. CFG Hassasiyet Analizi
        cfg_sonuclari = {}
        model.adaptor_agirligi_ayarla("stil_lora", 0.8)
        for cfg in cfg_degerleri:
            sampler_cfg = LatentDenoisingSampler(adim_sayisi=15, cfg_skalasi=cfg)
            z_cfg, enerjiler = sampler_cfg.ornekle_latent(model, kosul_metin, kosulsuz_metin)
            cfg_sonuclari[f"CFG_{cfg}"] = {
                "cfg": cfg,
                "latent_norm": float(torch.norm(z_cfg).item()),
                "adim_enerjileri": enerjiler
            }

        return {
            "parametre_verimliligi": cls.parametre_verimlilik_analizi(model),
            "skala_analizi": skala_sonuclari,
            "cfg_analizi": cfg_sonuclari
        }
