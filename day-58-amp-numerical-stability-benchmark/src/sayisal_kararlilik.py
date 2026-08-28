"""
Sayısal Kararlılık ve Kayan Nokta Formatı Analizörü (FP32, FP16, BF16 & Underflow/Overflow).
"""

from typing import Dict, Any, Tuple
import numpy as np
import torch


class SayisalKararlilikAnalizoru:
    """Kayan nokta (Floating Point) formatlarının dinamik aralıklarını ve underflow/overflow risklerini inceler."""

    @staticmethod
    def format_ozelliklerini_getir() -> Dict[str, Dict[str, Any]]:
        """FP32, FP16 ve BF16 formatlarının bit yapısını ve matematiksel sınırlarını döndürür."""
        return {
            "FP32 (Single)": {
                "toplam_bit": 32,
                "isaret_biti": 1,
                "us_biti (exponent)": 8,
                "mantis_biti (fraction)": 23,
                "min_pozitif_normal": 1.175494e-38,
                "maks_sonlu": 3.402823e+38,
                "dinamik_aralik_onluk": "~10^±38",
                "grad_scaler_gerekir_mi": False
            },
            "FP16 (Half)": {
                "toplam_bit": 16,
                "isaret_biti": 1,
                "us_biti (exponent)": 5,
                "mantis_biti (fraction)": 10,
                "min_pozitif_normal": 6.103516e-05,
                "maks_sonlu": 65504.0,
                "dinamik_aralik_onluk": "~10^±5",
                "grad_scaler_gerekir_mi": True
            },
            "BF16 (Brain Float)": {
                "toplam_bit": 16,
                "isaret_biti": 1,
                "us_biti (exponent)": 8,
                "mantis_biti (fraction)": 7,
                "min_pozitif_normal": 1.175494e-38,
                "maks_sonlu": 3.389531e+38,
                "dinamik_aralik_onluk": "~10^±38",
                "grad_scaler_gerekir_mi": False
            }
        }

    @staticmethod
    def gradyan_kaybi_simulasyonu(
        ornek_sayisi: int = 100_000,
        olcek_faktoru: float = 65536.0
    ) -> Dict[str, Dict[str, float]]:
        """Küçük gradyanların FP32, Ham FP16, Ölçeklenmiş FP16 ve BF16'daki sıfıra yuvarlanma (underflow) analizini yapar."""
        torch.manual_seed(42)

        # Tipik derin katman gradyan dağılımı: 10^-8 ile 10^-2 arasında log-normal
        log_gradyanlar = torch.randn(ornek_sayisi) * 2.5 - 11.0  # e^-11 ~ 1.6e-5
        gercek_gradyanlar_fp32 = torch.exp(log_gradyanlar)

        # 1. Ham FP16'ya doğrudan dönüşüm (Ölçekleme yok)
        ham_fp16 = gercek_gradyanlar_fp32.to(torch.float16)
        fp16_underflow_orani = float((ham_fp16 == 0.0).sum().item() / ornek_sayisi * 100.0)
        fp16_overflow_orani = float((torch.isinf(ham_fp16) | torch.isnan(ham_fp16)).sum().item() / ornek_sayisi * 100.0)

        # 2. GradScaler ile Ölçeklenmiş FP16 (Loss Scaling)
        olcekli_fp16 = (gercek_gradyanlar_fp32 * olcek_faktoru).to(torch.float16)
        olcekli_fp16_geri_fp32 = olcekli_fp16.to(torch.float32) / olcek_faktoru
        olcekli_underflow_orani = float((olcekli_fp16_geri_fp32 == 0.0).sum().item() / ornek_sayisi * 100.0)

        # 3. BF16'ya dönüşüm
        bf16_tensör = gercek_gradyanlar_fp32.to(torch.bfloat16)
        bf16_underflow_orani = float((bf16_tensör == 0.0).sum().item() / ornek_sayisi * 100.0)
        bf16_overflow_orani = float((torch.isinf(bf16_tensör) | torch.isnan(bf16_tensör)).sum().item() / ornek_sayisi * 100.0)

        # Truncation error (Kırpma Hata Ortalama Rölatif Farkı)
        hata_fp16 = torch.abs(gercek_gradyanlar_fp32 - ham_fp16.to(torch.float32)) / (gercek_gradyanlar_fp32 + 1e-12)
        hata_bf16 = torch.abs(gercek_gradyanlar_fp32 - bf16_tensör.to(torch.float32)) / (gercek_gradyanlar_fp32 + 1e-12)

        return {
            "Ham FP16 (Scalersız)": {
                "underflow_orani": fp16_underflow_orani,
                "overflow_orani": fp16_overflow_orani,
                "ort_relatif_hata": float(hata_fp16.mean().item())
            },
            "AMP-FP16 (GradScaler)": {
                "underflow_orani": olcekli_underflow_orani,
                "overflow_orani": 0.0,
                "ort_relatif_hata": float((torch.abs(gercek_gradyanlar_fp32 - olcekli_fp16_geri_fp32) / (gercek_gradyanlar_fp32 + 1e-12)).mean().item())
            },
            "BF16 (Bfloat16)": {
                "underflow_orani": bf16_underflow_orani,
                "overflow_orani": bf16_overflow_orani,
                "ort_relatif_hata": float(hata_bf16.mean().item())
            }
        }
