"""
RMSNorm & Residual Bellek Trafik Profilleyici Modülü (Day 188 - FAZ 10).
Standart PyTorch vs Fused Triton HBM Bant Genişliği ve Gecikme Analizi.
"""

from typing import Dict, Any, List


class RMSNormBellekProfilleyici:
    """
    Fused RMSNorm ve Residual Ekleme Bellek Profilleyicisi.
    80 Katmanlı Llama-3 70B modeli ölçeğinde HBM tasarrufunu hesaplar.
    """

    @classmethod
    def bellek_ve_gecis_analizi(
        cls,
        batch_size: int = 4,
        seq_len: int = 4096,
        hidden_dim: int = 8192,
        eleman_bayt: int = 2,  # FP16 / BF16
    ) -> Dict[str, Any]:
        """Tek bir Transformer katmanı için HBM bellek geçiş ve tasarruf analizi."""
        toplam_token = batch_size * seq_len
        tek_tensor_mb = (toplam_token * hidden_dim * eleman_bayt) / (1024.0 * 1024.0)

        # PyTorch Unfused (Ayrı Adımlar):
        # 1. X + Residual (2 Okuma + 1 Yazma = 3)
        # 2. X^2 (1 Okuma + 1 Yazma = 2)
        # 3. Mean (1 Okuma + 1 Yazma = 2)
        # 4. Rsqrt (1 Okuma + 1 Yazma = 2)
        # 5. Scale & Output (3 Okuma + 1 Yazma = 4)
        # Toplam: 13 Geçiş
        pytorch_gecis_sayisi = 13
        pytorch_hbm_mb = pytorch_gecis_sayisi * tek_tensor_mb
        pytorch_ara_bellek_mb = 4.0 * tek_tensor_mb

        # Fused Triton (Tek Geçişli):
        # 1. Okuma: X, Residual, Weight (3 Okuma)
        # 2. Yazma: Y, X_res (2 Yazma)
        # Toplam: 5 Geçiş
        triton_gecis_sayisi = 5
        triton_hbm_mb = triton_gecis_sayisi * tek_tensor_mb
        triton_ara_bellek_mb = 0.0

        tasarruf_orani = pytorch_hbm_mb / triton_hbm_mb
        hbm_kazanc_yuzde = ((pytorch_hbm_mb - triton_hbm_mb) / pytorch_hbm_mb) * 100.0

        return {
            "toplam_token": toplam_token,
            "hidden_dim": hidden_dim,
            "tek_tensor_mb": round(tek_tensor_mb, 2),
            "pytorch_hbm_mb": round(pytorch_hbm_mb, 2),
            "pytorch_ara_bellek_mb": round(pytorch_ara_bellek_mb, 2),
            "triton_hbm_mb": round(triton_hbm_mb, 2),
            "triton_ara_bellek_mb": round(triton_ara_bellek_mb, 2),
            "tasarruf_orani": round(tasarruf_orani, 2),
            "hbm_kazanc_yuzde": round(hbm_kazanc_yuzde, 1),
            "hizlanma_faktoru": f"{tasarruf_orani:.1f}x",
        }

    @classmethod
    def model_olcegi_tasarruf_raporu(cls) -> List[Dict[str, Any]]:
        """Llama-3-8B, Llama-3-70B ve Gemma-2-27B modelleri için tam model tasarrufu."""
        modeller = [
            {"ad": "Llama-3-8B", "katman": 32, "dim": 4096},
            {"ad": "Gemma-2-27B", "katman": 46, "dim": 4608},
            {"ad": "Llama-3-70B", "katman": 80, "dim": 8192},
        ]

        rapor = []
        for m in modeller:
            profil = cls.bellek_ve_gecis_analizi(batch_size=4, seq_len=4096, hidden_dim=m["dim"])
            katman_sayisi = m["katman"]
            # Her katmanda 2 RMSNorm (1 Attention öncesi, 1 MLP öncesi)
            toplam_rmsnorm = katman_sayisi * 2
            toplam_pytorch_gb = (profil["pytorch_hbm_mb"] * toplam_rmsnorm) / 1024.0
            toplam_triton_gb = (profil["triton_hbm_mb"] * toplam_rmsnorm) / 1024.0
            tasarruf_gb = toplam_pytorch_gb - toplam_triton_gb

            rapor.append({
                "model_adi": m["ad"],
                "katman_sayisi": katman_sayisi,
                "toplam_rmsnorm_sayisi": toplam_rmsnorm,
                "pytorch_hbm_gb": round(toplam_pytorch_gb, 2),
                "triton_hbm_gb": round(toplam_triton_gb, 2),
                "tasarruf_gb": round(tasarruf_gb, 2),
                "hizlanma": profil["hizlanma_faktoru"],
            })

        return rapor
