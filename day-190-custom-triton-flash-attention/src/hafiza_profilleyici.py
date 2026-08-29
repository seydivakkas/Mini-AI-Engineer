"""
FlashAttention-2 Bellek ve HBM IO Profilleyici Modülü (Day 190 - FAZ 10).
Standart O(N^2) Dikkat Matrisi vs Parçalı (Tiled) O(N) FlashAttention VRAM Analitiği.
"""

from typing import Dict, Any, List


class FlashAttentionBellekProfilleyici:
    """
    FlashAttention-2 Bellek ve Donanım IO Profilleyicisi.
    1k'dan 128k'ya kadar uzanan bağlam uzunluklarında (Context Length) VRAM tüketimini karşılaştırır.
    """

    @classmethod
    def baglam_uzunlugu_vram_analizi(
        cls,
        batch_size: int = 1,
        num_heads: int = 32,
        head_dim: int = 128,
        seq_len: int = 8192,
        eleman_bayt: int = 2,  # FP16 / BF16
    ) -> Dict[str, Any]:
        """Tek bir dikkat katmanı için bağlam uzunluğuna bağlı VRAM tüketimi."""
        # 1. Standart Attention (O(N^2) Matris Saklama):
        # S (Skor) ve P (Softmax) matrisleri: B * H * N * N * 2 bayt
        standart_nxn_bayt = 2.0 * batch_size * num_heads * (seq_len ** 2) * eleman_bayt
        standart_vram_mb = standart_nxn_bayt / (1024.0 * 1024.0)

        # 2. FlashAttention-2 (O(N) Parçalı Bloklar):
        # Yalnızca Q, K, V, O ve L (LogSumExp) saklanır. NxN matrisi belleğe asla yazılmaz!
        flash_o_n_bayt = (batch_size * num_heads * seq_len * head_dim * eleman_bayt) + (batch_size * num_heads * seq_len * 4)
        flash_vram_mb = flash_o_n_bayt / (1024.0 * 1024.0)

        tasarruf_orani = standart_vram_mb / max(flash_vram_mb, 1e-4)

        return {
            "batch_size": batch_size,
            "num_heads": num_heads,
            "seq_len": seq_len,
            "standart_vram_mb": round(standart_vram_mb, 2),
            "flash_vram_mb": round(flash_vram_mb, 2),
            "tasarruf_orani": round(tasarruf_orani, 1),
            "tasarruf_kat_sayisi": f"{tasarruf_orani:.1f}x",
        }

    @classmethod
    def baglam_tarama_raporu(cls) -> List[Dict[str, Any]]:
        """1k, 4k, 16k, 64k ve 128k bağlam uzunluklarında karşılaştırmalı VRAM raporu."""
        uzunluklar = [1024, 4096, 16384, 65536, 131072]
        rapor = []

        for n in uzunluklar:
            p = cls.baglam_uzunlugu_vram_analizi(batch_size=1, num_heads=32, head_dim=128, seq_len=n)
            std_gb = p["standart_vram_mb"] / 1024.0
            fa_gb = p["flash_vram_mb"] / 1024.0
            oom_mu = std_gb > 80.0  # 80GB H100 GPU taşması

            rapor.append({
                "context_length": n,
                "context_etiket": f"{n // 1024}k" if n >= 1024 else str(n),
                "standart_vram_gb": round(std_gb, 2),
                "flash_vram_gb": round(fa_gb, 4),
                "tasarruf_faktoru": p["tasarruf_kat_sayisi"],
                "standart_oom_durumu": "OOM (Bellek Taştı!)" if oom_mu else "Sığıyor",
            })

        return rapor
