"""
Mekansal Dikkat Haritası Analizörü Modülü (Day 174 - FAZ 9).
Kelime bazlı dikkat yoğunluğu (Attention Energy) ve semantik odaklanma skorlarını hesaplar.
"""

from typing import Dict, Any, List
import torch
import numpy as np


class DikkatHaritasiAnalizoru:
    """Cross-Attention Haritası ve Prompt Kontrol Analizörü."""

    @classmethod
    def kelime_odak_skorlarini_hesapla(
        cls,
        attn_map: torch.Tensor,
        kelimeler: List[str],
        H: int = 16,
        W: int = 16,
    ) -> List[Dict[str, Any]]:
        """
        attn_map: [B=1, H*W=256, S_text]
        Her kelime için piksel odaklanma oranını ve tepe koordinatlarını hesaplar.
        """
        B, HW, S = attn_map.shape
        sonuclar = []

        map_np = attn_map[0].detach().cpu().numpy()  # [256, S]

        for i, kelime in enumerate(kelimeler):
            if i >= S:
                break
            w_attn = map_np[:, i]  # [256]
            grid_attn = w_attn.reshape(H, W)

            enerji_orani = float(grid_attn.sum() / (map_np.sum() + 1e-8))
            tepe_idx = np.unravel_index(np.argmax(grid_attn), (H, W))

            sonuclar.append({
                "kelime": kelime,
                "enerji_orani": round(enerji_orani, 3),
                "tepe_konum": (int(tepe_idx[0]), int(tepe_idx[1])),
                "odak_bolgesi": f"[{tepe_idx[0]}, {tepe_idx[1]}]",
            })

        return sonuclar

    @classmethod
    def ornek_cross_attention_raporu(cls) -> Dict[str, Any]:
        """Örnek istem ve çapraz dikkat analiz verileri."""
        return {
            "prompt": "A cute astronaut cat wearing a red helmet in galaxy",
            "kelimeler": ["cute", "astronaut", "cat", "wearing", "red", "helmet", "galaxy"],
            "kelime_skorlari": [
                {"kelime": "cat", "enerji": 0.28, "odak": "Merkez Bölge [8, 8]", "renk": "#e74a3b"},
                {"kelime": "helmet", "enerji": 0.24, "odak": "Üst Bölge [4, 8]", "renk": "#4e73df"},
                {"kelime": "astronaut", "enerji": 0.21, "odak": "Gövde Bölgesi [10, 8]", "renk": "#f6c23e"},
                {"kelime": "galaxy", "enerji": 0.18, "odak": "Arka Plan [2, 2]", "renk": "#1cc88a"},
                {"kelime": "cute", "enerji": 0.09, "odak": "Yüz Çevresi [6, 8]", "renk": "#36b9cc"},
            ],
            "ortalama_cross_attention_entropisi": 2.14,
            "metin_piksel_hizalama_dogrulugu": 0.96,
        }
