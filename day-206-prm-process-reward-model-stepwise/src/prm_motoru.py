"""
Step-Level PRM (Process Reward Model) Motoru (Day 206 - FAZ 11).
Her Düşünce Adımını Ayrı Ayrı Skorlayan ve Test-Zamanı Arama Ağacı Budayan Doğrulayıcı (Lightman et al. PRM800K).
"""

from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PRMStepClassifier(nn.Module):
    """
    Adım Düzeyinde Süreç Ödül Modeli (Process Reward Model - PRM).
    Her düşünce adımının doğruluğunu (p in [0.0, 1.0]) skorlar.
    """

    def __init__(self, vocab_size: int = 128, embed_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=128, batch_first=True
        )
        self.score_head = nn.Linear(embed_dim, 1)

    def forward(self, step_tokens: torch.Tensor) -> torch.Tensor:
        """
        step_tokens: [batch_size, seq_len]
        Dönüş: [batch_size] Adım doğruluk olasılığı (0.0 - 1.0)
        """
        h = self.embedding(step_tokens)
        encoded = self.encoder(h)
        # Adım sonu havuzlama (Last Token Pooling)
        last_hidden = encoded[:, -1, :]
        logits = self.score_head(last_hidden).squeeze(-1)
        return torch.sigmoid(logits)


class MathReasoningTrajectory:
    """Çok Adımlı Matematiksel Akıl Yürütme Yörüngesi."""

    def __init__(self, problem_sorusu: str, adimlar: List[str], nihai_cevap: str):
        self.soru = problem_sorusu
        self.adimlar = adimlar
        self.nihai_cevap = nihai_cevap
        self.adim_skorlari: List[float] = []

    def prm_skorla(self, prm_model: PRMStepClassifier) -> List[float]:
        """Yörüngedeki her adımı PRM modelinden geçirerek adım skorlarını çıkarır."""
        self.adim_skorlari = []
        for i, adim in enumerate(self.adimlar):
            # Sahte token tensörü (adım metnine göre simülasyon)
            dummy_tokens = torch.randint(0, 128, (1, 10))
            with torch.no_grad():
                skor = prm_model(dummy_tokens).item()

            # Adım içeriğine göre kalibre edilmiş simülasyon puanı
            if "Hata" in adim or "yanlış" in adim.lower():
                skor = min(0.20, skor)
            else:
                skor = max(0.75, skor)

            self.adim_skorlari.append(skor)
        return self.adim_skorlari

    @property
    def carpim_skoru(self) -> float:
        """Tüm adım olasılıklarının çarpımı: prod(p_k)."""
        if not self.adim_skorlari:
            return 0.0
        return float(np.prod(self.adim_skorlari))

    @property
    def minimum_skor(self) -> float:
        """En zayıf adımın skoru (Zincir en zayıf halkası kadar güçlüdür): min(p_k)."""
        if not self.adim_skorlari:
            return 0.0
        return float(min(self.adim_skorlari))


class PRMTreeSearchEngine:
    """
    PRM Tabanlı Test-Zamanı Arama ve Erken Dal Budama (Early Branch Pruning) Motoru.
    """

    @classmethod
    def aday_yol_budama_simulasyonu(
        cls,
        prm_model: PRMStepClassifier,
        esik_deger: float = 0.40,
    ) -> Dict[str, Any]:
        """
        4 Farklı Aday Çözüm Yolunu Test Eder.
        Hatalı adım görüldüğü anda dalı budar (Pruning) ve GPU işlem israfını önler.
        """
        ornek_yollar = [
            MathReasoningTrajectory(
                "2x + 6 = 20",
                ["1. Adım: Her iki taraftan 6 çıkarıldı: 2x = 14", "2. Adım: Her iki taraf 2'ye bölündü: x = 7"],
                "7",
            ),
            MathReasoningTrajectory(
                "2x + 6 = 20",
                ["1. Adım: Hatalı işlem yapıldı: 2x = 26", "2. Adım: x = 13 bulundu"],
                "13",
            ),
            MathReasoningTrajectory(
                "2x + 6 = 20",
                ["1. Adım: 2x = 14", "2. Adım: Hata: 14/2 = 8 olarak hesaplandı"],
                "8",
            ),
            MathReasoningTrajectory(
                "2x + 6 = 20",
                ["1. Adım: Denklem düzenlendi: 2x = 14", "2. Adım: Doğrulama yapıldı: x = 7"],
                "7",
            ),
        ]

        sonuclar = []
        toplam_budanan_token = 0
        toplam_hesaplanan_token = 0

        for idx, yorunge in enumerate(ornek_yollar):
            skorlar = yorunge.prm_skorla(prm_model)
            budandi = False
            budandigi_adim = -1

            for adim_idx, s in enumerate(skorlar):
                toplam_hesaplanan_token += 25
                if s < esik_deger:
                    budandi = True
                    budandigi_adim = adim_idx + 1
                    # Kalan adımları üretmeyerek token tasarrufu sağla
                    tasarruf = (len(skorlar) - adim_idx - 1) * 25
                    toplam_budanan_token += tasarruf
                    break

            sonuclar.append({
                "yol_idx": idx + 1,
                "adim_sayisi": len(yorunge.adimlar),
                "adim_skorlari": skorlar,
                "carpim_skoru": yorunge.carpim_skoru,
                "minimum_skor": yorunge.minimum_skor,
                "budandi": budandi,
                "budandigi_adim": budandigi_adim,
                "nihai_cevap": yorunge.nihai_cevap,
            })

        return {
            "yollar": sonuclar,
            "toplam_budanan_token": toplam_budanan_token,
            "toplam_hesaplanan_token": toplam_hesaplanan_token,
            "hesaplama_tasarrufu_yuzde": (
                (toplam_budanan_token / (toplam_budanan_token + toplam_hesaplanan_token + 1e-8)) * 100.0
            ),
        }
