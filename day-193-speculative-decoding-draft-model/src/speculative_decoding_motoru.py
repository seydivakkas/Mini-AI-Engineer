"""
Spekülatif Çıkarım (Speculative Decoding) Motoru (Day 193 - FAZ 10).
Taslak Model (Draft Model) Önerisi, Paralel Doğrulama ve Rejection Sampling (Leviathan et al., 2023).
"""

from typing import List, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class KucukDraftModel(nn.Module):
    """
    Hızlı ve Küçük Taslak Model (Draft Model - M_q).
    Örn: Llama-3 1B veya 8B. Düşük gecikmeyle K token önerir.
    """

    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=128, batch_first=True)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        h = self.transformer_layer(x)
        logits = self.lm_head(h)
        return logits


class BuyukTargetModel(nn.Module):
    """
    Büyük ve Güçlü Hedef Model (Target Model - M_p).
    Örn: Llama-3 70B. K tokenı tek bir ileri geçişte (Parallel Verification) doğrular.
    """

    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=512, batch_first=True)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        h = self.transformer_layer(x)
        logits = self.lm_head(h)
        return logits


class RejectionSampler:
    """
    Leviathan et al. (2023) Matematiksel Olarak Özdeş Rejection Sampling Motoru.
    Hedef model dağılımından sapmayı sıfır (KL=0) tutarak kabul/red kararı verir.
    """

    @classmethod
    def kabul_olasiligi(cls, p_prob: float, q_prob: float) -> float:
        """Kabul olasılığı: alpha = min(1.0, p(x) / q(x))."""
        if q_prob <= 0.0:
            return 1.0
        return min(1.0, p_prob / q_prob)

    @classmethod
    def artik_dagilimdan_ornekle(cls, p_dist: torch.Tensor, q_dist: torch.Tensor) -> int:
        """
        Reddedilme durumunda artık dağılımdan örnekleme:
        P_resample(x) = max(0, p(x) - q(x)) / sum(max(0, p(x) - q(x)))
        """
        fark = torch.clamp(p_dist - q_dist, min=0.0)
        toplam = torch.sum(fark)
        if toplam.item() <= 1e-8:
            # Sayısal taşma koruması: Doğrudan p dağılımından örnekle
            probs = p_dist
        else:
            probs = fark / toplam
        return int(torch.multinomial(probs, num_samples=1).item())


class SpeculativeDecodingEngine:
    """
    Spekülatif Çıkarım Orkestrasyon Yürütücüsü.
    """

    def __init__(
        self,
        draft_model: KucukDraftModel,
        target_model: BuyukTargetModel,
        gamma: int = 4,  # Her döngüde spekülatif önerilecek token sayısı K
    ):
        self.draft_model = draft_model
        self.target_model = target_model
        self.gamma = gamma

    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 30,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Spekülatif döngü ile token üretimi.
        """
        mevcut_dizi = list(prompt_ids)
        hedef_uzunluk = len(prompt_ids) + max_new_tokens

        toplam_target_forward = 0
        toplam_taslak_onerilen = 0
        toplam_kabul_edilen = 0

        while len(mevcut_dizi) < hedef_uzunluk:
            # -------------------------------------------------------------
            # ADIM 1: Taslak Model (Draft) K Adım Spekülatif Öneri Üretir
            # -------------------------------------------------------------
            taslak_tokenlar: List[int] = []
            taslak_olasiliklar: List[torch.Tensor] = []

            gecici_dizi = list(mevcut_dizi)
            with torch.no_grad():
                for _ in range(self.gamma):
                    inp = torch.tensor([gecici_dizi], dtype=torch.long)
                    logits = self.draft_model(inp)[0, -1, :] / max(temperature, 1e-4)
                    probs = F.softmax(logits, dim=-1)
                    next_tok = int(torch.multinomial(probs, num_samples=1).item())

                    taslak_tokenlar.append(next_tok)
                    taslak_olasiliklar.append(probs)
                    gecici_dizi.append(next_tok)

            toplam_taslak_onerilen += len(taslak_tokenlar)

            # -------------------------------------------------------------
            # ADIM 2: Hedef Model (Target) Tek İleri Geçişte Paralel Doğrular
            # -------------------------------------------------------------
            with torch.no_grad():
                dogrulama_girdisi = torch.tensor([mevcut_dizi + taslak_tokenlar], dtype=torch.long)
                hedef_logits = self.target_model(dogrulama_girdisi)[0]  # [SeqLen, Vocab]
                toplam_target_forward += 1

            # Hedef modelin her pozisyon için olasılıkları
            baslangic_idx = len(mevcut_dizi) - 1
            hedef_olasiliklar = [
                F.softmax(hedef_logits[baslangic_idx + i] / max(temperature, 1e-4), dim=-1)
                for i in range(len(taslak_tokenlar) + 1)
            ]

            # -------------------------------------------------------------
            # ADIM 3: Rejection Sampling ile Kabul/Red Doğrulaması
            # -------------------------------------------------------------
            kabul_edildi_sayisi = 0
            for i, (tok, q_prob_dist) in enumerate(zip(taslak_tokenlar, taslak_olasiliklar)):
                p_prob_dist = hedef_olasiliklar[i]
                q_p = float(q_prob_dist[tok].item())
                p_p = float(p_prob_dist[tok].item())

                alpha = RejectionSampler.kabul_olasiligi(p_p, q_p)
                r = torch.rand(1).item()

                if r <= alpha:
                    # Kabul Edildi!
                    mevcut_dizi.append(tok)
                    kabul_edildi_sayisi += 1
                    toplam_kabul_edilen += 1
                    if len(mevcut_dizi) >= hedef_uzunluk:
                        break
                else:
                    # Reddedildi! Artık dağılımdan örnekle ve bu döngüyü bitir
                    duzeltilmis_tok = RejectionSampler.artik_dagilimdan_ornekle(p_prob_dist, q_prob_dist)
                    mevcut_dizi.append(duzeltilmis_tok)
                    break

            # Eğer K tokenın tamamı kabul edildiyse, bonus K+1. token doğrudan hedef modelden eklenir!
            if kabul_edildi_sayisi == len(taslak_tokenlar) and len(mevcut_dizi) < hedef_uzunluk:
                bonus_probs = hedef_olasiliklar[-1]
                bonus_tok = int(torch.multinomial(bonus_probs, num_samples=1).item())
                mevcut_dizi.append(bonus_tok)

        uretilen_toplam = len(mevcut_dizi) - len(prompt_ids)
        kabul_orani = toplam_kabul_edilen / max(toplam_taslak_onerilen, 1)
        hizlanma_orani = uretilen_toplam / max(toplam_target_forward, 1)

        return {
            "prompt_ids": prompt_ids,
            "uretilen_token_sayisi": uretilen_toplam,
            "target_forward_sayisi": toplam_target_forward,
            "taslak_onerilen_toplam": toplam_taslak_onerilen,
            "kabul_edilen_toplam": toplam_kabul_edilen,
            "kabul_orani": round(kabul_orani, 3),
            "hizlanma_faktoru": round(hizlanma_orani, 2),
            "sonuc_dizisi": mevcut_dizi,
        }
