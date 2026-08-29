"""
Uçtan Uca Speech-to-Speech LLM (DuaLLM / Moshi Mimarisi) Modülü (Day 171 - FAZ 9).
Tek bir Transformer omurgası ile eş zamanlı Metin ve 8 Kademeli RVQ Ses Tokenı Üretimi.
"""

from typing import Tuple, Dict, Any, List
import torch
import torch.nn as nn
from .cift_akisli_token_birlestirici import CiftAkisliTokenBirlestirici


class SpeechToSpeechLLM(nn.Module):
    """Moshi ve GPT-4o Tarzı Çift Başlıklı Speech-to-Speech Transformer."""

    def __init__(
        self,
        text_vocab_size: int = 1000,
        audio_codebook_size: int = 1024,
        num_quantizers: int = 8,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.d_model = d_model

        # 1. Çift Akışlı Gömme Katmanı
        self.combiner = CiftAkisliTokenBirlestirici(
            text_vocab_size=text_vocab_size,
            audio_codebook_size=audio_codebook_size,
            num_quantizers=num_quantizers,
            d_model=d_model,
        )

        # 2. Causal Transformer Omurgası
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=512, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # 3. Metin Çıkış Başlığı (Text LM Head)
        self.text_head = nn.Linear(d_model, text_vocab_size)

        # 4. 8 Kademeli Ses Kod Defteri Başlıkları (Audio RVQ Heads)
        self.audio_heads = nn.ModuleList([
            nn.Linear(d_model, audio_codebook_size)
            for _ in range(num_quantizers)
        ])

    def forward(
        self,
        audio_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        audio_tokens: [B, Num_Q=8, T]
        text_tokens: [B, T] (Hizalanmış metin tokenları)
        Döner: (text_logits, audio_logits_list)
        """
        # Gömme katmanlarını topla: [B, T, d_model]
        audio_embed = self.combiner.ses_tokenlarini_gom(audio_tokens)
        text_embed = self.combiner.metin_tokenlarini_gom(text_tokens)
        x = audio_embed + text_embed

        # Causal maskeleme
        T = x.shape[1]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)

        # Transformer ile bağlamı öğren: [B, T, d_model]
        h = self.transformer(x, mask=mask)

        # Başlık 1: Metin tahmini
        text_logits = self.text_head(h)  # [B, T, text_vocab]

        # Başlık 2: 8 Kademeli RVQ ses tahmini
        # [Num_Q, B, T, Codebook_Size] -> [B, Num_Q, T, Codebook_Size]
        audio_logits = torch.stack([head(h) for head in self.audio_heads], dim=1)

        return text_logits, audio_logits

    @classmethod
    def ornek_diyalog_senaryolarini_getir(cls) -> Dict[str, Any]:
        """Canlı sesli sohbet senaryoları ve gecikme verileri."""
        return {
            "senaryolar": [
                {
                    "diyalog_id": "sesli_sohbet_01",
                    "kullanici_sesi": "Bugün hava nasıl olacak?",
                    "asistan_yaniti_ses": "Bugün İstanbul'da hava güneşli ve 24 derece.",
                    "geleneksel_gecikme_ms": 1450,  # ASR (400ms) + LLM (650ms) + TTS (400ms)
                    "duallm_gecikme_ms": 160,       # Uçtan uca doğrudan ses tokenı üretimi
                    "akustik_uyum_skoru": 0.98,
                    "dogruluk_skoru": 1.0,
                },
                {
                    "diyalog_id": "sesli_sohbet_02",
                    "kullanici_sesi": "Bana kuantum fiziğini tek bir cümleyle özetler misin?",
                    "asistan_yaniti_ses": "Kuantum fiziği, atom altı parçacıkların olasılıksal davranışlarını inceler.",
                    "geleneksel_gecikme_ms": 1620,
                    "duallm_gecikme_ms": 185,
                    "akustik_uyum_skoru": 0.97,
                    "dogruluk_skoru": 1.0,
                }
            ],
            "ortalama_geleneksel_gecikme_ms": 1535.0,
            "ortalama_duallm_gecikme_ms": 172.5,
            "gecikme_iyilesmesi_kat": 8.9,
        }
