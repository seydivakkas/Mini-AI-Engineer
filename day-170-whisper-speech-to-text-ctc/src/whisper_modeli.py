"""
Whisper Modeli ve CTC Konuşma Tanıma Modülü (Day 170 - FAZ 9).
1D Conv2 + Transformer Encoder + Causal Cross-Attention Decoder mimarisi.
"""

from typing import Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class WhisperModeli(nn.Module):
    """OpenAI Whisper Tarzı Çok Dilli ASR Encoder-Decoder Mimarisi."""

    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 256,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        vocab_size: int = 1000,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.d_model = d_model
        self.vocab_size = vocab_size

        # 1. 1D Conv2 Spektrogram Kök Katmanı (Stem)
        self.conv1 = nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)

        # 2. Transformer Audio Encoder
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=512, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        # 3. CTC Hizalama Başlığı
        self.ctc_head = nn.Linear(d_model, vocab_size)

        # 4. Metin Gömme ve Causal Decoder
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=4, dim_feedforward=512, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        # 5. Dil Modeli Çıkış Başlığı
        self.lm_head = nn.Linear(d_model, vocab_size)

    def audio_kodla(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: [B, 80, T] -> [B, T//2, d_model]"""
        x = F.gelu(self.conv1(mel))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)  # [B, T//2, d_model]
        enc_out = self.encoder(x)
        return enc_out

    def forward(
        self,
        mel: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        mel: [B, 80, T]
        text_tokens: [B, S]
        Döner: (lm_logits, ctc_logits)
        """
        enc_out = self.audio_kodla(mel)
        ctc_logits = self.ctc_head(enc_out)

        # Causal maskeleme
        S = text_tokens.shape[1]
        tgt_mask = torch.triu(torch.full((S, S), float("-inf"), device=text_tokens.device), diagonal=1)

        tgt = self.text_embedding(text_tokens)
        dec_out = self.decoder(tgt=tgt, memory=enc_out, tgt_mask=tgt_mask)
        lm_logits = self.lm_head(dec_out)

        return lm_logits, ctc_logits

    @classmethod
    def ornek_transkripsiyon_senaryolarini_getir(cls) -> Dict[str, Any]:
        """Zaman damgalı konuşma tanıma senaryoları."""
        return {
            "senaryolar": [
                {
                    "ses_id": "tr_konusma_01.wav",
                    "dil": "Türkçe (TR)",
                    "sure_saniye": 3.5,
                    "gercek_metin": "Yapay zeka modelleri çok dilli konuşmayı anlıyor.",
                    "tahmin_metin": "Yapay zeka modelleri çok dilli konuşmayı anlıyor.",
                    "zaman_damgali_transkripsiyon": [
                        {"kelime": "Yapay", "baslangic": "00:00.00", "bitis": "00:00.60"},
                        {"kelime": "zeka", "baslangic": "00:00.60", "bitis": "00:01.10"},
                        {"kelime": "modelleri", "baslangic": "00:01.10", "bitis": "00:01.80"},
                        {"kelime": "çok dilli", "baslangic": "00:01.80", "bitis": "00:02.40"},
                        {"kelime": "konuşmayı", "baslangic": "00:02.40", "bitis": "00:02.95"},
                        {"kelime": "anlıyor.", "baslangic": "00:02.95", "bitis": "00:03.50"},
                    ],
                    "wer": 0.0,
                    "cer": 0.0,
                },
                {
                    "ses_id": "en_speech_02.wav",
                    "dil": "İngilizce (EN)",
                    "sure_saniye": 2.8,
                    "gercek_metin": "Whisper architecture achieves robust speech recognition.",
                    "tahmin_metin": "Whisper architecture achieves robust speech recognition.",
                    "zaman_damgali_transkripsiyon": [
                        {"kelime": "Whisper", "baslangic": "00:00.00", "bitis": "00:00.50"},
                        {"kelime": "architecture", "baslangic": "00:00.50", "bitis": "00:01.20"},
                        {"kelime": "achieves", "baslangic": "00:01.20", "bitis": "00:01.70"},
                        {"kelime": "robust", "baslangic": "00:01.70", "bitis": "00:02.10"},
                        {"kelime": "speech", "baslangic": "00:02.10", "bitis": "00:02.45"},
                        {"kelime": "recognition.", "baslangic": "00:02.45", "bitis": "00:02.80"},
                    ],
                    "wer": 0.0,
                    "cer": 0.0,
                }
            ],
            "ortalama_wer": 0.0,
            "ortalama_cer": 0.0,
        }
