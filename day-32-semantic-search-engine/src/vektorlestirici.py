"""
Cümle Vektörleştirici ve Bi-Encoder Yoğun Temsil Üreticisi (Sentence Vectorizer).
"""

from typing import List, Dict, Union
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniTransformerEncoder(nn.Module):
    """Hafif Bi-Encoder Transformer Kodlayıcı Modülü."""

    def __init__(self, vocab_size: int = 5000, embed_dim: int = 128, num_heads: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embeddings = nn.Parameter(torch.randn(1, 128, embed_dim) * 0.02)

        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        b, seq_len = input_ids.shape
        x = self.token_embeddings(input_ids) + self.pos_embeddings[:, :seq_len, :]

        # Key padding mask: True olan yerler göz ardı edilir
        key_padding_mask = (attention_mask == 0)
        attn_out, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class CumleVektorlestirici:
    """
    Metinleri anlamsal yoğun gömmelere (Dense Embeddings) dönüştüren Bi-Encoder motoru.
    """

    def __init__(self, embed_dim: int = 128, max_seq_len: int = 64, device: str = "cpu"):
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.device = torch.device(device)

        self.model = MiniTransformerEncoder(vocab_size=5000, embed_dim=embed_dim).to(self.device)
        self.model.eval()

    def _metni_sayisallastir(self, metin: str) -> List[int]:
        """Metni temizleyip kelime hash tabanlı token ID'lerine dönüştürür."""
        temiz = re.sub(r"[^\w\s]", " ", metin.lower()).strip()
        kelimeler = temiz.split()
        if not kelimeler:
            return [1]  # UNK token

        token_ids = []
        for k in kelimeler[:self.max_seq_len]:
            # Deterministik hash ile 2..4999 aralığında ID
            h = (abs(hash(k)) % 4998) + 2
            token_ids.append(h)
        return token_ids

    def vektorlestir(self, metinler: Union[str, List[str]]) -> np.ndarray:
        """
        Metin veya metin listesini (N, D) boyutunda L2-normalize yoğun vektörlere dönüştürür.
        """
        tekil = isinstance(metinler, str)
        liste = [metinler] if tekil else metinler
        if not liste:
            return np.empty((0, self.embed_dim), dtype=np.float32)

        batch_ids = []
        batch_masks = []

        for m in liste:
            ids = self._metni_sayisallastir(m)
            mask = [1] * len(ids)

            # Padding (0)
            if len(ids) < self.max_seq_len:
                pad_len = self.max_seq_len - len(ids)
                ids += [0] * pad_len
                mask += [0] * pad_len
            else:
                ids = ids[:self.max_seq_len]
                mask = mask[:self.max_seq_len]

            batch_ids.append(ids)
            batch_masks.append(mask)

        input_tensor = torch.tensor(batch_ids, dtype=torch.long, device=self.device)
        mask_tensor = torch.tensor(batch_masks, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            token_embeddings = self.model(input_tensor, mask_tensor)  # (B, L, D)

            # Mean Pooling (Attention Mask Ağırlıklı Ortalama Havuzlama)
            input_mask_expanded = mask_tensor.unsqueeze(-1).expand(token_embeddings.size())
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask

            # L2 Normalizasyonu (Birim Vektör)
            normalized_embeddings = F.normalize(mean_pooled, p=2, dim=1)

        cikis = normalized_embeddings.cpu().numpy().astype(np.float32)
        return cikis[0] if tekil else cikis
