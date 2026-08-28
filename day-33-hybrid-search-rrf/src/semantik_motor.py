"""
Semantik Vektör Arama Motoru (Dense Semantic Vector Engine).
"""

from typing import List, Dict, Any, Union, Tuple
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniBiEncoder(nn.Module):
    """Hafif Bi-Encoder Transformer."""

    def __init__(self, vocab_size: int = 5000, embed_dim: int = 128):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embeddings = nn.Parameter(torch.randn(1, 128, embed_dim) * 0.02)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, seq_len = input_ids.shape
        x = self.token_embeddings(input_ids) + self.pos_embeddings[:, :seq_len, :]
        key_padding_mask = (mask == 0)
        attn_out, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm(x + attn_out)

        # Mean pooling
        expanded_mask = mask.unsqueeze(-1).expand(x.size())
        sum_emb = torch.sum(x * expanded_mask, dim=1)
        sum_mask = torch.clamp(expanded_mask.sum(dim=1), min=1e-9)
        pooled = sum_emb / sum_mask
        return F.normalize(pooled, p=2, dim=1)


class SemantikVektorMotoru:
    """Yoğun vektör çıkarımı ve kosinüs benzerliği indeksi."""

    def __init__(self, embed_dim: int = 128, device: str = "cpu"):
        self.embed_dim = embed_dim
        self.device = torch.device(device)
        self.encoder = MiniBiEncoder(vocab_size=5000, embed_dim=embed_dim).to(self.device)
        self.encoder.eval()

        self.vektorler = np.empty((0, embed_dim), dtype=np.float32)
        self.doc_ids: List[str] = []
        self.dokumanlar: Dict[str, Dict[str, Any]] = {}

    def _sayisallastir(self, metin: str, max_len: int = 64) -> Tuple[List[int], List[int]]:
        temiz = re.sub(r"[^\w\s]", " ", metin.lower()).strip()
        kelimeler = temiz.split()
        if not kelimeler:
            return [1] + [0] * (max_len - 1), [1] + [0] * (max_len - 1)

        ids = [((abs(hash(k)) % 4998) + 2) for k in kelimeler[:max_len]]
        mask = [1] * len(ids)
        if len(ids) < max_len:
            pad = max_len - len(ids)
            ids += [0] * pad
            mask += [0] * pad
        return ids, mask

    def vektorlestir(self, metinler: Union[str, List[str]]) -> np.ndarray:
        tekil = isinstance(metinler, str)
        liste = [metinler] if tekil else metinler

        batch_ids = []
        batch_masks = []
        for m in liste:
            ids, mask = self._sayisallastir(m)
            batch_ids.append(ids)
            batch_masks.append(mask)

        t_ids = torch.tensor(batch_ids, dtype=torch.long, device=self.device)
        t_masks = torch.tensor(batch_masks, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            embs = self.encoder(t_ids, t_masks).cpu().numpy().astype(np.float32)

        return embs[0] if tekil else embs

    def dokuman_ekle(self, doc_id: str, baslik: str, icerik: str, metaveri: Dict[str, Any] = None):
        metin = f"{baslik}. {icerik}"
        v = self.vektorlestir(metin).reshape(1, self.embed_dim)

        if len(self.vektorler) == 0:
            self.vektorler = v
        else:
            self.vektorler = np.vstack([self.vektorler, v])

        self.doc_ids.append(doc_id)
        self.dokumanlar[doc_id] = {
            "baslik": baslik,
            "icerik": icerik,
            "metaveri": metaveri or {}
        }

    def ara(self, sorgu_metni: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if len(self.doc_ids) == 0:
            return []

        q_vec = self.vektorlestir(sorgu_metni).reshape(1, self.embed_dim)
        benzerlikler = np.dot(self.vektorler, q_vec.T).flatten()
        sirali = np.argsort(benzerlikler)[::-1]

        sonuclar = []
        for idx in sirali[:top_k]:
            doc_id = self.doc_ids[idx]
            sonuclar.append({
                "doc_id": doc_id,
                "skor": float(benzerlikler[idx]),
                "baslik": self.dokumanlar[doc_id]["baslik"],
                "icerik": self.dokumanlar[doc_id]["icerik"],
                "metaveri": self.dokumanlar[doc_id]["metaveri"]
            })
        return sonuclar
