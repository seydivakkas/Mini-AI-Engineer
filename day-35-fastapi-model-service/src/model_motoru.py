"""
Yapay Zeka Çıkarım Motoru (Multimodal AI Inference Engine).
"""

from typing import List, Dict, Any, Tuple
import asyncio
import io
import time
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniMetinSiniflandirici(nn.Module):
    """3 Sınıflı Metin Konu Sınıflandırıcısı ve Embedding Çıkarıcı."""

    def __init__(self, vocab_size: int = 5000, embed_dim: int = 64, num_classes: int = 3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding(input_ids)
        mask_exp = mask.unsqueeze(-1).expand_as(emb)
        pooled = (emb * mask_exp).sum(dim=1) / torch.clamp(mask_exp.sum(dim=1), min=1e-9)
        norm_emb = F.normalize(pooled, p=2, dim=1)
        logits = self.fc(norm_emb)
        return logits, norm_emb


class AIModelMotoru:
    """FastAPI servisi için thread-safe yapay zeka çıkarım motoru."""

    SINIFLAR = ["Görüntü İşleme & CV", "Doğal Dil İşleme & NLP", "MLOps & Sistem Mimarisi"]

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.metin_modeli = MiniMetinSiniflandirici().to(self.device)
        self.metin_modeli.eval()

        # Dahili RAG doküman havuzu
        self.rag_dokumanlari = [
            {"id": "KB-01", "konu": "YOLO", "metin": "YOLO mimarisi tek aşamalı gerçek zamanlı nesne tespiti yapar."},
            {"id": "KB-02", "konu": "RAG", "metin": "RAG mimarisi harici bağlamı enjekte ederek halüsinasyonu önler."},
            {"id": "KB-03", "konu": "FastAPI", "metin": "FastAPI asenkron uç noktalar ile yüksek verimli REST API sağlar."}
        ]

    def _sayisallastir(self, metin: str, max_len: int = 32) -> Tuple[List[int], List[int]]:
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

    def metin_tahmin_et_senkron(self, metin: str, embedding_iste: bool = False) -> Dict[str, Any]:
        """Senkron CPU yoğun metin sınıflandırması."""
        t0 = time.perf_counter()
        ids, mask = self._sayisallastir(metin)

        t_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        t_mask = torch.tensor([mask], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits, embs = self.metin_modeli(t_ids, t_mask)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

        en_iyi_idx = int(np.argmax(probs))
        tum_olasiliklar = {self.SINIFLAR[i]: float(probs[i]) for i in range(len(self.SINIFLAR))}
        gecikme_ms = (time.perf_counter() - t0) * 1000.0

        vektor = embs[0].cpu().numpy().tolist() if embedding_iste else None

        return {
            "metin": metin,
            "tahmin_edilen_etiket": self.SINIFLAR[en_iyi_idx],
            "olasilik": float(probs[en_iyi_idx]),
            "tum_olasiliklar": tum_olasiliklar,
            "vektor_embedding": vektor,
            "gecikme_ms": gecikme_ms
        }

    async def metin_tahmin_et(self, metin: str, embedding_iste: bool = False) -> Dict[str, Any]:
        """Event loop'u bloklamamak için worker thread havuzunda koşturur."""
        return await asyncio.to_thread(self.metin_tahmin_et_senkron, metin, embedding_iste)

    def gorsel_analiz_et_senkron(self, dosya_adi: str, icerik_baytlari: bytes) -> Dict[str, Any]:
        """Senkron görsel piksel analizi ve nesne tespiti simülasyonu."""
        t0 = time.perf_counter()
        boyut_bayt = len(icerik_baytlari)

        # Görsel baytlarından renk ve özellik çıkarımı
        dizi = np.frombuffer(icerik_baytlari, dtype=np.uint8)
        if len(dizi) >= 3:
            baskin_renk = [int(dizi[0]), int(dizi[1 % len(dizi)]), int(dizi[2 % len(dizi)])]
        else:
            baskin_renk = [128, 128, 128]

        # Tespit edilen nesneler
        tespitler = [
            {"etiket": "Nesne_A", "guven": 0.94, "kutu": [25, 40, 180, 220]},
            {"etiket": "Nesne_B", "guven": 0.88, "kutu": [210, 150, 300, 310]}
        ]
        gecikme_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "dosya_adi": dosya_adi,
            "boyut_bayt": boyut_bayt,
            "tespit_edilen_nesneler": tespitler,
            "en_baskin_renk": baskin_renk,
            "gecikme_ms": gecikme_ms
        }

    async def gorsel_analiz_et(self, dosya_adi: str, icerik_baytlari: bytes) -> Dict[str, Any]:
        return await asyncio.to_thread(self.gorsel_analiz_et_senkron, dosya_adi, icerik_baytlari)

    async def rag_sorgula(self, soru: str, top_k: int = 2) -> Dict[str, Any]:
        """RAG doküman sorgulama ve kaynak atıflı yanıt."""
        t0 = time.perf_counter()
        soru_alt = soru.lower()

        eslesenler = []
        for doc in self.rag_dokumanlari:
            if any(k in doc["metin"].lower() or k in doc["konu"].lower() for k in soru_alt.split()):
                eslesenler.append(doc)

        if not eslesenler:
            eslesenler = self.rag_dokumanlari[:1]

        secilenler = eslesenler[:top_k]
        kaynaklar = [d["id"] for d in secilenler]
        yanit = f"Bilgi tabanına göre: {' '.join([d['metin'] for d in secilenler])} [Kaynak: {', '.join(kaynaklar)}]"
        gecikme_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "soru": soru,
            "yanit": yanit,
            "kaynaklar": kaynaklar,
            "guven_skoru": 0.92,
            "durum": "BASARILI",
            "gecikme_ms": gecikme_ms
        }
