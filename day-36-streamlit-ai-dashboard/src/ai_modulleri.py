"""
Dashboard AI Çıkarım ve Analiz Motoru (Dashboard AI Engine).
"""

from typing import Dict, List, Any, Tuple
import time
import re
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniAIEngine(nn.Module):
    """Hafif Sinir Ağı Modeli."""

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


class DashboardAIEngine:
    """Streamlit kontrol paneli için çoklu modalite yapay zeka motoru."""

    SINIFLAR = ["Görüntü İşleme & CV", "Doğal Dil İşleme & NLP", "MLOps & Sistem Mimarisi"]

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = MiniAIEngine().to(self.device)
        self.model.eval()

        self.rag_dokumanlari = [
            {"id": "KB-01", "baslik": "YOLO Nesne Tespiti", "metin": "YOLO tek aşamalı gerçek zamanlı nesne tespiti yapar."},
            {"id": "KB-02", "baslik": "RAG Mimarisi", "metin": "RAG mimarisi harici bağlamı enjekte ederek halüsinasyonu önler."},
            {"id": "KB-03", "baslik": "FastAPI & Streamlit", "metin": "FastAPI arka uç ve Streamlit ön yüz ile interaktif AI panelleri kurulur."}
        ]

        self.istek_sayisi: int = 0
        self.gecikme_gecmisi: List[float] = []

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

    def metin_analiz_et(self, metin: str) -> Dict[str, Any]:
        """Metin sınıflandırma, güven skorları ve embedding çıkarımı."""
        t0 = time.perf_counter()
        ids, mask = self._sayisallastir(metin)

        t_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        t_mask = torch.tensor([mask], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits, embs = self.model(t_ids, t_mask)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

        en_iyi_idx = int(np.argmax(probs))
        olasiliklar = {self.SINIFLAR[i]: float(probs[i]) for i in range(len(self.SINIFLAR))}
        gecikme_ms = (time.perf_counter() - t0) * 1000.0

        self.istek_sayisi += 1
        self.gecikme_gecmisi.append(gecikme_ms)

        return {
            "metin": metin,
            "tahmin_sinifi": self.SINIFLAR[en_iyi_idx],
            "guven": float(probs[en_iyi_idx]),
            "olasiliklar": olasiliklar,
            "embedding": embs[0].cpu().numpy().tolist(),
            "gecikme_ms": gecikme_ms
        }

    def gorsel_analiz_et(self, gorsel: Image.Image, guven_esigi: float = 0.5) -> Dict[str, Any]:
        """Görsel üzerinde tespit simülasyonu ve renk ayrıştırma."""
        t0 = time.perf_counter()
        w, h = gorsel.size
        img_np = np.array(gorsel.convert("RGB"))

        # Baskın RGB rengi
        ortalama_renk = [int(img_np[:, :, c].mean()) for c in range(3)]

        # Simüle edilen tespitler
        tum_tespitler = [
            {"etiket": "Dokuma_Kusuru_A", "guven": 0.92, "kutu": [int(w*0.1), int(h*0.15), int(w*0.45), int(h*0.6)]},
            {"etiket": "Iplik_Hatasi_B", "guven": 0.78, "kutu": [int(w*0.6), int(h*0.5), int(w*0.9), int(h*0.85)]},
            {"etiket": "Leke_C", "guven": 0.42, "kutu": [int(w*0.4), int(h*0.7), int(w*0.55), int(h*0.9)]}
        ]

        filtrelenmis_tespitler = [t for t in tum_tespitler if t["guven"] >= guven_esigi]
        gecikme_ms = (time.perf_counter() - t0) * 1000.0

        self.istek_sayisi += 1
        self.gecikme_gecmisi.append(gecikme_ms)

        return {
            "genislik": w,
            "yukseklik": h,
            "baskin_renk": ortalama_renk,
            "tespitler": filtrelenmis_tespitler,
            "gecikme_ms": gecikme_ms
        }

    def rag_soru_sor(self, soru: str, top_k: int = 2) -> Dict[str, Any]:
        """RAG doküman sorgulama ve kaynak atıflı yanıt."""
        t0 = time.perf_counter()
        soru_kucuk = soru.lower()

        eslesenler = []
        for doc in self.rag_dokumanlari:
            if any(k in doc["metin"].lower() or k in doc["baslik"].lower() for k in soru_kucuk.split()):
                eslesenler.append(doc)

        if not eslesenler:
            eslesenler = self.rag_dokumanlari[:1]

        secilenler = eslesenler[:top_k]
        kaynaklar = [d["id"] for d in secilenler]
        yanit = f"Bilgi tabanına göre: {' '.join([d['metin'] for d in secilenler])} [Kaynak: {', '.join(kaynaklar)}]"
        gecikme_ms = (time.perf_counter() - t0) * 1000.0

        self.istek_sayisi += 1
        self.gecikme_gecmisi.append(gecikme_ms)

        return {
            "soru": soru,
            "yanit": yanit,
            "kaynaklar": kaynaklar,
            "guven": 0.94,
            "gecikme_ms": gecikme_ms
        }
