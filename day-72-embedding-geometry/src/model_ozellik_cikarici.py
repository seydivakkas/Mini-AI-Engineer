"""
Model Özellik Çıkarıcı ve Temsil Uzayı Üreticisi
------------------------------------------------
Bu modül, derin görsel ağlardan yüksek boyutlu öznitelik vektörleri (embedding)
çıkaran mimariyi ve geometrik analizler için kontrollü temsil veri kümelerini sunar.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GorselTemsilAgi(nn.Module):
    """
    Görüntülerden düşük ve yüksek boyutlu temsil vektörleri çıkaran konvolüsyonel ağ.
    """
    def __init__(self, giris_kanali: int = 3, temsil_boyutu: int = 64, sinif_sayisi: int = 5):
        super().__init__()
        self.temsil_boyutu = temsil_boyutu
        
        # Konvolüsyonel Özellik Çıkarıcı Omurga
        self.omurga = nn.Sequential(
            nn.Conv2d(giris_kanali, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 16x16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 8x8
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)) # 128x1x1
        )
        
        # Temsil (Embedding) Projeksiyon Katmanı
        self.projeksiyon = nn.Sequential(
            nn.Linear(128, temsil_boyutu),
            nn.BatchNorm1d(temsil_boyutu)
        )
        
        # Sınıflandırma Kafası (Opsiyonel)
        self.siniflandirici = nn.Linear(temsil_boyutu, sinif_sayisi)

    def forward(self, x: torch.Tensor, normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        İleri besleme.
        Döndürür: (temsil_vektoru, sinif_logits)
        """
        b = x.size(0)
        ozellikler = self.omurga(x).view(b, -1)
        temsil = self.projeksiyon(ozellikler)
        
        if normalize:
            temsil = F.normalize(temsil, p=2, dim=1)
            
        logits = self.siniflandirici(temsil)
        return temsil, logits

    def temsil_cikar(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Yalnızca temsil (embedding) vektörünü döner."""
        temsil, _ = self.forward(x, normalize=normalize)
        return temsil


class TemsilVeriUreteci:
    """
    Geometrik analizler ve boyut indirgeme algoritmaları için
    farklı topolojik karakteristiklere sahip temsil kümeleri üretir.
    """
    @staticmethod
    def uret_kontrollu_temsiller(
        ornek_sayisi: int = 600,
        boyut: int = 64,
        sinif_sayisi: int = 5,
        gurultu: float = 0.25,
        tohum: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, str]]:
        """
        Belirgin kümelere sahip, gerçekçi çok sınıflı temsil uzayı üretir.
        """
        np.random.seed(tohum)
        ornek_basina = ornek_sayisi // sinif_sayisi
        
        temsiller = []
        etiketler = []
        
        # Her sınıf için rastgele bir ortogonal yön/merkez oluştur
        merkezler = np.random.randn(sinif_sayisi, boyut)
        merkezler = merkezler / np.linalg.norm(merkezler, axis=1, keepdims=True) * 3.0
        
        for c in range(sinif_sayisi):
            sinif_merkezi = merkezler[c]
            # Küme içi saçılım
            noktalar = sinif_merkezi + np.random.randn(ornek_basina, boyut) * gurultu
            # Non-lineer manifold bükülmesi
            noktalar[:, :4] += np.sin(noktalar[:, 4:8]) * 0.5
            
            # L2 Normalize et (Birim Hiperküre Üzerine İzdüşür)
            noktalar = noktalar / np.linalg.norm(noktalar, axis=1, keepdims=True)
            
            temsiller.append(noktalar)
            etiketler.append(np.full(ornek_basina, c, dtype=np.int64))
            
        X = np.vstack(temsiller)
        y = np.concatenate(etiketler)
        
        meta = {
            "tip": "Yapılandırılmış Çok Kümeli Temsil",
            "ornek_sayisi": str(X.shape[0]),
            "boyut": str(X.shape[1]),
            "sinif_sayisi": str(sinif_sayisi)
        }
        return X, y, meta

    @staticmethod
    def uret_boyutsal_cokmus_temsiller(
        ornek_sayisi: int = 600,
        boyut: int = 64,
        efektif_boyut: int = 2,
        tohum: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Boyutsal Çöküşe (Dimensional Collapse / Anisotropy) uğramış temsil uzayı üretir.
        Varyansın %95'ten fazlası sadece 1-2 eksende toplanmıştır (Dar Koni Dağılımı).
        """
        np.random.seed(tohum)
        ham = np.random.randn(ornek_sayisi, boyut)
        
        # İlk efektif_boyut eksen dışındakileri aşırı bastır
        agirliklar = np.ones(boyut) * 0.01
        agirliklar[:efektif_boyut] = 5.0
        
        cokmus = ham * agirliklar
        cokmus = cokmus / np.linalg.norm(cokmus, axis=1, keepdims=True)
        etiketler = np.random.randint(0, 5, size=ornek_sayisi)
        return cokmus, etiketler
