"""
FastAPI Çıkarım Servisi Yönetici Motoru (Day 98).
Modelin bellekte tutulması, ön/son işleme, asenkron çıkarım ve metrik takibini yönetir.
"""

import io
import time
import base64
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from .konfigurasyon import MiniViTConfig
from .model import MiniViTForImageClassification
from .semalar import (
    TahminOgesi,
    TahminYaniti,
    TopluTahminYaniti,
    SaglikYaniti,
    ModelMetaveriYaniti,
)


class ServisYoneticisi:
    """
    MiniViT FastAPI Çıkarım Servisi için Singleton Yönetici Sınıfı.
    Modeli belleğe yükler, görsel ön işlemesini yapar ve çıkarım taleplerini karşılar.
    """
    _instance: Optional["ServisYoneticisi"] = None

    def __init__(self, config: Optional[MiniViTConfig] = None):
        self.config = config or MiniViTConfig()
        self.cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MiniViTForImageClassification(self.config).to(self.cihaz)
        self.model.eval()

        self.baslangic_zamani = time.time()
        self.toplam_istek_sayisi = 0
        self.gecikme_kayitlari: List[float] = []

        # Normalizasyon sabitleri (CIFAR-10)
        self.mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
        self.std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

        # Isınma
        self._isinma()

    @classmethod
    def get_instance(cls) -> "ServisYoneticisi":
        if cls._instance is None:
            cls._instance = ServisYoneticisi()
        return cls._instance

    def _isinma(self):
        """Modeli başlatıp GPU/CPU önbelleğini ısıtır."""
        dummy = torch.randn(1, 3, self.config.goruntu_boyutu, self.config.goruntu_boyutu, device=self.cihaz)
        with torch.no_grad():
            for _ in range(3):
                _ = self.model(dummy)

    def goruntu_on_isle(self, image: Image.Image) -> torch.Tensor:
        """PIL görüntüsünü modele uygun tensöre dönüştürür."""
        img_rgb = image.convert("RGB").resize(
            (self.config.goruntu_boyutu, self.config.goruntu_boyutu),
            Image.Resampling.BILINEAR,
        )
        arr = np.array(img_rgb, dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(self.cihaz)
        return tensor

    def base64_coz(self, base64_str: str) -> Image.Image:
        """Base64 dizgisini çözüp PIL Image nesnesine dönüştürür."""
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        img_bytes = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(img_bytes))

    def tahmin_et(self, gorsel: Image.Image, top_k: int = 5) -> TahminYaniti:
        """Tekli görsel üzerinde çıkarım yapar."""
        t0 = time.perf_counter()
        tensor = self.goruntu_on_isle(gorsel)

        with torch.no_grad():
            logits = self.model(tensor).logits
            olasiliklar = F.softmax(logits, dim=-1)[0]

        top_k = min(top_k, self.config.sinif_sayisi)
        probs, indices = torch.topk(olasiliklar, k=top_k)

        t1 = time.perf_counter()
        gecikme_ms = (t1 - t0) * 1000.0

        # Metrik güncelle
        self.toplam_istek_sayisi += 1
        self.gecikme_kayitlari.append(gecikme_ms)

        tahminler: List[TahminOgesi] = []
        for p, idx in zip(probs.cpu().numpy(), indices.cpu().numpy()):
            idx_int = int(idx)
            label = self.config.id2label.get(idx_int, f"LABEL_{idx_int}")
            tahminler.append(TahminOgesi(
                sinif_adi=label,
                sinif_id=idx_int,
                olasilik=float(p),
            ))

        return TahminYaniti(
            durum="basarili",
            en_iyi_tahmin=tahminler[0],
            top_k_tahminler=tahminler,
            gecikme_ms=round(gecikme_ms, 2),
            model_surumu="v1.0",
        )

    def toplu_tahmin_et(self, gorseller: List[Image.Image], top_k: int = 5) -> TopluTahminYaniti:
        """Çoklu görseller üzerinde toplu çıkarım yapar."""
        t0 = time.perf_counter()
        sonuclar = [self.tahmin_et(g, top_k=top_k) for g in gorseller]
        t1 = time.perf_counter()
        toplam_gecikme = (t1 - t0) * 1000.0

        return TopluTahminYaniti(
            durum="basarili",
            toplam_goruntu=len(gorseller),
            sonuclar=sonuclar,
            toplam_gecikme_ms=round(toplam_gecikme, 2),
        )

    def saglik_raporu(self) -> SaglikYaniti:
        """Liveness / Readiness probeları için sağlık durumunu döner."""
        uptime = time.time() - self.baslangic_zamani
        return SaglikYaniti(
            status="HEALTHY",
            model_loaded=True,
            cihaz=self.cihaz.type,
            toplam_istek=self.toplam_istek_sayisi,
            calisma_suresi_sn=round(uptime, 2),
        )

    def metaveri_raporu(self) -> ModelMetaveriYaniti:
        """Model mimari ve etiket metaverisini döner."""
        toplam_param = sum(p.numel() for p in self.model.parameters())
        return ModelMetaveriYaniti(
            model_adi="MiniViT-v1.0",
            parametre_sayisi=toplam_param,
            goruntu_boyutu=self.config.goruntu_boyutu,
            sinif_sayisi=self.config.sinif_sayisi,
            id2label=self.config.id2label,
        )

    def metrik_raporu(self) -> Dict[str, Any]:
        """Prometheus uyumlu performans metrikleri."""
        gecikmeler = self.gecikme_kayitlari if self.gecikme_kayitlari else [0.0]
        return {
            "toplam_istek_sayisi": self.toplam_istek_sayisi,
            "p50_gecikme_ms": round(float(np.percentile(gecikmeler, 50)), 2),
            "p90_gecikme_ms": round(float(np.percentile(gecikmeler, 90)), 2),
            "p99_gecikme_ms": round(float(np.percentile(gecikmeler, 99)), 2),
            "ortalama_gecikme_ms": round(float(np.mean(gecikmeler)), 2),
        }
