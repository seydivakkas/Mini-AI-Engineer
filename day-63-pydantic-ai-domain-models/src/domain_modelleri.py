"""
Pydantic v2 Tip Güvenli Domain Modelleri ve Girdi/Çıktı Sözleşmeleri.
"""

from typing import List, Optional, Dict, Any
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ConfigDict
)
import numpy as np


class GorselMetadatasi(BaseModel):
    """Görsel giriş metaverisi için doğrulanmış tip sözleşmesi."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    genislik: int = Field(gt=0, le=7680, description="Piksel cinsinden genişlik (maks 8K)")
    yukseklik: int = Field(gt=0, le=4320, description="Piksel cinsinden yükseklik (maks 8K)")
    kanal_sayisi: int = Field(default=3, ge=1, le=4, description="Kanal sayısı (1: Grayscale, 3: RGB, 4: RGBA)")
    format: str = Field(pattern="^(JPEG|PNG|WEBP|TIFF)$", description="Desteklenen görsel formatı")
    dosya_boyutu_kb: float = Field(gt=0.0, le=50000.0, description="Dosya boyutu (maks 50MB)")

    @property
    def toplam_piksel(self) -> int:
        return self.genislik * self.yukseklik


class BoundingBoxModeli(BaseModel):
    """Normalize edilmiş ([0.0, 1.0]) koordinatlarla sınır kutusu (Bounding Box) sözleşmesi."""
    model_config = ConfigDict(extra="forbid")

    x_min: float = Field(ge=0.0, le=1.0, description="Sol üst köşe x")
    y_min: float = Field(ge=0.0, le=1.0, description="Sol üst köşe y")
    x_max: float = Field(ge=0.0, le=1.0, description="Sağ alt köşe x")
    y_max: float = Field(ge=0.0, le=1.0, description="Sağ alt köşe y")

    @model_validator(mode="after")
    def koordinat_siralamasini_dogrula(self) -> "BoundingBoxModeli":
        """Geometrik tutarlılık: x_min < x_max ve y_min < y_max olmalıdır."""
        if self.x_min >= self.x_max:
            raise ValueError(f"Geçersiz x koordinatları: x_min ({self.x_min}) < x_max ({self.x_max}) olmalıdır!")
        if self.y_min >= self.y_max:
            raise ValueError(f"Geçersiz y koordinatları: y_min ({self.y_min}) < y_max ({self.y_max}) olmalıdır!")
        return self

    def alan(self) -> float:
        """Kutunun normalize alanını hesaplar."""
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def iou(self, diger: "BoundingBoxModeli") -> float:
        """İki sınır kutusu arasındaki Intersection over Union (IoU) oranını hesaplar."""
        kesisim_x1 = max(self.x_min, diger.x_min)
        kesisim_y1 = max(self.y_min, diger.y_min)
        kesisim_x2 = min(self.x_max, diger.x_max)
        kesisim_y2 = min(self.y_max, diger.y_max)

        kesisim_genislik = max(0.0, kesisim_x2 - kesisim_x1)
        kesisim_yukseklik = max(0.0, kesisim_y2 - kesisim_y1)
        kesisim_alani = kesisim_genislik * kesisim_yukseklik

        birlesim_alani = self.alan() + diger.alan() - kesisim_alani
        if birlesim_alani <= 0.0:
            return 0.0
        return float(kesisim_alani / birlesim_alani)


class NesneTespitiSonucu(BaseModel):
    """Tekil nesne tespiti tahmin çıktısı sözleşmesi."""
    model_config = ConfigDict(extra="forbid")

    sinif_adi: str = Field(min_length=1, max_length=64, description="Tespit edilen nesne sınıfı")
    guven_skoru: float = Field(ge=0.0, le=1.0, description="Olasılık / Güven skoru [0.0, 1.0]")
    kutu: BoundingBoxModeli = Field(description="Tespit sınır kutusu")


class VektorEmbeddingSozlesmesi(BaseModel):
    """Birim hiperküre üzerinde L2-normalize özellik vektörü sözleşmesi."""
    model_config = ConfigDict(extra="forbid")

    vektor: List[float] = Field(description="Embedding değerleri listesi")
    beklenen_boyut: int = Field(default=512, gt=0, description="Beklenen vektör boyutu (örn: 512, 768)")

    @field_validator("vektor")
    @classmethod
    def vektor_boyutu_ve_normunu_dogrula(cls, v: List[float]) -> List[float]:
        """Vektör boyutunu ve L2 normunun birim uzunlukta (||e||=1.0) olduğunu doğrular."""
        if len(v) == 0:
            raise ValueError("Vektör boş olamaz!")
        
        arr = np.array(v, dtype=np.float32)
        norm = float(np.linalg.norm(arr))

        # L2 norm toleransı: [0.95, 1.05]
        if abs(norm - 1.0) > 0.05:
            raise ValueError(f"Vektör L2-normalize değil! Hesaplanmış norm: {norm:.4f} (Beklenen: 1.0 ± 0.05)")
        return v


class InferenceIstekSozlesmesi(BaseModel):
    """AI Model Çıkarım API İstek Sözleşmesi."""
    model_config = ConfigDict(extra="forbid")

    istek_id: str = Field(min_length=8, description="Benzersiz UUID veya Trace ID")
    model_adi: str = Field(min_length=2, description="Çağrılacak modelin adı")
    gorsel_meta: GorselMetadatasi = Field(description="Giriş görseli üstverisi")
    nms_esigi: float = Field(default=0.45, ge=0.0, le=1.0, description="Non-Maximum Suppression eşiği")
    guven_esigi: float = Field(default=0.50, ge=0.0, le=1.0, description="Minimum güven skoru filtresi")


class InferenceYanitSozlesmesi(BaseModel):
    """AI Model Çıkarım API Yanıt Sözleşmesi."""
    model_config = ConfigDict(extra="forbid")

    istek_id: str = Field(description="İstek kimliği ile eşleşen Trace ID")
    model_adi: str = Field(description="Çıkarım yapan model")
    tespitler: List[NesneTespitiSonucu] = Field(default_factory=list, description="Tespit edilen nesneler listesi")
    embedding: Optional[VektorEmbeddingSozlesmesi] = Field(default=None, description="Opsiyonel semantik embedding")
    gecikme_ms: float = Field(ge=0.0, description="Uçtan uca model çıkarım süresi (ms)")
    basarili: bool = Field(default=True, description="Çıkarım işlem durumu")
