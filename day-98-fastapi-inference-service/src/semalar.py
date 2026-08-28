"""
FastAPI Pydantic v2 Veri Sözleşmeleri ve Doğrulama Şemaları (Day 98).
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class TahminOgesi(BaseModel):
    """Tekil sınıflandırma tahmini ve olasılık skoru."""
    sinif_adi: str = Field(..., description="Sınıf etiketi (CIFAR-10)")
    sinif_id: int = Field(..., description="Sınıf indeksi (0-9)")
    olasilik: float = Field(..., ge=0.0, le=1.0, description="Tahmin olasılığı (Softmax)")


class TahminYaniti(BaseModel):
    """Tekli görüntü sınıflandırma API yanıtı."""
    durum: str = Field("basarili", description="İşlem durumu")
    en_iyi_tahmin: TahminOgesi = Field(..., description="En yüksek olasılığa sahip sınıf")
    top_k_tahminler: List[TahminOgesi] = Field(..., description="Top-K sınıf sıralaması")
    gecikme_ms: float = Field(..., description="Uçtan uca çıkarım gecikmesi (milisaniye)")
    model_surumu: str = Field("v1.0", description="Model sürüm bilgisi")


class Base64Istegi(BaseModel):
    """Base64 kodlu görüntü sınıflandırma istek gövdesi."""
    base64_goruntu: str = Field(..., description="Data URI veya salt Base64 formatında görüntü dizgisi")
    top_k: int = Field(5, ge=1, le=10, description="Döndürülecek en iyi tahmin sayısı")


class TopluTahminYaniti(BaseModel):
    """Çoklu görüntü sınıflandırma API yanıtı."""
    durum: str = Field("basarili", description="İşlem durumu")
    toplam_goruntu: int = Field(..., description="İşlenen toplam görüntü sayısı")
    sonuclar: List[TahminYaniti] = Field(..., description="Görüntü bazlı tahmin sonuçları")
    toplam_gecikme_ms: float = Field(..., description="Toplam işlem süresi (milisaniye)")


class SaglikYaniti(BaseModel):
    """Kubernetes Liveness & Readiness Sağlık Kontrolü Yanıtı."""
    status: str = Field("HEALTHY", description="Servis sağlık durumu (HEALTHY/DEGRADED/UNHEALTHY)")
    model_loaded: bool = Field(True, description="Modelin bellekte hazır olup olmadığı")
    cihaz: str = Field(..., description="Modelin çalıştığı donanım cihazı (cpu/cuda)")
    toplam_istek: int = Field(..., description="Ayağa kalkıştan bu yana karşılanan toplam istek")
    calisma_suresi_sn: float = Field(..., description="Servis çalışma süresi (saniye)")


class ModelMetaveriYaniti(BaseModel):
    """Model mimari ve etiket metaveri yanıtı."""
    model_adi: str = Field("MiniViT-v1.0", description="Model adı")
    parametre_sayisi: int = Field(..., description="Toplam eğitilebilir parametre sayısı")
    goruntu_boyutu: int = Field(32, description="Girdi çözünürlüğü (32x32)")
    sinif_sayisi: int = Field(10, description="Toplam sınıf sayısı")
    id2label: Dict[int, str] = Field(..., description="ID - Etiket sözlüğü")
