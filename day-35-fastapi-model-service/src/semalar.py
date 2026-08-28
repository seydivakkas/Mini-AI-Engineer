"""
Pydantic v2 Veri Doğrulama ve İstek/Yanıt Şemaları.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator


class SaglikYaniti(BaseModel):
    """Sistem sağlık ve hazırlık (Readiness/Liveness) durumu."""
    durum: str = Field(default="aktif", description="Servis çalışma durumu")
    versiyon: str = Field(default="1.0.0", description="API versiyonu")
    cihaz: str = Field(default="cpu", description="Modelin çalıştığı donanım (CPU/GPU)")
    yuklu_modeller: List[str] = Field(default_factory=list, description="Bellekte hazır bulunan modeller")


class MetinTahminIstegi(BaseModel):
    """Metin sınıflandırma ve embedding çıkarma istek şeması."""
    metin: str = Field(..., min_length=3, max_length=2000, description="Analiz edilecek metin")
    kategori: Optional[str] = Field(default="Genel", description="İsteğe bağlı metin kategorisi")
    embedding_iste: bool = Field(default=False, description="Vektör embedding çıktısı istensin mi?")

    @field_validator("metin")
    @classmethod
    def bos_metin_engelle(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Metin boşluklardan ibaret olamaz.")
        return v.strip()


class MetinTahminYaniti(BaseModel):
    """Metin tahmin yanıt şeması."""
    metin: str
    tahmin_edilen_etiket: str
    olasilik: float = Field(..., ge=0.0, le=1.0)
    tum_olasiliklar: Dict[str, float]
    vektor_embedding: Optional[List[float]] = None
    gecikme_ms: float


class GorselAnalizYaniti(BaseModel):
    """Görüntü analiz ve tespit yanıt şeması."""
    dosya_adi: str
    boyut_bayt: int
    tespit_edilen_nesneler: List[Dict[str, Any]]
    en_baskin_renk: List[int]
    gecikme_ms: float


class RAGSorguIstegi(BaseModel):
    """RAG doküman soru-cevap istek şeması."""
    soru: str = Field(..., min_length=3, max_length=500, description="Kullanıcı sorusu")
    top_k: int = Field(default=3, ge=1, le=10, description="Getirilecek parça sayısı")
    filtre_kategori: Optional[str] = Field(default=None, description="Filtrelenecek doküman kategorisi")


class RAGSorguYaniti(BaseModel):
    """RAG soru-cevap yanıt şeması."""
    soru: str
    yanit: str
    kaynaklar: List[str]
    guven_skoru: float
    durum: str
    gecikme_ms: float


class HataDetayi(BaseModel):
    """Hata yanıt şeması."""
    hata: str
    mesaj: str
    kod: int
