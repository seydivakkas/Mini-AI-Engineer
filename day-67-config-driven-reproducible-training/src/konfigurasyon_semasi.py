"""
Pydantic v2 Tip Guvenli Konfigurasyon Semalari (Config Schemas)
==============================================================
YAML dosyalarini tiplere baglayan, aralik ve deger kontrollerini
otomatik gerceklestiren hiyerarsik veri modelleri.
"""

from typing import List, Literal, Optional, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator


class VeriKonfigurasyonu(BaseModel):
    """Veri seti ve DataLoader parametreleri."""
    veri_seti_adi: str = Field(default="sentetik_veri", description="Veri kumesi ismi")
    girdi_boyutu: List[int] = Field(default=[3, 32, 32], description="[Kanal, Yukseklik, Genislik]")
    ornek_sayisi: int = Field(default=500, ge=10, description="Uretilecek ornek sayisi")
    batch_size: int = Field(default=32, ge=1, le=1024, description="Mini-batch boyutu")
    num_workers: int = Field(default=0, ge=0, description="DataLoader is parcacigi sayisi")
    pin_memory: bool = Field(default=False, description="GPU bellek kilitleme")
    egitim_orani: float = Field(default=0.8, gt=0.0, lt=1.0, description="Train boluntusu orani")

    @field_validator("girdi_boyutu")
    @classmethod
    def girdi_boyutu_kontrol(cls, v: List[int]) -> List[int]:
        if len(v) != 3:
            raise ValueError("Girdi boyutu tam olarak 3 elemanli olmalidir: [C, H, W]")
        if any(dim <= 0 for dim in v):
            raise ValueError("Tum girdi boyutlari pozitif tam sayi olmalidir.")
        return v


class ModelKonfigurasyonu(BaseModel):
    """Sinir agi mimari parametreleri."""
    mimari_adi: str = Field(default="ModulerVisionNet", description="Model sinifi ismi")
    girdi_kanali: int = Field(default=3, ge=1, description="Girdi kanal sayisi")
    sinif_sayisi: int = Field(default=5, ge=2, description="Hedef sinif adedi")
    taban_kanal: int = Field(default=32, ge=8, le=512, description="Baslangic evrisim kanali")
    dropout_orani: float = Field(default=0.15, ge=0.0, le=0.8, description="Dropout orani")


class OptimizerKonfigurasyonu(BaseModel):
    """Optimizasyon algoritmasi parametreleri."""
    tur: Literal["adamw", "adam", "sgd"] = Field(default="adamw", description="Optimizer turu")
    lr: float = Field(default=1e-3, gt=0.0, le=1.0, description="Ogrenme orani (Learning Rate)")
    weight_decay: float = Field(default=1e-2, ge=0.0, description="Agirlik bozunumu L2 cezasi")
    betas: Tuple[float, float] = Field(default=(0.9, 0.999), description="Adam momentum katsayilari")
    momentum: float = Field(default=0.9, ge=0.0, le=1.0, description="SGD momentum katsayisi")


class SchedulerKonfigurasyonu(BaseModel):
    """Ogrenme orani zamanlayicisi (LR Scheduler) parametreleri."""
    tur: Literal["cosine", "step", "none"] = Field(default="cosine", description="Scheduler turu")
    t_max: int = Field(default=10, ge=1, description="Maksimum epoch donemi")
    eta_min: float = Field(default=1e-5, ge=0.0, description="Minimum ogrenme orani tabani")
    warmup_epochs: int = Field(default=2, ge=0, description="Isinma donemi epoch sayisi")


class EgitimKonfigurasyonu(BaseModel):
    """Genel egitim dongusu ve determinizm parametreleri."""
    epoch_sayisi: int = Field(default=10, ge=1, le=1000, description="Toplam egitim epoch'u")
    tohum: int = Field(default=42, ge=0, description="Deterministik rastgelelik tohumu (Seed)")
    deterministik_mod: bool = Field(default=True, description="Tam deterministik CUDA/CPU modu")
    amp_aktif: bool = Field(default=False, description="Otomatik karma hassasiyet (AMP)")
    grad_clip_norm: float = Field(default=1.0, gt=0.0, description="Gradyan kirpma (Gradient Clipping) normu")
    checkpoint_araligi: int = Field(default=5, ge=1, description="Checkpoint kaydetme sikligi (epoch)")


class KokKonfigurasyon(BaseModel):
    """Tum alt konfigurasyonlari toplayan ana kok sema."""
    deney_adi: str = Field(default="deney_varsayilan", description="Deney etiketi")
    versiyon: str = Field(default="1.0.0", description="Konfigurasyon versiyonu")
    veri: VeriKonfigurasyonu = Field(default_factory=VeriKonfigurasyonu)
    model: ModelKonfigurasyonu = Field(default_factory=ModelKonfigurasyonu)
    optimizer: OptimizerKonfigurasyonu = Field(default_factory=OptimizerKonfigurasyonu)
    scheduler: SchedulerKonfigurasyonu = Field(default_factory=SchedulerKonfigurasyonu)
    egitim: EgitimKonfigurasyonu = Field(default_factory=EgitimKonfigurasyonu)

    @model_validator(mode="after")
    def uyumluluk_denetle(self) -> "KokKonfigurasyon":
        """Model girdi kanali ile veri seti kanal uyumunu garanti eder."""
        if self.model.girdi_kanali != self.veri.girdi_boyutu[0]:
            raise ValueError(
                f"Model girdi kanali ({self.model.girdi_kanali}) ile "
                f"veri girdi boyutu kanali ({self.veri.girdi_boyutu[0]}) eslesmiyor!"
            )
        return self
