"""
Sıfırdan LoRA (Low-Rank Adaptation) Doğrusal Katmanı
---------------------------------------------------
Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models" formülasyonu.
Dondurulmuş ana ağırlık matrisi W_0 ve düşük dereceli eğitilebilir adaptör matrisleri A, B.
Çıkarım aşamasında sıfır gecikme için dinamik ağırlık birleştirme (Weight Merging) desteği.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import math
import torch
import torch.nn as nn


class LoRADogrusalKatman(nn.Module):
    """
    Orijinal Linear Katmanını Sarmalayan LoRA Adaptör Modülü:
    h = W_0 x + ΔW x = W_0 x + (α / r) * (B · A) x
    """
    def __init__(
        self,
        orijinal_katman: nn.Linear,
        r: int = 4,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0
    ):
        super().__init__()
        assert r > 0, "LoRA derecesi (r) pozitif bir tamsayı olmalıdır!"

        self.orijinal_katman = orijinal_katman
        self.in_features = orijinal_katman.in_features
        self.out_features = orijinal_katman.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.olcek = lora_alpha / r
        self.birlestirildi = False

        # Orijinal katmanın ağırlıklarını dondur (Freeze)
        self.orijinal_katman.weight.requires_grad = False
        if self.orijinal_katman.bias is not None:
            self.orijinal_katman.bias.requires_grad = False

        # LoRA Düşük Dereceli Matrisleri:
        # A matrisi: (r, in_features) - Kaiming Uniform / Gauss ile başlatılır
        # B matrisi: (out_features, r) - SIFIR ile başlatılır (Başlangıçta ΔW = 0)
        cihaz = orijinal_katman.weight.device
        veri_tipi = orijinal_katman.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, device=cihaz, dtype=veri_tipi))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, device=cihaz, dtype=veri_tipi))

        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        self._parametreleri_baslat()

    def _parametreleri_baslat(self):
        # A matrisini Kaiming Uniform ile başlat
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B matrisi kesinlikle SIFIR başlatılmalıdır (B * A = 0 ile başlar)
        nn.init.zeros_(self.lora_B)

    def birlestir(self):
        """
        Dağıtım / Çıkarım (Inference) için ΔW = (α/r) * (B @ A) ağırlığını
        orijinal dondurulmuş W_0 matrisine kalıcı olarak ekler.
        Bu sayede çıkarımda 0 ms ek hesaplama gecikmesi elde edilir.
        """
        if not self.birlestirildi:
            delta_w = (self.lora_B @ self.lora_A) * self.olcek
            self.orijinal_katman.weight.data += delta_w
            self.birlestirildi = True

    def ayir(self):
        """
        Birleştirilmiş ağırlıkları geri çıkararak LoRA matrislerini bağımsız eğitime hazır hale getirir.
        """
        if self.birlestirildi:
            delta_w = (self.lora_B @ self.lora_A) * self.olcek
            self.orijinal_katman.weight.data -= delta_w
            self.birlestirildi = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.birlestirildi:
            return self.orijinal_katman(x)

        # Temel Çıktı: W_0 * x
        ana_cikis = self.orijinal_katman(x)

        # LoRA Yolu: (α/r) * (x @ A.T @ B.T)
        # x: (..., in_features)
        lora_cikis = self.dropout(x) @ self.lora_A.T  # (..., r)
        lora_cikis = lora_cikis @ self.lora_B.T       # (..., out_features)

        return ana_cikis + lora_cikis * self.olcek
