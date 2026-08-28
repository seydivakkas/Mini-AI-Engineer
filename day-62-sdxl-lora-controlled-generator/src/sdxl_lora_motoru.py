"""
Stable Diffusion XL (SDXL) Düşük Sıralı Adaptasyon (LoRA) ve Latent Difüzyon Motoru.
"""

from typing import Dict, Any, Optional, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAKatmani(nn.Module):
    """
    Dondurulmuş bir lineer katmana eklenen Low-Rank Adaptation (LoRA) katmanı.
    Matematik: W_eff = W_0 + lambda * (alpha / r) * (B @ A)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        adapter_agirligi: float = 1.0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.adapter_agirligi = adapter_agirligi
        self.skala = alpha / rank
        self.birlestirildi = False

        # LoRA matrisleri: A ~ N(0, 1/r), B = 0 (Eğitim başında sıfır etki)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor, temel_cikti: torch.Tensor) -> torch.Tensor:
        """Eğer ağırlıklar birleştirilmemişse LoRA delta çıktısını ekler."""
        if self.birlestirildi or self.adapter_agirligi == 0.0:
            return temel_cikti

        # x: (..., in_features) -> (x @ A.T) @ B.T
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return temel_cikti + (self.adapter_agirligi * self.skala * delta)

    def delta_agirlik_hesapla(self) -> torch.Tensor:
        """Mevcut LoRA matrislerinin (out_features, in_features) delta ağırlığını hesaplar."""
        return (self.adapter_agirligi * self.skala) * (self.lora_B @ self.lora_A)


class SDXLLoRAMotoru(nn.Module):
    """SDXL Cross-Attention katmanları için dinamik LoRA adaptör yöneticisi."""

    def __init__(self, d_model: int = 512, d_text: int = 768):
        super().__init__()
        self.d_model = d_model
        self.d_text = d_text

        # SDXL Çapraz Dikkat (Cross-Attention) projeksiyon katmanları (Dondurulmuş)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_text, d_model, bias=False)
        self.v_proj = nn.Linear(d_text, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Taban ağırlıkları dondur
        for p in self.parameters():
            p.requires_grad = False

        # Adlandırılmış LoRA adaptör depoları: {adapter_adi: {katman_adi: LoRAKatmani}}
        self.adaptorler: nn.ModuleDict = nn.ModuleDict()

    def adaptor_ekle(
        self,
        adapter_adi: str,
        rank: int = 8,
        alpha: float = 16.0,
        adapter_agirligi: float = 1.0
    ) -> None:
        """Yeni bir adlandırılmış LoRA adaptör kümesini sisteme kaydeder."""
        katmanlar = nn.ModuleDict({
            "q_lora": LoRAKatmani(self.d_model, self.d_model, rank, alpha, adapter_agirligi),
            "k_lora": LoRAKatmani(self.d_text, self.d_model, rank, alpha, adapter_agirligi),
            "v_lora": LoRAKatmani(self.d_text, self.d_model, rank, alpha, adapter_agirligi),
            "out_lora": LoRAKatmani(self.d_model, self.d_model, rank, alpha, adapter_agirligi)
        })
        self.adaptorler[adapter_adi] = katmanlar

    def adaptor_agirligi_ayarla(self, adapter_adi: str, yeni_agirlik: float) -> None:
        """Çalışma zamanında LoRA adaptörünün etkinleştirme çarpanını (lambda) günceller."""
        if adapter_adi not in self.adaptorler:
            raise KeyError(f"Adaptör bulunamadı: {adapter_adi}")
        for lora_katman in self.adaptorler[adapter_adi].values():
            lora_katman.adapter_agirligi = yeni_agirlik

    def simule_et_egitilmis_lora(self, adapter_adi: str, std: float = 0.05) -> None:
        """Eğitilmiş bir LoRA kontrol noktasını simüle etmek için B matrisine küçük rastgele ağırlıklar atar."""
        if adapter_adi not in self.adaptorler:
            raise KeyError(f"Adaptör bulunamadı: {adapter_adi}")
        for lora_katman in self.adaptorler[adapter_adi].values():
            nn.init.normal_(lora_katman.lora_B, mean=0.0, std=std)

    def forward(
        self,
        latent_x: torch.Tensor,
        text_embedding: torch.Tensor,
        aktif_adaptorler: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Latent ve Text koşullandırmasıyla Cross-Attention ve LoRA ileri geçişi.
        latent_x: (B, Seq_img, d_model)
        text_embedding: (B, Seq_txt, d_text)
        """
        B, S_img, _ = latent_x.shape

        # 1. Taban Projeksiyonlar
        q = self.q_proj(latent_x)
        k = self.k_proj(text_embedding)
        v = self.v_proj(text_embedding)

        # 2. Aktif LoRA Deltalarını Ekle
        if aktif_adaptorler is None:
            aktif_adaptorler = list(self.adaptorler.keys())

        for ad in aktif_adaptorler:
            if ad in self.adaptorler:
                q = self.adaptorler[ad]["q_lora"](latent_x, q)
                k = self.adaptorler[ad]["k_lora"](text_embedding, k)
                v = self.adaptorler[ad]["v_lora"](text_embedding, v)

        # 3. Ölçeklenmiş Noktasal Çarpım Dikkati (Scaled Dot-Product Attention)
        skala = 1.0 / math.sqrt(self.d_model)
        attn_skorlari = torch.matmul(q, k.transpose(-2, -1)) * skala
        attn_agirliklari = F.softmax(attn_skorlari, dim=-1)
        dikkat_ciktisi = torch.matmul(attn_agirliklari, v)

        # 4. Çıktı Projeksiyonu ve Çıkış LoRA'sı
        out = self.out_proj(dikkat_ciktisi)
        for ad in aktif_adaptorler:
            if ad in self.adaptorler:
                out = self.adaptorler[ad]["out_lora"](dikkat_ciktisi, out)

        return out


class LatentDenoisingSampler:
    """Classifier-Free Guidance (CFG) destekli Latent Gürültüden Arındırma ve Örnekleme Motoru."""

    def __init__(self, adim_sayisi: int = 20, cfg_skalasi: float = 7.5):
        self.adim_sayisi = adim_sayisi
        self.cfg_skalasi = cfg_skalasi

    def ornekle_latent(
        self,
        model: SDXLLoRAMotoru,
        kosul_metin: torch.Tensor,
        kosulsuz_metin: torch.Tensor,
        latent_sekli: Optional[Tuple[int, int, int]] = None,
        seed: int = 42
    ) -> Tuple[torch.Tensor, List[float]]:
        """Gürültülü latent uzaydan adım adım temiz latent vektör üretir."""
        torch.manual_seed(seed)
        if latent_sekli is None:
            latent_sekli = (kosul_metin.shape[0], 64, model.d_model)

        z_t = torch.randn(latent_sekli, dtype=torch.float32)
        adim_enerjileri = []

        # Zaman adımı simülasyonu
        for adim in range(self.adim_sayisi):
            # 1. Koşullu tahmin
            eps_cond = model(z_t, kosul_metin)
            # 2. Koşulsuz (unconditional) tahmin
            eps_uncond = model(z_t, kosulsuz_metin)

            # 3. Classifier-Free Guidance Formülü: eps = eps_uncond + s * (eps_cond - eps_uncond)
            eps_hat = eps_uncond + self.cfg_skalasi * (eps_cond - eps_uncond)

            # 4. Euler Adımı: Gürültüyü kademeli olarak düşür
            alfa_t = 1.0 - ((adim + 1) / self.adim_sayisi) * 0.8
            z_t = z_t * 0.9 + eps_hat * 0.1 * alfa_t
            adim_enerjileri.append(float(torch.norm(z_t).item()))

        return z_t, adim_enerjileri
