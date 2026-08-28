"""
Vision Transformer Modeline LoRA Enjekte Edici ve PEFT Yöneticisi
-----------------------------------------------------------------
Önceden eğitilmiş bir Vision Transformer modelinin tüm omurgasını dondurup (Freeze),
belirlenen dikkat katmanlarına (w_q, w_v, vb.) LoRA adaptörleri ekleyen motor.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn

from .lora_katmani import LoRADogrusalKatman


class ViTLoRAEnjekteEdici:
    """
    MiniVisionTransformer modeline LoRA adaptörleri enjekte eden ve
    parametre verimliliği istatistiklerini hesaplayan yönetici sınıf.
    """
    def __init__(
        self,
        hedef_moduller: List[str] = ["w_q", "w_v"],
        r: int = 4,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
        yeni_sinif_sayisi: int = None
    ):
        self.hedef_moduller = hedef_moduller
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.yeni_sinif_sayisi = yeni_sinif_sayisi
        self.enjekte_edilen_katmanlar: List[LoRADogrusalKatman] = []

    def enjekte_et(self, model: nn.Module) -> nn.Module:
        """
        1. Modelin tüm ana parametrelerini dondurur (requires_grad = False).
        2. Hedef modülleri (ör. w_q, w_v) LoRADogrusalKatman ile sarmalar.
        3. Yeni bir görev sınıf sayısı verilmişse sınıflandırma kafasını günceller.
        """
        # 1. Tüm ana modeli dondur
        for param in model.parameters():
            param.requires_grad = False

        self.enjekte_edilen_katmanlar.clear()

        # 2. Transformer Encoder bloklarındaki hedef katmanları değiştir
        for blok_idx, blok in enumerate(model.bloklar):
            for modul_adi in self.hedef_moduller:
                if hasattr(blok.dikkat, modul_adi):
                    eski_katman = getattr(blok.dikkat, modul_adi)
                    if isinstance(eski_katman, nn.Linear):
                        lora_katmani = LoRADogrusalKatman(
                            orijinal_katman=eski_katman,
                            r=self.r,
                            lora_alpha=self.lora_alpha,
                            lora_dropout=self.lora_dropout
                        )
                        setattr(blok.dikkat, modul_adi, lora_katmani)
                        self.enjekte_edilen_katmanlar.append(lora_katmani)

        # 3. İsteğe bağlı yeni downstream sınıflandırma kafası
        if self.yeni_sinif_sayisi is not None:
            gomulme_boyutu = model.head.in_features
            cihaz = model.head.weight.device
            veri_tipi = model.head.weight.dtype
            model.head = nn.Linear(gomulme_boyutu, self.yeni_sinif_sayisi, device=cihaz, dtype=veri_tipi)
            model.head.weight.requires_grad = True
            if model.head.bias is not None:
                model.head.bias.requires_grad = True
            model.sinif_sayisi = self.yeni_sinif_sayisi
        else:
            # Mevcut sınıflandırma kafasını eğitilebilir yap
            for param in model.head.parameters():
                param.requires_grad = True

        return model

    def parametre_istatistikleri(self, model: nn.Module) -> Dict[str, Any]:
        """
        Dondurulan, eğitilebilir ve toplam parametre sayılarını hesaplar.
        """
        toplam_param = sum(p.numel() for p in model.parameters())
        egitilebilir_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        dondurulan_param = toplam_param - egitilebilir_param
        oran = (egitilebilir_param / toplam_param) * 100.0

        return {
            "toplam_param": toplam_param,
            "dondurulan_param": dondurulan_param,
            "egitilebilir_param": egitilebilir_param,
            "egitilebilir_yuzde": oran,
            "lora_katman_sayisi": len(self.enjekte_edilen_katmanlar)
        }

    def birlestir_tum_adapterleri(self):
        """Tüm enjekte edilmiş adaptörlerin ağırlıklarını birleştirir (Inference modu)."""
        for katman in self.enjekte_edilen_katmanlar:
            katman.birlestir()

    def ayir_tum_adapterleri(self):
        """Tüm adaptörleri geri ayırır (Fine-tuning modu)."""
        for katman in self.enjekte_edilen_katmanlar:
            katman.ayir()

    def state_dict_sadece_lora(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Sadece eğitilebilir LoRA parametrelerini ve sınıflandırma kafasını içeren
        son derece hafif bir state_dict döndürür (Megabaytlar yerine Kilobaytlar!).
        """
        return {
            k: v.cpu() for k, v in model.state_dict().items()
            if "lora_" in k or "head" in k
        }
