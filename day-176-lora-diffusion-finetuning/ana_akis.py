"""
Day 176: Difüzyon Modellerinde LoRA & DreamBooth İnce Ayarı Ana Akışı (FAZ 9).
Düşük Dereceli Matris Ayrışımı (B*A), Özel Özne Eğitimi ve Teşhis Panosu.
"""

import os
import sys
import torch
import torch.nn as nn

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.lora_katmani import LoRALinear
from src.lora_enjektoru import LoRAEnjektoru
from src.dreambooth_egitici import DreamBoothEgitici
from src.gorsellestirici import LoRAGorsellestirici


class DummyUNet(nn.Module):
    """Simüle Edilmiş Cross-Attention UNet Bloğu."""
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(512, 512)
        self.to_k = nn.Linear(512, 512)
        self.to_v = nn.Linear(512, 512)
        self.to_out = nn.Linear(512, 512)
        self.ffn = nn.Linear(512, 2048)


def main():
    print("=" * 110)
    print(">>> Day 176 (FAZ 9): LORA & DREAMBOOTH DIFFUSION FINE-TUNING: LOW-RANK MATRICES & PRIOR PRESERVATION")
    print("=" * 110)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # -------------------------------------------------------------
    # 1. UNET VE LORA ENJEKSİYONU
    # -------------------------------------------------------------
    print("\n[1/3] Difüzyon Cross-Attention Katmanlarına LoRA Enjekte Ediliyor...")
    model = DummyUNet()
    onceki_istatistik = LoRAEnjektoru.parametre_sayilarini_getir(model)

    degisen_sayisi = LoRAEnjektoru.lora_enjekte_et(model, r=16, lora_alpha=32.0)
    sonraki_istatistik = LoRAEnjektoru.parametre_sayilarini_getir(model)

    print(f"  • Dönüştürülen Cross-Attention Katmanı Sayısı: {degisen_sayisi}")
    print(f"  • Orijinal Toplam Parametre Sayısı          : {onceki_istatistik['toplam_parametre']:,}")
    print(f"  • Eğitilebilir LoRA Parametre Sayısı        : {sonraki_istatistik['egitilebilir_parametre']:,} (%{sonraki_istatistik['egitilebilir_oran_yuzde']})")
    print(f"  • Bellek ve Depolama Tasarrufu             : {sonraki_istatistik['tasarruf_orani']}")

    # -------------------------------------------------------------
    # 2. DREAMBOOTH SINIF KORUMA KAYBI SİMÜLASYONU
    # -------------------------------------------------------------
    print("\n[2/3] DreamBooth Çift Kayıp ve Sınıf Koruma Analizi Yapılıyor...")
    egitici = DreamBoothEgitici(prior_loss_weight=1.0)
    rapor = DreamBoothEgitici.ornek_lora_raporu_getir()

    print(f"\n>> HEDEF KAVRAM: {rapor['hedef_kavram']}")
    print("-" * 90)
    print(f"{'LoRA Derecesi (Rank)':<20} | {'Dosya Boyutu':<15} | {'Parametre Oranı':<18} | {'Özne Sadakati'}")
    print("-" * 90)
    for r_exp in rapor["rank_deneyleri"]:
        print(f"r = {r_exp['r']:<16} | {r_exp['dosya_mb']} MB{'':<8} | %{r_exp['param_yuzde']:<16.2f} | %{r_exp['sadakat']*100:.1f} ({r_exp['durum']})")
    print("-" * 90)
    print(f"{'Genel Depolama Tasarrufu':<38} | {rapor['dosya_boyut_kazanci']}")
    print(f"{'Sınıf Koruma (Prior Fidelity)':<38} | %{rapor['sinif_koruma_skoru']*100:.1f} (Sıfır Dil Kayması)")

    # -------------------------------------------------------------
    # 3. 6 PANELLİ TEŞHİS PANOSU ÜRETİMİ
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli LoRA & DreamBooth Teşhis Panosu Üretiliyor...")
    gorsellestirici = LoRAGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "lora_diffusion_finetuning_paneli.png")
    gorsellestirici.pano_olustur(rapor, kayit_yolu=cikis_resmi)

    print("\n" + "=" * 110)
    print("✓ Day 176: LORA & DREAMBOOTH DIFFUSION BAŞARIYLA TAMAMLANDI!")
    print("=" * 110)


if __name__ == "__main__":
    main()
