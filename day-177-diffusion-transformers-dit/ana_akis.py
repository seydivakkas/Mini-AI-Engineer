"""
Day 177: Diffusion Transformers (DiT - Sora & Flux Omurgası) Ana Akışı (FAZ 9).
Patchify Mekansal Ayrıştırma, adaLN-Zero Modülasyonu ve Teşhis Panosu.
"""

import os
import sys
import torch

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.dit_modeli import DiffusionTransformer
from src.gorsellestirici import DiTGorsellestirici


def main():
    print("=" * 110)
    print(">>> Day 177 (FAZ 9): DIFFUSION TRANSFORMERS (DiT): PATCHIFY, adaLN-ZERO & SCALING LAWS")
    print("=" * 110)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # -------------------------------------------------------------
    # 1. DIT MODELİ VE PATCHIFY İLERİ BESLEMESİ
    # -------------------------------------------------------------
    print("\n[1/3] Diffusion Transformer (DiT-XL/2) Modeli Başlatılıyor...")
    model = DiffusionTransformer(
        input_size=16,
        patch_size=2,
        in_channels=4,
        hidden_size=128,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        cond_size=128,
    )

    x = torch.randn(1, 4, 16, 16)
    t = torch.tensor([500])

    pred_noise = model(x, t)

    print(f"  • Girdi Gizli Tensörü (z_t) : {list(x.shape)} [Batch=1, C=4, 16x16]")
    print(f"  • Patchify Yama Boyutu      : p=2x2 (Toplam {model.num_patches} Görsel Token)")
    print(f"  • Zaman Adımı Koşulu (t)    : {t.item()}")
    print(f"  • Tahmin Edilen Gürültü     : {list(pred_noise.shape)} [Batch=1, C=4, 16x16]")

    # -------------------------------------------------------------
    # 2. DIT ÖLÇEKLENME YASASI ANALİZİ
    # -------------------------------------------------------------
    print("\n[2/3] DiT Model Ailesi Ölçeklenme ve Performans Analizi Yapılıyor...")
    rapor = DiffusionTransformer.ornek_dit_karsilastirma_raporu()

    print("\n" + "-" * 90)
    print(f"{'Model Mimarisi':<22} | {'Parametre':<12} | {'Hesaplama':<14} | {'FID Skoru':<12} | {'Kullanım Alanı'}")
    print("-" * 90)
    for m in rapor["model_varyantlari"]:
        print(f"{m['model']:<22} | {m['param_m']}M{'':<9} | {m['gflops']} GFLOPs{'':<4} | {m['fid']:<12.2f} | {m['aciklama']}")
    print("-" * 90)
    print(f"Mimari Üstünlük: {rapor['unet_vs_dit_avantaji']}")

    # -------------------------------------------------------------
    # 3. 6 PANELLİ TEŞHİS PANOSU ÜRETİMİ
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli DiT Teşhis Panosu Üretiliyor...")
    gorsellestirici = DiTGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "dit_diffusion_transformers_paneli.png")
    gorsellestirici.pano_olustur(rapor, kayit_yolu=cikis_resmi)

    print("\n" + "=" * 110)
    print("✓ Day 177: DIFFUSION TRANSFORMERS (DiT) BAŞARIYLA TAMAMLANDI!")
    print("=" * 110)


if __name__ == "__main__":
    main()
