"""
Day 174: Metinden Görüntüye: UNet & DiT Cross-Attention Mekanizması Ana Akışı (FAZ 9).
CLIP/T5 Metin Koşullandırma, Mekansal Çapraz Dikkat Haritaları ve Teşhis Panosu.
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

from src.text_conditioned_unet_dit import TextConditionedDiffusionBlock
from src.dikkat_haritasi_analizoru import DikkatHaritasiAnalizoru
from src.gorsellestirici import CrossAttentionGorsellestirici


def main():
    print("=" * 110)
    print(">>> Day 174 (FAZ 9): TEXT-TO-IMAGE: SPATIAL CROSS-ATTENTION & PIXEL-TOKEN ATTENTION MAPS (UNET & DIT)")
    print("=" * 110)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # -------------------------------------------------------------
    # 1. METİN KOŞULLU DİFÜZYON BLOĞU VE ÇAPRAZ DİKKAT
    # -------------------------------------------------------------
    print("\n[1/3] Metin Koşullu UNet / DiT Difüzyon Bloğu Başlatılıyor...")
    block = TextConditionedDiffusionBlock(channels=128, context_dim=256, heads=4)

    # 16x16 mekansal VAE tensörü ve 8 metin belirteci (CLIP Text Embeddings)
    x_latent = torch.randn(1, 128, 16, 16)
    context_text = torch.randn(1, 8, 256)

    out_latent, attn_map = block(x_latent, context_text)

    print(f"  • Giriş VAE Gizli Tensörü      : {list(x_latent.shape)} [Batch=1, Channels=128, 16x16]")
    print(f"  • CLIP/T5 Metin Gömüşü (c)     : {list(context_text.shape)} [Batch=1, Tokens=8, Dim=256]")
    print(f"  • Çıkış Gizli Haritası         : {list(out_latent.shape)}")
    print(f"  • Mekansal Çapraz Dikkat Haritası: {list(attn_map.shape)} [Batch=1, Pikseller=256, Tokens=8]")

    # -------------------------------------------------------------
    # 2. KELİME BAZLI MEKANSAL DİKKAT ANALİZİ
    # -------------------------------------------------------------
    print("\n[2/3] Kelime Bazlı Mekansal Odak ve Enerji Yoğunluğu Analiz Ediliyor...")
    rapor = DikkatHaritasiAnalizoru.ornek_cross_attention_raporu()

    print(f"\n>> PROMPT: \"{rapor['prompt']}\"")
    print("-" * 80)
    print(f"{'Kelime':<15} | {'Dikkat Enerjisi':<20} | {'Mekansal Odak Bölgesi'}")
    print("-" * 80)
    for k in rapor["kelime_skorlari"]:
        print(f"{k['kelime']:<15} | %{k['enerji']*100:<19.1f} | {k['odak']}")
    print("-" * 80)
    print(f"{'Metin-Piksel Hizalama Doğruluğu':<38} | %{rapor['metin_piksel_hizalama_dogrulugu']*100:.1f} (Yüksek Anlamsal Bütünlük)")
    print(f"{'Ortalama Çapraz Dikkat Entropisi':<38} | {rapor['ortalama_cross_attention_entropisi']} nats")

    # -------------------------------------------------------------
    # 3. 6 PANELLİ TEŞHİS PANOSU ÜRETİMİ
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli Cross-Attention Teşhis Panosu Üretiliyor...")
    gorsellestirici = CrossAttentionGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "cross_attention_text_to_image_paneli.png")
    gorsellestirici.pano_olustur(rapor, kayit_yolu=cikis_resmi)

    print("\n" + "=" * 110)
    print("✓ Day 174: CROSS-ATTENTION TEXT-TO-IMAGE BAŞARIYLA TAMAMLANDI!")
    print("=" * 110)


if __name__ == "__main__":
    main()
