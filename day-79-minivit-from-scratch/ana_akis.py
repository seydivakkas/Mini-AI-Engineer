"""
Day 79: Sıfırdan Mini Vision Transformer (MiniViT) Ana Akış Scripti
-------------------------------------------------------------------
Görseli 4x4 yamalara bölen, [CLS] token ve öğrenilebilir 1D pozisyonel gömülmeleri
ekleyip 4 katmanlı Transformer Encoder ile sınıflandıran laboratuvar.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.minivit_modeli import MiniVisionTransformer
from src.dikkat_haritasi import ViTDikkatCikarici
from src.gorsellestirici import MiniViTGorsellestirici


def tohum_belirle(tohum: int = 42):
    random.seed(tohum)
    np.random.seed(tohum)
    torch.manual_seed(tohum)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tohum)


def main():
    print("=" * 80)
    print("🚀 Day 79: Sıfırdan Mini Vision Transformer (MiniViT) Mimarisi")
    print("=" * 80)

    tohum_belirle(42)
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📌 Çalışma Ortamı Cihazı: {cihaz.upper()}")

    # 1. MiniViT Model Parametreleri
    gorsel_boyutu = 32
    yama_boyutu = 4
    giris_kanali = 3
    sinif_sayisi = 10
    gomulme_boyutu = 64
    derinlik = 4
    kafa_sayisi = 4
    mlp_orani = 4

    num_patches = (gorsel_boyutu // yama_boyutu) ** 2

    print(f"\n[1/5] Mini Vision Transformer Başlatılıyor...")
    print(f"  • Girdi Çözünürlüğü: {gorsel_boyutu}x{gorsel_boyutu}x{giris_kanali}")
    print(f"  • Yama (Patch) Boyutu: {yama_boyutu}x{yama_boyutu} ──> Toplam Yama Sayısı: {num_patches}")
    print(f"  • Dizi Uzunluğu (1 CLS + {num_patches} Patch): {num_patches + 1} Token")
    print(f"  • Gömülme Boyutu (D): {gomulme_boyutu}, Kafa: {kafa_sayisi}, Derinlik (L): {derinlik}")

    model = MiniVisionTransformer(
        gorsel_boyutu=gorsel_boyutu,
        yama_boyutu=yama_boyutu,
        giris_kanali=giris_kanali,
        sinif_sayisi=sinif_sayisi,
        gomulme_boyutu=gomulme_boyutu,
        derinlik=derinlik,
        kafa_sayisi=kafa_sayisi,
        mlp_orani=mlp_orani,
        dropout_orani=0.1
    ).to(cihaz)

    # 2. Sentetik Görsel Batch ve İleri Geçiş
    print("\n[2/5] İleri Geçiş (Forward Pass) ve Logit Üretimi...")
    b_size = 4
    # Sentetik renkli görsel oluştur (merkezde belirgin bir daire)
    x = torch.randn(b_size, giris_kanali, gorsel_boyutu, gorsel_boyutu, device=cihaz)
    # 0. görselin merkezine parlak desen ekle
    x[0, :, 10:22, 10:22] += 2.0
    x = torch.clamp(x, -2.0, 2.0)
    targets = torch.tensor([3, 1, 7, 0], device=cihaz)

    logitler, dikkat_listesi = model(x, dikkat_haritalarini_don=True)

    print(f"  ✓ Model Çıktı Logit Şekli: {list(logitler.shape)} (Beklenen: [{b_size}, {sinif_sayisi}])")
    print(f"  ✓ Encoder Katman Sayısı: {len(dikkat_listesi)} katman")
    print(f"  ✓ Her Katmanın Dikkat Matrisi Şekli: {list(dikkat_listesi[0].shape)} (Beklenen: [{b_size}, {kafa_sayisi}, {num_patches + 1}, {num_patches + 1}])")

    # 3. Model Parametre Dağılımı
    print("\n[3/5] Model Parametre Dağılımı Hesaplanıyor...")
    patch_param = sum(p.numel() for p in model.patch_embed.parameters())
    token_pe_param = model.cls_token.numel() + model.pos_embed.numel()
    encoder_param = sum(p.numel() for p in model.bloklar.parameters())
    head_param = sum(p.numel() for p in model.head.parameters()) + sum(p.numel() for p in model.norm.parameters())
    toplam_param = sum(p.numel() for p in model.parameters())

    param_dict = {
        "Yama Gömülme (Conv2D)": patch_param,
        "CLS & Pozisyonel Gömülme": token_pe_param,
        "Encoder Blokları (MHSA+FFN)": encoder_param,
        "Norm & MLP Head": head_param
    }

    print("=" * 65)
    print(f"{'Bileşen':^30} | {'Parametre Sayısı':^16} | {'Oran (%)':^12}")
    print("=" * 65)
    for k, v in param_dict.items():
        print(f"{k:<30} | {v:^16,d} | %{v/toplam_param*100:^10.1f}")
    print("-" * 65)
    print(f"{'TOPLAM MİNİVİT KAPASİTESİ':<30} | {toplam_param:^16,d} | %100.0")
    print("=" * 65)

    # 4. Geriye Yayılım ve Gradyan Doğrulaması
    print("\n[4/5] Geriye Yayılım (Backpropagation) ve Gradyan Doğrulaması...")
    loss = F.cross_entropy(logitler, targets)
    loss.backward()

    print(f"  ✓ Cross-Entropy Kaybı: {loss.item():.4f}")
    print(f"  ✓ Patch Embedding Projeksiyon Gradyanı: {model.patch_embed.projeksiyon.weight.grad.norm().item():.4f}")
    print(f"  ✓ [CLS] Token Gradyanı: {model.cls_token.grad.norm().item():.4f}")
    print(f"  ✓ Pozisyonel Gömülme (pos_embed) Gradyanı: {model.pos_embed.grad.norm().item():.4f}")
    print(f"  ✓ MLP Sınıflandırma Kafası Gradyanı: {model.head.weight.grad.norm().item():.4f}")

    # 5. Attention Rollout ve 2D Isı Haritası
    print("\n[5/5] [CLS] Token Attention Rollout ve 6 Panelli Teşhis Panosu Çiziliyor...")
    isi_haritasi = ViTDikkatCikarici.cls_dikkat_haritasi_2d(
        dikkat_listesi,
        grid_boyutu=(gorsel_boyutu // yama_boyutu, gorsel_boyutu // yama_boyutu),
        orijinal_boyut=(gorsel_boyutu, gorsel_boyutu)
    )

    sample_img = x[0].detach().cpu().numpy()
    # Normalize to [0, 1] for visualization
    sample_img = (sample_img - sample_img.min()) / (sample_img.max() - sample_img.min() + 1e-12)

    gorsellestirici = MiniViTGorsellestirici()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "minivit_mimari_paneli.png")

    gorsellestirici.olustur_teshis_paneli(
        orijinal_gorsel=sample_img,
        dikkat_isi_haritasi=isi_haritasi[0],
        pos_embed_tensor=model.pos_embed.data,
        parametre_dagilimi=param_dict,
        kayit_yolu=cikti_yolu
    )
    print(f"  ✓ 6 Panelli Teşhis Panosu Kaydedildi: {cikti_yolu}")
    print("\n✅ Day 79: Sıfırdan Mini Vision Transformer Başarıyla Tamamlandı!")


if __name__ == "__main__":
    main()
