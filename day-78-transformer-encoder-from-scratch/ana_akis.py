"""
Day 78: Sıfırdan Transformer Encoder Bloğu Ana Akış Scripti
-----------------------------------------------------------
Pre-LN vs Post-LN mimarilerini, Sinüzoidal Pozisyonel Kodlamayı, GELU FFN
genişlemesini ve katmanlar arası gradyan kararlılığını test eden laboratuvar.

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

from src.encoder_blogu import TransformerEncoderGovdesi
from src.pozisyonel_kodlama import SinusoidalPozisyonelKodlama
from src.feed_forward import BeslemeliIleriAg
from src.gorsellestirici import EncoderGorsellestirici


def tohum_belirle(tohum: int = 42):
    random.seed(tohum)
    np.random.seed(tohum)
    torch.manual_seed(tohum)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tohum)


def main():
    print("=" * 80)
    print("🚀 Day 78: Sıfırdan Transformer Encoder Bloğu (Pre-LN, PE, FFN, Residual)")
    print("=" * 80)

    tohum_belirle(42)
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📌 Çalışma Ortamı Cihazı: {cihaz.upper()}")

    # 1. Mimari Parametreleri
    b_size = 4
    seq_len = 16
    d_model = 64
    kafa_sayisi = 4
    katman_sayisi = 4

    print(f"\n[1/5] Pre-LN ve Post-LN Transformer Encoder Modelleri Kuruluyor...")
    print(f"  • Model Boyutu (D): {d_model}, Kafa Sayısı (H): {kafa_sayisi}, Katman (L): {katman_sayisi}")
    print(f"  • Girdi Şekli: (Batch={b_size}, Dizi_Uzunlugu={seq_len}, Model_Boyutu={d_model})")

    encoder_pre_ln = TransformerEncoderGovdesi(
        katman_sayisi=katman_sayisi,
        model_boyutu=d_model,
        kafa_sayisi=kafa_sayisi,
        genisleme_faktoru=4,
        norm_tipi="pre_ln",
        pozisyon_tipi="sinusoidal",
        aktivasyon="gelu"
    ).to(cihaz)

    encoder_post_ln = TransformerEncoderGovdesi(
        katman_sayisi=katman_sayisi,
        model_boyutu=d_model,
        kafa_sayisi=kafa_sayisi,
        genisleme_faktoru=4,
        norm_tipi="post_ln",
        pozisyon_tipi="sinusoidal",
        aktivasyon="gelu"
    ).to(cihaz)

    # 2. İleri Geçiş ve Ara Katman Çıktılarının Alınması
    print("\n[2/5] İleri Geçiş (Forward Pass) ve Katman Temsilleri Çıkarılıyor...")
    x = torch.randn(b_size, seq_len, d_model, device=cihaz, requires_grad=True)

    out_pre, atts_pre, layers_pre = encoder_pre_ln(x, tum_katmanlari_don=True)
    out_post, atts_post, layers_post = encoder_post_ln(x, tum_katmanlari_don=True)

    print(f"  ✓ Pre-LN Çıktı Şekli: {list(out_pre.shape)}")
    print(f"  ✓ Dikkat Haritaları Sayısı: {len(atts_pre)} katman x {list(atts_pre[0].shape)}")

    # 3. Pre-LN vs Post-LN Gradyan Kararlılığı Karşılaştırması
    print("\n[3/5] Katman Bazında Gradyan Normları (Pre-LN vs Post-LN) Hesaplanıyor...")
    # Pre-LN Gradyanları (Hedef tensör ile MSE kaybı)
    hedef_t = torch.randn_like(out_pre)
    loss_pre = F.mse_loss(out_pre, hedef_t)
    loss_pre.backward(retain_graph=True)
    pre_ln_grads = [
        float(blok.ffn.fc1.weight.grad.norm().item()) for blok in encoder_pre_ln.bloklar
    ]

    # Post-LN Gradyanları
    loss_post = F.mse_loss(out_post, hedef_t)
    loss_post.backward()
    post_ln_grads = [
        float(blok.ffn.fc1.weight.grad.norm().item()) for blok in encoder_post_ln.bloklar
    ]

    print("\n" + "=" * 65)
    print(f"{'Katman':^15} | {'Pre-LN Gradyan Normu':^22} | {'Post-LN Gradyan Normu':^22}")
    print("=" * 65)
    for i in range(katman_sayisi):
        print(f"{f'Katman {i+1}':^15} | {pre_ln_grads[i]:^22.4f} | {post_ln_grads[i]:^22.4f}")
    print("=" * 65)

    # 4. Katmanlar Arası Benzerlik ve FFN Aktivasyon Analizi
    print("\n[4/5] Katmanlar Arası Kosinüs Benzerliği ve FFN Aktivasyon Dağılımı İnceleniyor...")
    katman_benzerlikleri = []
    for i in range(len(layers_pre) - 1):
        l1 = F.normalize(layers_pre[i].view(b_size * seq_len, -1), p=2, dim=1)
        l2 = F.normalize(layers_pre[i+1].view(b_size * seq_len, -1), p=2, dim=1)
        sim = (l1 * l2).sum(dim=1).mean().item()
        katman_benzerlikleri.append(float(sim))
        print(f"  ✓ Katman {i+1} -> Katman {i+2} Temsil Benzerliği: {sim:.4f}")

    # FFN GELU vs ReLU karşılaştırması
    ffn_gelu = BeslemeliIleriAg(d_model, 4, aktivasyon="gelu")
    ffn_relu = BeslemeliIleriAg(d_model, 4, aktivasyon="relu")
    test_x = torch.randn(100, 32, d_model)
    with torch.no_grad():
        gelu_out = ffn_gelu.fc1(test_x)
        gelu_akt = ffn_gelu.akt(gelu_out).numpy()
        relu_out = ffn_relu.fc1(test_x)
        relu_akt = ffn_relu.akt(relu_out).numpy()

    # 5. 6 Panelli Teşhis Panosu Çizimi
    print("\n[5/5] 6 Panelli Teşhis Panosu Çiziliyor...")
    pe_modul = SinusoidalPozisyonelKodlama(model_boyutu=64, maksimum_uzunluk=32)
    pe_matrix = pe_modul.pe[0, :32, :].numpy()

    gorsellestirici = EncoderGorsellestirici()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "transformer_encoder_paneli.png")

    gorsellestirici.olustur_teshis_paneli(
        pe_matrisi=pe_matrix,
        pre_ln_gradyanlar=pre_ln_grads,
        post_ln_gradyanlar=post_ln_grads,
        gelu_ciktilari=gelu_akt,
        relu_ciktilari=relu_akt,
        katman_benzerlikleri=katman_benzerlikleri,
        kayit_yolu=cikti_yolu
    )
    print(f"  ✓ 6 Panelli Teşhis Panosu Kaydedildi: {cikti_yolu}")
    print("\n✅ Day 78: Sıfırdan Transformer Encoder Bloğu Başarıyla Tamamlandı!")


if __name__ == "__main__":
    main()
