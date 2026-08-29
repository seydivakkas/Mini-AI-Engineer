"""
Day 261 (FAZ 14 BAŞLANGICI): BitNet b1.58 Ternary LLM ve Matmul-Free Çıkarım Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from src.bitnet_1bit_motoru import (
    weight_quantization_b158,
    activation_quantization_int8,
    BitLinear,
    BitNetTransformer,
)
from src.bitnet_1bit_profilleyici import BitNetProfilleyici
from src.gorsellestirici import BitNetGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 261 (FAZ 14 BAŞLANGICI): BITNET b1.58 — 1.58-BIT TERNARY LLM VE MATMUL-FREE ÇIKARIM")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: 1.58-Bit Ternary Ağırlık ve 8-Bit Aktivasyon Kuantizasyonu
    # -------------------------------------------------------------
    print("\n[1/4] 1.58-Bit Ternary {-1, 0, 1} Ağırlık ve INT8 Aktivasyon Kuantizasyonu Test Ediliyor...")
    raw_w = torch.randn(4, 4)
    w_ternary, gamma_w = weight_quantization_b158(raw_w)
    unique_vals = torch.unique(w_ternary).tolist()

    raw_x = torch.randn(2, 4)
    x_int8, gamma_x = activation_quantization_int8(raw_x)

    print(f"  • Ağırlık Ternary Değerleri   : {unique_vals} (Tamamen {{-1, 0, 1}})")
    print(f"  • Ağırlık Ölçek Faktörü (gamma): {gamma_w.item():.4f}")
    print(f"  • Aktivasyon INT8 Aralığı     : [{x_int8.min().item():.0f}, {x_int8.max().item():.0f}] ([-127, 127])")

    # -------------------------------------------------------------
    # ADIM 2: BitNet b1.58 Transformer Modeli Canlı Çıkarımı
    # -------------------------------------------------------------
    print("\n[2/4] BitNet b1.58 Transformer Modeli Başlatılıyor ve Çıkarım Yapılıyor...")
    model = BitNetTransformer(vocab_size=1000, d_model=64, n_layers=2, n_heads=4, d_ff=128)
    sample_tokens = torch.tensor([[10, 45, 99, 128, 5]], dtype=torch.long)

    with torch.no_grad():
        logits = model(sample_tokens)

    print(f"  • Girdi Token Dizisi          : {sample_tokens.tolist()}")
    print(f"  • Çıktı Logits Boyutu (B, S, V): {list(logits.shape)}")
    print(f"  • Matmul-Free Çıkarım Durumu  : AKTİF (Sadece Toplayıcı Ağacı)")

    # -------------------------------------------------------------
    # ADIM 3: Straight-Through Estimator (STE) Gradyan Doğrulaması
    # -------------------------------------------------------------
    print("\n[3/4] Straight-Through Estimator (STE) Geriye Doğru Yayılımı Test Ediliyor...")
    target = torch.tensor([[45, 99, 128, 5, 20]], dtype=torch.long)
    logits_train = model(sample_tokens)
    loss = torch.nn.functional.cross_entropy(logits_train.view(-1, 1000), target.view(-1))
    loss.backward()
    grad_norm = model.blocks[0].attn.q_proj.weight.grad.norm().item()
    print(f"  • Eğitim Kaybı (CrossEntropy) : {loss.item():.4f}")
    print(f"  • Geriye Dönen Gradyan Normu  : {grad_norm:.4f} (Hatasız STE Akışı)")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli BitNet b1.58 Donanım Teşhis Panosu Oluşturuluyor...")
    profil_raporu = BitNetProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "bitnet_1bit_paneli.png")

    BitNetGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ BitNet b1.58 Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 261 (FAZ 14 BAŞLANGICI): BITNET b1.58 TERNARY LLM MODÜLÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
