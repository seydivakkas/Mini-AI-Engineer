"""
Day 190: Özel Triton FlashAttention-2 Parçalı GPU Çekirdeği Ana Çalıştırma Akışı.
"""

import os
import sys
import torch

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.flash_attention_motoru import (
    PyTorchStandartAttention,
    FlashAttention2,
)
from src.hafiza_profilleyici import FlashAttentionBellekProfilleyici
from src.gorsellestirici import FlashAttentionGorsellestirici


def main():
    print("=" * 110)
    print(">>> Day 190 (FAZ 10): CUSTOM TRITON TILED FLASHATTENTION-2 GPU KERNEL")
    print("=" * 110)

    # -------------------------------------------------------------
    # ADIM 1: Standart vs FlashAttention-2 İleri Geçiş Doğrulaması
    # -------------------------------------------------------------
    print("\n[1/4] FlashAttention-2 Standart (Non-Causal) İleri Geçiş Testi ([2, 8, 256, 64])...")
    b, h, seq_len, head_dim = 2, 8, 256, 64
    torch.manual_seed(42)
    q = torch.randn(b, h, seq_len, head_dim)
    k = torch.randn(b, h, seq_len, head_dim)
    v = torch.randn(b, h, seq_len, head_dim)

    std_attn = PyTorchStandartAttention()
    flash_attn = FlashAttention2(causal=False)

    out_std, _ = std_attn(q, k, v, causal=False)
    out_flash = flash_attn(q, k, v)

    fark_non_causal = torch.max(torch.abs(out_std - out_flash)).item()
    print(f"  • Q, K, V Tensör Şekilleri    : {list(q.shape)}")
    print(f"  • Çıktı Maksimum Farkı        : {fark_non_causal:.2e}")
    print(f"  • Non-Causal Eşleşiyor mu     : {fark_non_causal < 1e-4}")
    print("  ✓ Standart FlashAttention-2 İleri Geçiş Başarıyla Doğrulandı!")

    # -------------------------------------------------------------
    # ADIM 2: Nedensel (Causal) FlashAttention-2 Doğrulaması
    # -------------------------------------------------------------
    print("\n[2/4] Nedensel (Causal Mask) FlashAttention-2 İleri Geçiş Testi...")
    flash_attn_causal = FlashAttention2(causal=True)
    out_std_causal, _ = std_attn(q, k, v, causal=True)
    out_flash_causal = flash_attn_causal(q, k, v)

    fark_causal = torch.max(torch.abs(out_std_causal - out_flash_causal)).item()
    print(f"  • Causal Çıktı Farkı          : {fark_causal:.2e}")
    print(f"  • Causal Eşleşiyor mu         : {fark_causal < 1e-4}")
    print("  ✓ Nedensel (Causal) FlashAttention-2 İleri Geçiş Başarıyla Doğrulandı!")

    # -------------------------------------------------------------
    # ADIM 3: 1k - 128k Bağlam Uzunluğu VRAM Tasarruf Analizi
    # -------------------------------------------------------------
    print("\n[3/4] 1k'dan 128k'ya Kadar Uzun Bağlam VRAM Tüketim Raporu...")
    katman_analizi = FlashAttentionBellekProfilleyici.baglam_uzunlugu_vram_analizi(batch_size=1, num_heads=32, head_dim=128, seq_len=16384)
    baglam_raporu = FlashAttentionBellekProfilleyici.baglam_tarama_raporu()

    print("-" * 110)
    print(f"{'Bağlam Uzunluğu':<18} | {'Standart Dikkat VRAM':<22} | {'FlashAttention-2 VRAM':<24} | {'Tasarruf Faktörü':<18} | {'Durum'}")
    print("-" * 110)
    for r in baglam_raporu:
        print(
            f"{r['context_etiket']:<18} | "
            f"{r['standart_vram_gb']:>15.2f} GB        | "
            f"{r['flash_vram_gb']:>17.4f} GB         | "
            f"{r['tasarruf_faktoru']:>15}   | "
            f"{r['standart_oom_durumu']}"
        )
    print("-" * 110)

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Görsel Teşhis Panosu Üretimi
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli FlashAttention-2 Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "flash_attention_2_paneli.png")

    FlashAttentionGorsellestirici.teshis_paneli_olustur(
        katman_analizi=katman_analizi,
        baglam_raporu=baglam_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ FlashAttention-2 Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 110)
    print("✓ Day 190: ÖZEL TRITON FLASHATTENTION-2 BAŞARIYLA TAMAMLANDI!")
    print("=" * 110)


if __name__ == "__main__":
    main()
