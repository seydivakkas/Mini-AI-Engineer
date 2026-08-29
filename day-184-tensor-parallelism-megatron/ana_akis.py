"""
Day 184: Megatron-LM Tensor Parallelism (TP) Ana Çalıştırma ve Doğrulama Akışı.
"""

import os
import sys
import torch
import torch.nn as nn

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.megatron_tp_motoru import ColumnParallelLinear, RowParallelLinear
from src.megatron_transformer_blok import (
    MegatronMLP,
    MegatronSelfAttention,
    MegatronTransformerKatmani,
    TPDogrulayici,
)
from src.gorsellestirici import TPGorsellestirici


def main():
    print("=" * 110)
    print(">>> Day 184 (FAZ 10): NVIDIA MEGATRON-LM TENSOR PARALLELISM (TP) ENGINE")
    print("=" * 110)

    # -------------------------------------------------------------
    # ADIM 1: Sütun ve Satır Paralel Doğrusal Katman Testi
    # -------------------------------------------------------------
    print("\n[1/4] ColumnParallelLinear ve RowParallelLinear Katmanları Test Ediliyor...")
    tp_world_size = 4
    col_layer = ColumnParallelLinear(512, 2048, tp_world_size=tp_world_size, tp_rank=0)
    row_layer = RowParallelLinear(2048, 512, tp_world_size=tp_world_size, tp_rank=0)

    x_dummy = torch.randn(8, 512)
    col_out = col_layer(x_dummy)
    row_out = row_layer(col_out)

    print(f"  • Giriş Boyutu                 : {list(x_dummy.shape)}")
    print(f"  • Sütun Paralel Çıktı (1/K Dilim): {list(col_out.shape)} (512 -> 512)")
    print(f"  • Satır Paralel Çıktı (Tam Boyut): {list(row_out.shape)} (512 -> 512)")
    print("  ✓ Column & Row Parallel Katman Geçişleri Başarılı!")

    # -------------------------------------------------------------
    # ADIM 2: Megatron Multi-Head Attention & Transformer Katmanı
    # -------------------------------------------------------------
    print("\n[2/4] Megatron Transformer Bloğu (Heads/K + Fused 2 All-Reduce) İnceleniyor...")
    transformer_layer = MegatronTransformerKatmani(
        hidden_size=512,
        num_heads=8,
        ffn_hidden_size=2048,
        tp_world_size=tp_world_size,
        tp_rank=0,
    )

    x_seq = torch.randn(4, 32, 512)  # [B, S, D]
    out_seq = transformer_layer(x_seq)

    print(f"  • Dizi Giriş Şekli             : {list(x_seq.shape)}")
    print(f"  • Transformer Çıktı Şekli      : {list(out_seq.shape)}")
    print(f"  • Katman Başı All-Reduce Sayısı: 2 All-Reduce (1 Attention + 1 MLP)")
    print("  ✓ Megatron Transformer İleri Geçiş Başarılı!")

    # -------------------------------------------------------------
    # ADIM 3: Tek GPU vs Megatron TP Matematiksel Eşdeğerlik Doğrulaması
    # -------------------------------------------------------------
    print("\n[3/4] Matematiksel Eşdeğerlik Doğrulaması Yapılıyor (TP=2, TP=4, TP=8)...")
    dogrulama_sonuclari = []
    for k in [2, 4, 8]:
        res = TPDogrulayici.mlp_esdegerlik_dogrula(hidden_size=256, ffn_hidden_size=1024, tp_world_size=k)
        dogrulama_sonuclari.append(res)
        print(
            f"  • TP = {k} GPU -> Maksimum Mutlak Hata: {res['maksimum_mutlak_hata']:.2e} | "
            f"Eşleşiyor: {res['matematiksel_olarak_eslesiyor']}"
        )

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Görsel Teşhis Panosu Üretimi
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Megatron TP Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "tensor_parallelism_megatron_paneli.png")

    TPGorsellestirici.tp_teshis_paneli_olustur(
        dogrulama_sonuclari=dogrulama_sonuclari,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Megatron TP Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 110)
    print("✓ Day 184: MEGATRON-LM TENSOR PARALLELISM (TP) BAŞARIYLA TAMAMLANDI!")
    print("=" * 110)


if __name__ == "__main__":
    main()
