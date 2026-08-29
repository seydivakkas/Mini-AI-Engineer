"""
Day 185: Pipeline Parallelism (PP) ve 1F1B Zaman Çizelgesi Ana Çalıştırma Akışı.
"""

import os
import sys
import torch
import torch.nn as nn

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.pipeline_paralellik_motoru import PipelineStage, P2PIletisimKuyrugu
from src.zaman_cizelgesi_1f1b import ZamanCizelgesiTuru, PipelineZamanCizelgesiMotoru
from src.gorsellestirici import PipelineGorsellestirici


def main():
    print("=" * 110)
    print(">>> Day 185 (FAZ 10): PIPELINE PARALLELISM (PP) & 1F1B SCHEDULE ENGINE")
    print("=" * 110)

    # -------------------------------------------------------------
    # ADIM 1: 4 Aşamalı Pipeline ve P2P Transfer Simülasyonu
    # -------------------------------------------------------------
    print("\n[1/4] 4 Aşamalı Pipeline Modeli ve P2P İletişim Kuyruğu Başlatılıyor...")
    num_stages = 4
    stages = []
    for s_id in range(num_stages):
        stage_layers = nn.ModuleList([
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        ])
        stages.append(PipelineStage(layers=stage_layers, stage_id=s_id, num_stages=num_stages))

    kuyruk = P2PIletisimKuyrugu(num_stages=num_stages)

    # Mikro-batch 0 İleri ve Geri Geçiş Simülasyonu
    microbatch_x = torch.randn(8, 256)
    curr_t = microbatch_x

    # İleri Geçiş (Stage 0 -> 1 -> 2 -> 3)
    for s_id in range(num_stages):
        if s_id > 0:
            curr_t = kuyruk.forward_al(to_stage=s_id, microbatch_id=0)
        out_t = stages[s_id].forward_step(microbatch_id=0, input_tensor=curr_t)
        if s_id < num_stages - 1:
            kuyruk.forward_gonder(from_stage=s_id, microbatch_id=0, tensor=out_t)

    print(f"  • Giriş Mikro-Batch Şekli : {list(microbatch_x.shape)}")
    print(f"  • Son Aşama Çıktı Şekli    : {list(out_t.shape)}")
    print(f"  • Aşama 0 Önbellek Sayısı  : {stages[0].get_cached_activation_count()} aktivasyon")
    print("  ✓ P2P İleri Transfer Zinciri Başarıyla Tamamlandı!")

    # -------------------------------------------------------------
    # ADIM 2: GPipe vs 1F1B vs Interleaved 1F1B Kıyaslama Raporu
    # -------------------------------------------------------------
    print("\n[2/4] P=8 Aşama ve M=32 Mikro-Batch İçin Zaman Çizelgesi ve Balon Analizi...")
    cizelge_raporu = PipelineZamanCizelgesiMotoru.karsilastirmali_cizelge_raporu(
        num_stages=8,
        num_microbatches=32,
        microbatch_aktivasyon_mb=250.0,
    )

    print("-" * 110)
    print(f"{'Zaman Çizelgesi Adı':<35} | {'Balon Oranı (%)':<18} | {'Tepe Aktivasyon (GB)':<22} | {'Bellek Karmaşıklığı'}")
    print("-" * 110)
    for r in cizelge_raporu:
        print(
            f"{r['cizelge_adi']:<35} | "
            f"%{r['balon_orani_yuzde']:>14.1f} | "
            f"{r['tepe_aktivasyon_gb']:>19.2f} GB | "
            f"{r['bellek_karmasikligi']}"
        )
    print("-" * 110)

    # -------------------------------------------------------------
    # ADIM 3: 6 Panelli Görsel Teşhis Panosu Üretimi
    # -------------------------------------------------------------
    print("\n[3/4] 6 Panelli Pipeline Parallelism Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "pipeline_parallelism_1f1b_paneli.png")

    PipelineGorsellestirici.pipeline_teshis_paneli_olustur(
        cizelge_raporu=cizelge_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Pipeline Parallelism Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 110)
    print("✓ Day 185: PIPELINE PARALLELISM (PP) & 1F1B BAŞARIYLA TAMAMLANDI!")
    print("=" * 110)


if __name__ == "__main__":
    main()
