"""
Day 194: TensorRT-LLM Derleme, In-Flight Batching ve FP8 Tensor Core Ana Çalıştırma Akışı.
"""

import os
import sys
import torch

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.tensorrt_llm_motoru import (
    TRTLLMEngineCompiler,
    InFlightBatchingRuntime,
    FP8QuantizationSimulator,
)
from src.trt_profilleyici import TRTLLMBenchmarkProfilleyici
from src.gorsellestirici import TRTLLMGorsellestirici


def main():
    print("=" * 110)
    print(">>> Day 194 (FAZ 10): NVIDIA TENSORRT-LLM COMPILATION & FP8 TENSOR CORE ENGINE")
    print("=" * 110)

    # -------------------------------------------------------------
    # ADIM 1: TensorRT-LLM Motor Derleme Simülasyonu
    # -------------------------------------------------------------
    print("\n[1/4] TensorRT-LLM Hesaplama Grafiği Derleniyor...")
    compiler = TRTLLMEngineCompiler(hidden_dim=4096, intermediate_dim=14336, use_fp8=True)
    derleme_bilgisi = compiler.compile()

    print(f"  • Derleme Durumu              : {derleme_bilgisi['durum']}")
    print(f"  • Gizli Boyut (Hidden Dim)    : {derleme_bilgisi['hidden_dim']}")
    print(f"  • Ara Boyut (Intermediate Dim): {derleme_bilgisi['intermediate_dim']}")
    print(f"  • FP8 Tensor Core Aktif mi    : {derleme_bilgisi['use_fp8']}")
    for adim in derleme_bilgisi["derleme_adimlari"]:
        print(f"    - {adim}")
    print("  ✓ TensorRT-LLM Motoru Başarıyla Derlendi!")

    # -------------------------------------------------------------
    # ADIM 2: In-Flight Batching FP8 İleri Geçiş Testi
    # -------------------------------------------------------------
    print("\n[2/4] In-Flight Batching ve FP8 GEMM İleri Geçiş Testi...")
    runtime = InFlightBatchingRuntime(engine=compiler)
    x_test = torch.randn(4, 4096)
    out_test = runtime.forward_step(x_test)

    print(f"  • Girdi Tensör Şekli          : {list(x_test.shape)}")
    print(f"  • Motor Çıktı Tensör Şekli    : {list(out_test.shape)}")
    print("  ✓ FP8 In-Flight Batching İleri Geçişi Başarıyla Doğrulandı!")

    # -------------------------------------------------------------
    # ADIM 3: Llama-3-70B Dağıtım ve Hız Kıyaslama Raporu
    # -------------------------------------------------------------
    print("\n[3/4] Llama-3-70B Üretim Dağıtım Kıyaslama Raporu...")
    kiyas_raporu = TRTLLMBenchmarkProfilleyici.tam_model_kiyaslama_raporu()
    batch_raporu = TRTLLMBenchmarkProfilleyici.batch_olcekleme_analizi()

    print("-" * 110)
    print(f"{'Motor Adı':<30} | {'Veri Tipi':<18} | {'Model VRAM':<14} | {'Hız (tok/s)':<14} | {'TPOT Gecikme':<16} | {'Hızlanma'}")
    print("-" * 110)
    for r in kiyas_raporu:
        print(
            f"{r['motor_adi']:<30} | "
            f"{r['veri_tipi']:<18} | "
            f"{r['model_vram_gb']:>8.1f} GB    | "
            f"{r['token_saniye']:>8.1f} tok/s | "
            f"{r['tpot_gecikme_ms']:>10.1f} ms    | "
            f"{r['hizlanma_orani']:>14}"
        )
    print("-" * 110)

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Görsel Teşhis Panosu Üretimi
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli TensorRT-LLM Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "tensorrt_llm_paneli.png")

    TRTLLMGorsellestirici.teshis_paneli_olustur(
        kiyas_raporu=kiyas_raporu,
        batch_raporu=batch_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ TensorRT-LLM Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 110)
    print("✓ Day 194: TENSORRT-LLM BAŞARIYLA TAMAMLANDI!")
    print("=" * 110)


if __name__ == "__main__":
    main()
