"""
Day 66: PyTorch -> ONNX Export, INT8 PTQ Kuantizasyon & ONNX Runtime Capstone Ana Akis Betigi
=============================================================================================
1. PyTorch UretimVisionNet modelini ilklendirir.
2. ONNX formatina dinamik eksenlerle (Dynamic Axes) disa aktarir ve grafigi dogrular.
3. Post-Training INT8 Kuantizasyon uygulayarak modeli %75 kucultur.
4. PyTorch FP32, ONNX FP32 ve ONNX INT8 arasinda sayisal esdegerlik ve gecikme benchmark'i kosar.
5. 6 Panelli yuksek cozunurluklu capstone teshis panosunu kaydeder.
"""

import os
import sys
import numpy as np
import torch

from src.model_mimari import UretimVisionNet
from src.onnx_aktarici import ONNXDonusturucu
from src.kuantizasyon_motoru import INT8Kuantizator
from src.karsilastirici_benchmark import ModelBenchmarkKarsilastirici
from src.gorsellestirici import CapstoneGorsellestirici


def main() -> None:
    print("=" * 95)
    print(">>> DAY 66: PYTORCH -> ONNX EXPORT, INT8 PTQ KUANTIZASYON & ONNX RUNTIME CAPSTONE")
    print("=" * 95)

    ciktilar_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(ciktilar_dizini, exist_ok=True)

    onnx_fp32_yolu = os.path.join(ciktilar_dizini, "uretim_model_fp32.onnx")
    onnx_int8_yolu = os.path.join(ciktilar_dizini, "uretim_model_int8.onnx")
    dashboard_yolu = os.path.join(ciktilar_dizini, "onnx_int8_karsilastirma_paneli.png")

    # 1. Adım: PyTorch Modelini Oluştur
    print("\n[+] 1. Adim: PyTorch UretimVisionNet Modeli Ilklendiriliyor...")
    model = UretimVisionNet(girdi_kanali=3, sinif_sayisi=10, taban_kanal=32)
    model.eval()

    toplam_param = sum(p.numel() for p in model.parameters())
    print(f"    - Model Mimarisi     : Conv2d + BatchNorm + ReLU + ResidualBlok + Linear Head")
    print(f"    - Toplam Parametre   : {toplam_param:,} Adet")

    # 2. Adım: ONNX FP32 İhracatı ve Doğrulama
    print("\n[+] 2. Adim: Model ONNX FP32 Formatina Aktariliyor (Opset 18)...")
    ornek_tensör = torch.randn(1, 3, 64, 64, dtype=torch.float32)
    aktarici = ONNXDonusturucu(opset_versiyonu=18)
    aktarici.disa_aktar(
        model=model,
        ornek_girdi=ornek_tensör,
        cikti_yolu=onnx_fp32_yolu,
        girdi_adi="girdi_gorsel",
        cikti_adi="cikis_lojiti"
    )

    ozet_fp32 = aktarici.model_ozeti_al(onnx_fp32_yolu)
    print(f"    - ONNX FP32 Dosyasi  : {onnx_fp32_yolu}")
    print(f"    - Opset Versiyonu    : {ozet_fp32['opset_versiyonu']}")
    print(f"    - Dugum (Node) Sayisi: {ozet_fp32['dugum_sayisi']}")
    print(f"    - Dosya Boyutu       : {ozet_fp32['boyut_mb']} MB")

    # 3. Adım: INT8 Post-Training Quantization (PTQ)
    print("\n[+] 3. Adim: INT8 Post-Training Kuantizasyon Uygulaniyor (Dynamic PTQ)...")
    kuantizator = INT8Kuantizator()
    kuant_sonuc = kuantizator.dinamik_kuantize_et(
        girdi_onnx_yolu=onnx_fp32_yolu,
        cikti_int8_yolu=onnx_int8_yolu
    )
    print(f"    - ONNX INT8 Dosyasi  : {onnx_int8_yolu}")
    print(f"    - INT8 Boyutu        : {kuant_sonuc['int8_boyut_mb']} MB")
    print(f"    - Sikistirma Orani   : {kuant_sonuc['sikistirma_orani']}x (Tasarruf: %{kuant_sonuc['tasarruf_yuzdesi']})")

    # 4. Adım: Sayısal Eşdeğerlik ve Benchmark Kıyaslaması
    print("\n[+] 4. Adim: Sayisal Esdegerlik (Parity) ve Latency/Throughput Benchmark'i Kosuluyor...")
    karsilastirici = ModelBenchmarkKarsilastirici(
        pytorch_model=model,
        onnx_fp32_yolu=onnx_fp32_yolu,
        onnx_int8_yolu=onnx_int8_yolu,
        is_parcacigi=4
    )

    test_girdisi = np.random.randn(1, 3, 64, 64).astype(np.float32)
    esdegerlik = karsilastirici.sayisal_esdegerlik_test_et(test_girdisi)

    print("\n" + "=" * 95)
    print(">>> 5. SAYISAL ESDEGERLIK (NUMERICAL PARITY) ANALIZI")
    print("=" * 95)
    print(f"* PyTorch vs ONNX FP32 Kosinus Benzerligi : % {esdegerlik['fp32_kosinus_benzerligi']*100:.5f}")
    print(f"* PyTorch vs ONNX FP32 Maksimum Mutlak Fark :   {esdegerlik['fp32_maks_fark']:.7f}")
    print(f"* PyTorch vs ONNX INT8 Kosinus Benzerligi : % {esdegerlik['int8_kosinus_benzerligi']*100:.5f}")
    print(f"* PyTorch vs ONNX INT8 Maksimum Mutlak Fark :   {esdegerlik['int8_maks_fark']:.7f}")

    benchmark_sonuclari = karsilastirici.tam_benchmark_kos(test_girdisi, tekrar=100)

    pt = benchmark_sonuclari["pytorch_fp32"]
    fp32 = benchmark_sonuclari["onnx_fp32"]
    int8 = benchmark_sonuclari["onnx_int8"]

    print("\n" + "=" * 95)
    print(">>> 6. PERFORMANS, GECIKME VE THROUGHPUT BENCHMARK TABLOSU")
    print("=" * 95)
    print(f"{'Model Varyanti':<18} | {'Gecikme (ms)':<14} | {'Throughput (FPS)':<18} | {'Boyut (MB)':<12} | {'Hizlanma (Speedup)':<18}")
    print("-" * 95)
    print(f"{'PyTorch FP32':<18} | {pt['gecikme_ms']:<14.2f} | {pt['fps']:<18.1f} | {pt['boyut_mb']:<12.2f} | {pt['speedup']:<18.2f}x (Referans)")
    print(f"{'ONNX FP32':<18} | {fp32['gecikme_ms']:<14.2f} | {fp32['fps']:<18.1f} | {fp32['boyut_mb']:<12.2f} | {fp32['speedup']:<18.2f}x")
    print(f"{'ONNX INT8':<18} | {int8['gecikme_ms']:<14.2f} | {int8['fps']:<18.1f} | {int8['boyut_mb']:<12.2f} | {int8['speedup']:<18.2f}x")

    # 5. Adım: Teşhis Panosunu Kaydet
    print("\n[+] 7. Adim: 6 Panelli Capstone Teşhis Panosu Olusturuluyor...")
    grafik_yolu = CapstoneGorsellestirici.panoyu_ciz_ve_kaydet(
        benchmark_sonuclari=benchmark_sonuclari,
        esdegerlik_sonuclari=esdegerlik,
        cikti_yolu=dashboard_yolu
    )
    print(f"[+] 6 Panelli Capstone Panosu Kaydedildi: {grafik_yolu}")
    print("=" * 95)
    print("DAY 66: ONNX INT8 PRODUCTION CAPSTONE BASARIYLA TAMAMLANDI!")
    print("=" * 95)


if __name__ == "__main__":
    main()
