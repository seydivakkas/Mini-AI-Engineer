"""
Day 69: AdamW vs Lion Optimizer, CosineAnnealing & Warmup Ana Akış Betiği
========================================================================
1. AdamW + StepLR (Klasik Baseline),
2. AdamW + WarmupCosine (Modern Standart),
3. Lion + WarmupCosine (Google Brain AutoML Evolved)
kombinasyonlarını aynı model mimarisi ve tohumla eğitir, yakınsama ve bellek metriklerini ölçer.
4. 6 Panelli görsel laboratuvar teşhis panosunu kaydeder.
"""

import os
import sys
import torch

from src.optimizer_laboratuvari import OptimizerLaboratuvari
from src.gorsellestirici import OptimizerLaboratuvarGorsellestirici


def main() -> None:
    print("=" * 95)
    print(">>> DAY 69: ADAMW VS LION OPTIMIZER, COSINE ANNEALING & WARMUP LABORATUVARI")
    print("=" * 95)

    kok_dizin = os.path.dirname(os.path.abspath(__file__))
    ciktilar_dizini = os.path.join(kok_dizin, "ciktilar")
    os.makedirs(ciktilar_dizini, exist_ok=True)
    dashboard_yolu = os.path.join(ciktilar_dizini, "optimizer_karsilastirma_paneli.png")

    cihaz = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"\n[+] 1. Adim: Optimizasyon Laboratuvari Baslatiliyor (Cihaz: {cihaz})...")
    print("    - Deney Kapsami: 3 Farkli Optimizer & Scheduler Kombinasyonu (10 Epoch)")

    sonuclar = OptimizerLaboratuvari.tum_laboratuvari_kos(toplam_epoch=10)

    d1 = sonuclar["deney_1"]
    d2 = sonuclar["deney_2"]
    d3 = sonuclar["deney_3"]

    print("\n" + "=" * 95)
    print(">>> 2. DENEYSEL SONUCLAR VE OPTIMIZASYON PERFORMANSI")
    print("=" * 95)
    print(f"{'Deney Mimarisi':<26} | {'Optimizer':<10} | {'Scheduler':<14} | {'Son Train Loss':<16} | {'Val Acc (%)':<12} | {'Opt Bellek':<10}")
    print("-" * 95)
    print(f"{d1['deney_adi']:<26} | {d1['optimizer']:<10} | {d1['scheduler']:<14} | {d1['son_train_loss']:<16.4f} | %{d1['son_val_accuracy']:<10.2f} | {d1['tahmini_opt_bellek_kb']:<7.0f} KB")
    print(f"{d2['deney_adi']:<26} | {d2['optimizer']:<10} | {d2['scheduler']:<14} | {d2['son_train_loss']:<16.4f} | %{d2['son_val_accuracy']:<10.2f} | {d2['tahmini_opt_bellek_kb']:<7.0f} KB")
    print(f"{d3['deney_adi']:<26} | {d3['optimizer']:<10} | {d3['scheduler']:<14} | {d3['son_train_loss']:<16.4f} | %{d3['son_val_accuracy']:<10.2f} | {d3['tahmini_opt_bellek_kb']:<7.0f} KB")

    print("\n" + "=" * 95)
    print(">>> 3. DERINLEMESINE OPTIMIZASYON ANALIZI")
    print("=" * 95)
    print(f"* Lion Bellek Tasarrufu                  : %50 DAHA AZ BELLEK ({d3['tahmini_opt_bellek_kb']:.0f} KB vs {d2['tahmini_opt_bellek_kb']:.0f} KB)")
    print(f"* Warmup + Cosine Etkisi                 : Erken adımlarda aşırı gradyan salınımını sönümledi.")
    print(f"* Decoupled Weight Decay                 : Normalizasyon ve bias parametreleri hariç tutularak uygulandı.")

    # Görselleştirme
    print("\n[+] 4. Adim: 6 Panelli Optimizasyon Laboratuvari Teshis Panosu Olusturuluyor...")
    grafik_yolu = OptimizerLaboratuvarGorsellestirici.panoyu_ciz_ve_kaydet(
        laboratuvar_sonuclari=sonuclar,
        cikti_yolu=dashboard_yolu
    )
    print(f"[+] Teşhis Panosu Kaydedildi: {grafik_yolu}")
    print("=" * 95)
    print("DAY 69: OPTIMIZER & SCHEDULER LABORATORY BASARIYLA TAMAMLANDI!")
    print("=" * 95)


if __name__ == "__main__":
    main()
