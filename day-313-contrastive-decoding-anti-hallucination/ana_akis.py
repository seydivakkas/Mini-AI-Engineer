"""
Day 313: Contrastive Decoding Anti-Hallucination
Ana Akış Çalıştırma Scripti
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

import sys
import os
import torch
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure local imports work reliably
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.karsitsal_kod_cozucu import (
    ContrastiveDecodingConfig,
    ContrastiveDecodingResult,
    ContrastiveDecoderEngine
)
from src.karsitsal_kod_profilleyici import ContrastiveDecodingProfiler
from src.gorsellestirici import ContrastiveDecodingGorsellestirici


def main():
    print("=" * 80)
    print("🛡️ DAY 313: KARŞITSAL KOD ÇÖZME (CONTRASTIVE DECODING) İLE HALÜSİNASYON BASKILAMA")
    print("=" * 80)
    
    # -------------------------------------------------------------
    # ADIM 1: Konfigürasyon ve Kod Çözücü Motor Kurulumu
    # -------------------------------------------------------------
    print("\n📦 [ADIM 1] Uzman ve Amatör Logit Modelleri Kuruluyor...")
    
    config = ContrastiveDecodingConfig(
        vocab_size=256,
        alpha=1.2,
        beta=0.1,
        temperature=0.8,
        num_prompts=50,
        generation_length=25,
        seed=42
    )
    
    engine = ContrastiveDecoderEngine(config)
    
    print(f"  • Kelime Dağarcığı Boyutu (Vocab Size): {config.vocab_size}")
    print(f"  • Amatör Ceza Ağırlığı (Alpha): {config.alpha}")
    print(f"  • Uyarlanabilir Başlık Eşiği (Beta): {config.beta}")
    print(f"  • Sıcaklık Katsayısı (Temperature): {config.temperature}")
    print(f"  • Değerlendirilecek İstem Sayısı: {config.num_prompts}")
    
    # -------------------------------------------------------------
    # ADIM 2 & 3: Karşılaştırmalı Çıkarım ve Benchmark
    # -------------------------------------------------------------
    print("\n⚡ [ADIM 2 & 3] Standart ve Karşıtsal Kod Çözme Kıyaslaması Yapılıyor...")
    result: ContrastiveDecodingResult = engine.run_benchmark()
    
    print(f"  ✓ Standart Çıkarım Doğruluğu: %{result.standard_factuality_pct:.2f}")
    print(f"  ✓ Contrastive Decoding Doğruluğu: %{result.contrastive_factuality_pct:.2f}")
    print(f"  ✓ Halüsinasyon Azaltma Oranı: %{result.hallucination_reduction_pct:.2f}")
    print(f"  ✓ Standart ECE: {result.standard_ece:.4f} -> CD ECE: {result.contrastive_ece:.4f}")
    
    print("\n  🔍 Örnek İstem Başarı Karşılaştırmaları:")
    for s in result.sample_generations:
        print(f"    • Prompt #{s['prompt_id']} -> Standart: %{s['std_accuracy']:.1f} | Contrastive: %{s['cd_accuracy']:.1f}")
        
    # -------------------------------------------------------------
    # ADIM 4: Profilleme, Teşhis ve 6-Panelli Görselleştirme
    # -------------------------------------------------------------
    print("\n📊 [ADIM 4] Profilleme ve 6-Panelli Teşhis Panosu Üretiliyor...")
    profil_ozeti = ContrastiveDecodingProfiler.profile_results(result)
    
    cikti_yolu = os.path.join(current_dir, "ciktilar", "karsitsal_kod_paneli.png")
    ContrastiveDecodingGorsellestirici.ciz(result, cikti_yolu, profil_ozeti)
    
    print("\n" + "=" * 80)
    print("📋 KARŞITSAL KOD ÇÖZME HALÜSİNASYON TEŞHİS RAPORU")
    print("=" * 80)
    for k, v in profil_ozeti.items():
        print(f"  • {k:<34}: {v}")
    print("=" * 80)
    print("✅ GÜN 313 BAŞARIYLA TAMAMLANDI!")


if __name__ == "__main__":
    main()
