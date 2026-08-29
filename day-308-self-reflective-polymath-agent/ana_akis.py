"""
Day 308: Self-Reflective Polymath Agent: Recursive Skill Synthesis & Memory Graphs
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

from src.polymath_motoru import (
    PolymathConfig,
    PolymathResult,
    PolymathAgent
)
from src.polymath_profilleyici import PolymathProfiler
from src.gorsellestirici import PolymathGorsellestirici


def main():
    print("=" * 80)
    print("👑 DAY 308: ÇOK ALANLI (POLYMATH) AJAN: ÖZYİNELEMELİ BECERİ SENTEZİ VE BİRLEŞİMİ")
    print("=" * 80)
    
    # -------------------------------------------------------------
    # ADIM 1: Konfigürasyon ve Ajan Kurulumu
    # -------------------------------------------------------------
    print("\n📦 [ADIM 1] Polymath Ajanı, İzole Sandbox ve Vektör Hafıza Grafiği Kuruluyor...")
    
    config = PolymathConfig(
        embedding_dim=32,
        max_reflection_iters=3,
        retrieval_similarity_threshold=0.60,
        num_benchmark_tasks=50,
        seed=42
    )
    
    agent = PolymathAgent(config)
    
    print(f"  • Vektör Gömme Boyutu: {config.embedding_dim}D")
    print(f"  • Maksimum Öz-Yansıma (Reflection) İterasyonu: {config.max_reflection_iters}")
    print(f"  • Hafıza Getirme Eşiği (Similarity Threshold): {config.retrieval_similarity_threshold}")
    print(f"  • Yürütülecek Çok-Disiplinli Görev Sayısı: {config.num_benchmark_tasks}")
    
    # -------------------------------------------------------------
    # ADIM 2 & 3: Beceri Sentezi ve Öz-Yansıma Simülasyonu
    # -------------------------------------------------------------
    print("\n⚡ [ADIM 2 & 3] Çapraz-Alan Görevleri, Dinamik Kod Sentezi ve Öz-Yansıma Yürütülüyor...")
    result: PolymathResult = agent.run_benchmark()
    
    print(f"  ✓ Beceri Sentezi Başarı Oranı: %{result.skill_synthesis_success_rate_pct:.2f}")
    print(f"  ✓ Çapraz-Alan Hafıza Kullanım Oranı: %{result.cross_domain_reuse_efficiency_pct:.2f}")
    print(f"  ✓ Öz-Yansıma Hata Telafisi: %{result.reflection_error_recovery_rate_pct:.2f}")
    print(f"  ✓ Toplam Sentezlenen Yeni Beceri: {result.total_skills_synthesized}")
    print(f"  ✓ Hafıza Grafiği Yoğunluğu: {result.memory_graph_density:.4f}")
    print(f"  ✓ Ortalama İcra Gecikmesi: {result.avg_execution_latency_ms:.2f} ms")
    
    # -------------------------------------------------------------
    # ADIM 4: Profilleme, Teşhis ve 6-Panelli Görselleştirme
    # -------------------------------------------------------------
    print("\n📊 [ADIM 4] Profilleme ve 6-Panelli Teşhis Panosu Üretiliyor...")
    profil_ozeti = PolymathProfiler.profile_results(result)
    
    cikti_yolu = os.path.join(current_dir, "ciktilar", "polymath_paneli.png")
    PolymathGorsellestirici.ciz(result, cikti_yolu, profil_ozeti)
    
    print("\n" + "=" * 80)
    print("📋 ÇOK ALANLI POLYMATH AJAN TEŞHİS RAPORU")
    print("=" * 80)
    for k, v in profil_ozeti.items():
        print(f"  • {k:<34}: {v}")
    print("=" * 80)
    print("✅ GÜN 308 BAŞARIYLA TAMAMLANDI!")


if __name__ == "__main__":
    main()
