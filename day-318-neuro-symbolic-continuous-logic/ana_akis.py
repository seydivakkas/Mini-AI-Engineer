"""
Day 318: Neuro-Symbolic Continuous Logic & Differentiable Theorem Prover
Ana Akış Çalıştırma Scripti
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas) - All Rights Reserved.
"""

import sys
import os
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure local imports work reliably
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.noro_sembolik_mantik import (
    ContinuousLogicConfig,
    TNormType,
    SoftTheoremProver,
    NeuroSymbolicResult
)
from src.noro_sembolik_profilleyici import NeuroSymbolicProfiler
from src.gorsellestirici import NeuroSymbolicGorsellestirici


def main():
    print("=" * 80)
    print("🧠📐 DAY 318: NÖRO-SEMBOLİK SÜREKLİ MANTIK VE BULANIK TEOREM DOĞRULAMA")
    print("=" * 80)
    
    # -------------------------------------------------------------
    # ADIM 1: Konfigürasyon ve Nöro-Sembolik Bilgi Tabanı Kurulumu
    # -------------------------------------------------------------
    print("\n📦 [ADIM 1] Nöro-Sembolik Bilgi Tabanı ve Sürekli T-Norm Operatörleri Kuruluyor...")
    
    config = ContinuousLogicConfig(
        t_norm=TNormType.LUKASIEWICZ,
        embedding_dim=16,
        temperature=0.5,
        num_entities=6,
        num_steps=50,
        learning_rate=0.05,
        logic_loss_weight=1.0,
        seed=42
    )
    
    prover = SoftTheoremProver(config)
    
    print(f"  • Seçilen T-Norm Çerçevesi: {config.t_norm.value.upper()}")
    print(f"  • Varlık (Entity) Sayısı: {config.num_entities} | Vektör Boyutu (D): {config.embedding_dim}")
    print(f"  • Taban Gerçeklik Olguları (Ground Facts): Parent(0,1), Parent(1,2), Parent(2,3), Parent(3,4)")
    print(f"  • Doğrulanacak Aksiyomlar: R1 (Taban), R2 (Geçişlilik: X->Y ^ Y->Z => X->Z), R3 (Asimetri)")
    
    # -------------------------------------------------------------
    # ADIM 2 & 3: Nöro-Sembolik Optimizasyon ve Geriye Doğru Zincirleme Teorem İspatı
    # -------------------------------------------------------------
    print("\n⚡ [ADIM 2 & 3] Türetilebilir Geriye Zincirleme (Backward Chaining) ve Kural Eğitimi Yürütülüyor...")
    result: NeuroSymbolicResult = prover.train_and_prove()
    
    print(f"  ✓ Teorem Kanıtlama Başarısı: %{result.theorem_proof_accuracy_pct:.2f}")
    print(f"  ✓ Son Toplam Kayıp: {result.total_loss:.4f} | Mantıksal İhlal Kaybı: {result.final_logical_violation_loss:.4f}")
    
    print("\n  🔍 Aksiyom Sağlanma Oranları (Fuzzy Rule Truth Values):")
    for k, v in result.rule_satisfaction_rates.items():
        print(f"    • {k:<25}: %{v*100:.2f} (Doğruluk: {v:.4f})")
        
    print("\n  📜 Türetilen Teorem Sorguları ve İspat İzi (Proof Trace):")
    for q in result.proven_queries:
        status_symbol = "✓" if q["is_proven"] else "✗"
        print(f"    [{status_symbol}] {q['query']:<18} -> Doğruluk: {q['truth_value']:.4f} | İspat: {q['proof_trace']}")
        
    # -------------------------------------------------------------
    # ADIM 4: Profilleme, Teşhis ve 6-Panelli Görselleştirme
    # -------------------------------------------------------------
    print("\n📊 [ADIM 4] Profilleme ve 6-Panelli Teşhis Panosu Üretiliyor...")
    profil_ozeti = NeuroSymbolicProfiler.profile_results(result)
    
    cikti_yolu = os.path.join(current_dir, "ciktilar", "noro_sembolik_paneli.png")
    NeuroSymbolicGorsellestirici.ciz(result, cikti_yolu, profil_ozeti)
    
    print("\n" + "=" * 80)
    print("📋 NÖRO-SEMBOLİK SÜREKLİ MANTIK VE TEOREM DOĞRULAMA TEŞHİS RAPORU")
    print("=" * 80)
    for k, v in profil_ozeti.items():
        print(f"  • {k:<38}: {v}")
    print("=" * 80)
    print("✅ GÜN 318 BAŞARIYLA TAMAMLANDI!")


if __name__ == "__main__":
    main()
