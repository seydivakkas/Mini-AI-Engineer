"""
Day 305: Cross-Coder Sparse Autoencoder (SAE)
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

from src.cross_coder_motoru import (
    CrossCoderSAE,
    CrossCoderConfig,
    CrossCoderResult,
    SyntheticActivationGenerator,
    CrossCoderTrainer
)
from src.sae_profilleyici import SAEProfiler
from src.gorsellestirici import CrossCoderGorsellestirici


def main():
    print("=" * 80)
    print("👑 DAY 305: ÇAPRAZ-KODLAYICI SPARSE AUTOENCODER (CROSS-CODER SAE)")
    print("=" * 80)
    
    # -------------------------------------------------------------
    # ADIM 1: Çok Katmanlı Sentetik Nöron Aktivasyonları Üretimi
    # -------------------------------------------------------------
    print("\n📦 [ADIM 1] Çok Katmanlı Polisemantik Aktivasyon Havuzu Üretiliyor...")
    
    config = CrossCoderConfig(
        num_layers=3,
        d_model=32,
        dict_multiplier=8,  # d_sae = 256
        top_k=16,
        l1_coeff=0.002,
        lr=0.002,
        batch_size=64,
        epochs=40,
        seed=42
    )
    
    generator = SyntheticActivationGenerator(
        num_layers=config.num_layers,
        d_model=config.d_model,
        num_true_concepts=128,
        seed=config.seed
    )
    
    # Generate train and test batches
    train_batches = [generator.generate_batch(batch_size=config.batch_size, sparsity_p=0.06) for _ in range(30)]
    test_x = generator.generate_batch(batch_size=500, sparsity_p=0.06)
    
    print(f"  • Model Katman Sayısı (K): {config.num_layers} (L_0, L_1, L_2)")
    print(f"  • Katman Boyutu (d_model): {config.d_model} | SAE Sözlük Boyutu (d_sae): {config.d_sae}")
    print(f"  • Aşırı Tamamlanmışlık (Overcompleteness): {config.dict_multiplier}x")
    print(f"  • Hedef Top-K Seyreklik: {config.top_k} / {config.d_sae}")
    print(f"  • Eğitim Örnek Sayısı: {len(train_batches) * config.batch_size} | Test Örnek Sayısı: {test_x.size(0)}")
    
    # -------------------------------------------------------------
    # ADIM 2 & 3: Cross-Coder SAE Eğitimi
    # -------------------------------------------------------------
    print("\n⚡ [ADIM 2 & 3] Cross-Coder SAE Eğitimi Başlatılıyor (Top-K + Group L1)...")
    trainer = CrossCoderTrainer(config)
    
    for ep in range(1, config.epochs + 1):
        metrics = trainer.train_epoch(train_batches)
        if ep % 10 == 0 or ep == config.epochs:
            print(f"  • Epok [{ep:02d}/{config.epochs:02d}] - Kayıp: {metrics['total_loss']:.5f} | "
                  f"MSE: {metrics['recon_loss']:.5f} | L0: {metrics['l0_sparsity']:.1f}")
                  
    # -------------------------------------------------------------
    # ADIM 4: Mekanistik Değerlendirme, Profilleme ve Görselleştirme
    # -------------------------------------------------------------
    print("\n📊 [ADIM 4] Süperpozisyon Çözümleme Profillemesi ve Teşhis Panosu Üretiliyor...")
    result: CrossCoderResult = trainer.evaluate(test_x)
    profil_ozeti = SAEProfiler.profile_results(result)
    
    cikti_yolu = os.path.join(current_dir, "ciktilar", "cross_coder_paneli.png")
    CrossCoderGorsellestirici.ciz(result, cikti_yolu, profil_ozeti)
    
    print("\n" + "=" * 80)
    print("📋 CROSS-CODER SAE MEKANİSTİK YORUMLANABİLİRLİK RAPORU")
    print("=" * 80)
    for k, v in profil_ozeti.items():
        print(f"  • {k:<32}: {v}")
    print("=" * 80)
    print("✅ GÜN 305 BAŞARIYLA TAMAMLANDI!")


if __name__ == "__main__":
    main()
