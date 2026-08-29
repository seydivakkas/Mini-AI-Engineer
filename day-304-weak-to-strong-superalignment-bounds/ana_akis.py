"""
Day 304: Weak-to-Strong Superalignment with Confidence Bounds
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

from src.superalignment_motoru import (
    WeakToStrongTrainer,
    SuperalignmentConfig,
    SuperalignmentResult
)
from src.superalignment_profilleyici import SuperalignmentProfiler
from src.gorsellestirici import SuperalignmentGorsellestirici


def generate_synthetic_data(in_features=16, num_classes=4, n_samples=1600, seed=42):
    """
    Synthesizes a 4-cluster Gaussian mixture task where:
    - True classification boundary is clear in high-dimensional space
    - Weak supervisor only sees noisy/blurred features or noisy labels
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 4 distinct cluster centers in 16D space
    centers = np.array([
        [ 2.0,  2.0,  1.0, -1.0] + [0.0] * 12,
        [-2.0,  2.0, -1.0,  1.0] + [0.0] * 12,
        [-2.0, -2.0,  1.0,  1.0] + [0.0] * 12,
        [ 2.0, -2.0, -1.0, -1.0] + [0.0] * 12,
    ], dtype=np.float32)
    
    samples_per_class = n_samples // num_classes
    X_list = []
    y_list = []
    
    for c in range(num_classes):
        cluster_data = centers[c] + 0.9 * np.random.randn(samples_per_class, in_features).astype(np.float32)
        X_list.append(cluster_data)
        y_list.append(np.full(samples_per_class, c, dtype=np.int64))
        
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    # Shuffle
    perm = np.random.permutation(n_samples)
    X = X[perm]
    y = y[perm]
    
    # Dynamic split ratios: 25% train, 45% unlabeled, 15% calib, 15% test
    n_train = int(0.25 * n_samples)
    n_unlabeled = int(0.45 * n_samples)
    n_calib = int(0.15 * n_samples)
    
    idx1 = n_train
    idx2 = idx1 + n_unlabeled
    idx3 = idx2 + n_calib
    
    # Add noise to weak supervisor's training labels to simulate a weak human/proxy
    y_weak_train = y[:idx1].copy()
    noise_mask = np.random.rand(idx1) < 0.35  # 35% label noise for weak supervisor
    y_weak_train[noise_mask] = np.random.randint(0, num_classes, size=np.sum(noise_mask))
    
    splits = {
        "train": (torch.tensor(X[:idx1]), torch.tensor(y_weak_train)),
        "train_clean": (torch.tensor(X[:idx1]), torch.tensor(y[:idx1])),
        "unlabeled": (torch.tensor(X[idx1:idx2]), torch.tensor(y[idx1:idx2])),
        "calib": (torch.tensor(X[idx2:idx3]), torch.tensor(y[idx2:idx3])),
        "test": (torch.tensor(X[idx3:]), torch.tensor(y[idx3:]))
    }
    
    def make_loader(x_t, y_t, batch_size=32):
        loader = []
        n = x_t.size(0)
        for i in range(0, n, batch_size):
            loader.append((x_t[i:i+batch_size], y_t[i:i+batch_size]))
        return loader
        
    return {k: make_loader(v[0], v[1]) for k, v in splits.items()}


def main():
    print("=" * 80)
    print("👑 DAY 304: GÜVEN ARALIKLARIYLA ZAYIFTAN-GÜÇLÜYE SÜPER-HİZALAMA (WEAK-TO-STRONG)")
    print("=" * 80)
    
    # -------------------------------------------------------------
    # ADIM 1: Veri Bölümleme ve Konfigürasyon
    # -------------------------------------------------------------
    print("\n📦 [ADIM 1] Sentetik Veri Kümeleri ve Konfigürasyon Hazırlanıyor...")
    
    config = SuperalignmentConfig(
        in_features=16,
        num_classes=4,
        weak_epochs=20,
        strong_epochs=35,
        lr_weak=0.015,
        lr_strong=0.003,
        confidence_gate_tau=0.40,
        lambda_consistency=0.50,
        conformal_alpha=0.10,
        seed=42
    )
    
    data_splits = generate_synthetic_data(config.in_features, config.num_classes, n_samples=1200, seed=config.seed)
    
    print(f"  • Özellik Boyutu: {config.in_features} | Sınıf Sayısı: {config.num_classes}")
    print(f"  • Zayıf Denetim Eğitim Örnekleri: 300 | Zayıf Etiketleme Havuzu: 400")
    print(f"  • Kalibrasyon Kümesi: 200 | Test Kümesi: 300")
    print(f"  • Güven Eşik Değeri (tau_gate): {config.confidence_gate_tau}")
    print(f"  • Konformal Güven Garantisi (1-alpha): %{int((1.0 - config.conformal_alpha) * 100)}")
    
    # -------------------------------------------------------------
    # ADIM 2 & 3: Zayıftan-Güçlüye Süper-Hizalama Eğitimi
    # -------------------------------------------------------------
    print("\n⚡ [ADIM 2 & 3] Weak-to-Strong Süper-Hizalama Eğitimi ve Kalibrasyon Başlatılıyor...")
    trainer = WeakToStrongTrainer(config)
    result: SuperalignmentResult = trainer.run_superalignment(
        train_loader=data_splits["train"],
        unlabeled_loader=data_splits["unlabeled"],
        calib_loader=data_splits["calib"],
        test_loader=data_splits["test"],
        train_clean_loader=data_splits["train_clean"]
    )
    
    print(f"  ✓ Zayıf Denetçi (Weak Supervisor) Başarımı: %{result.weak_acc:.2f}")
    print(f"  ✓ Güçlü Tavan (Strong Ceiling) Başarımı: %{result.strong_ceiling_acc:.2f}")
    print(f"  ✓ Weak-to-Strong Model Başarımı: %{result.weak_to_strong_acc:.2f}")
    print(f"  ✓ Genelleme Farkı (Delta): +%{result.weak_to_strong_acc - result.weak_acc:.2f}")
    print(f"  ✓ Performance Gap Recovered (PGR): %{result.pgr_score:.2f}")
    
    # -------------------------------------------------------------
    # ADIM 4: Profilleme, Teşhis ve 6-Panelli Görselleştirme
    # -------------------------------------------------------------
    print("\n📊 [ADIM 4] Profilleme ve 6-Panelli Teşhis Panosu Üretiliyor...")
    profil_ozeti = SuperalignmentProfiler.profile_results(result)
    
    cikti_yolu = os.path.join(current_dir, "ciktilar", "superalignment_paneli.png")
    SuperalignmentGorsellestirici.ciz(result, cikti_yolu, profil_ozeti)
    
    print("\n" + "=" * 80)
    print("📋 WEAK-TO-STRONG SUPERALIGNMENT TEŞHİS RAPORU")
    print("=" * 80)
    for k, v in profil_ozeti.items():
        print(f"  • {k:<32}: {v}")
    print("=" * 80)
    print("✅ GÜN 304 BAŞARIYLA TAMAMLANDI!")


if __name__ == "__main__":
    main()
