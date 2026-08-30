"""
Day 325: Brain-Computer Interface (BCI) & Riemannian Geometry on EEG
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu betik; EEG Motor İmgelemi sinyallerinden SCM kovaryans matrisi çıkarımını,
SPD manifold üzerinde Riemann mesafesi ve teğet uzayı sınıflandırmasını simüle eder.
"""

import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import numpy as np
from sklearn.model_selection import train_test_split
from src.riemann_bci_motoru import (
    EEGMotorImageryGenerator,
    CovarianceEstimator,
    RiemannianGeometryEngine,
    RiemannianMDMClassifier,
    TangentSpaceClassifier,
)
from src.riemann_gorsellestirici import RiemannGorsellestirici
from src.riemann_profilleyici import RiemannProfilleyici


def main():
    print("=" * 75, flush=True)
    print("🧠 DAY 325: Beyin-Bilgisayar Arayüzü (BCI) & EEG Riemann Geometrisi", flush=True)
    print("=" * 75, flush=True)

    # Hiperparametreler
    num_channels = 8
    sampling_rate = 250
    trials_per_class = 35

    print(f"📌 EEG Kurulumu: {num_channels} Kanal (C3, Cz, C4 motor korteks), {sampling_rate} Hz Örnekleme", flush=True)
    print(f"📌 Motor İmgelemi Sınıfları: Sol El (0), Sağ El (1), Ayaklar (2)", flush=True)

    # 1. EEG Sinyal Simülasyonu
    print("\n⚡ Çok Kanallı EEG Motor İmgelemi Deneyleri Üretiliyor...", flush=True)
    generator = EEGMotorImageryGenerator(num_channels=num_channels, sampling_rate=sampling_rate)
    x_eeg, y = generator.uret_eeg_deneyleri(num_trials_per_class=trials_per_class, seed=42)
    print(f"✅ Üretim Tamamlandı! Toplam Deney: {x_eeg.shape[0]} Epoch | Boyut: {x_eeg.shape}", flush=True)

    # 2. SCM Kovaryans Matrisi Hesaplama
    print("\n📐 Örnek Kovaryans Matrisleri (SCM in S_++^C) Hesaplanıyor...", flush=True)
    start_time = time.time()
    sigmas = [CovarianceEstimator.hesapla_scm(x_eeg[i]) for i in range(len(y))]
    elapsed_time = time.time() - start_time
    print(f"✅ SCM Çıkarımı Tamamlandı! Süre: {elapsed_time*1000.0:.2f} ms", flush=True)

    # Train / Test Bölünmesi
    sigmas_train, sigmas_test, y_train, y_test = train_test_split(sigmas, y, test_size=0.3, random_state=42, stratify=y)

    # 3. Riemannian MDM Sınıflandırma
    print("\n🎯 1) Riemannian Minimum Distance to Mean (MDM) Eğitiliyor...", flush=True)
    mdm_clf = RiemannianMDMClassifier()
    mdm_clf.fit(sigmas_train, y_train)
    mdm_preds = mdm_clf.predict(sigmas_test)
    mdm_acc = float(np.mean(mdm_preds == y_test) * 100.0)
    print(f"  • Riemannian MDM Test Doğruluğu: %{mdm_acc:.2f}", flush=True)

    # 4. Tangent Space + SVM Sınıflandırma
    print("\n🎯 2) Riemann Teğet Uzayı (Tangent Space Projection) + Lojistik Regresyon Eğitiliyor...", flush=True)
    tangent_clf = TangentSpaceClassifier()
    tangent_clf.fit(sigmas_train, y_train)
    tangent_preds = tangent_clf.predict(sigmas_test)
    tangent_acc = float(np.mean(tangent_preds == y_test) * 100.0)
    print(f"  • Tangent Space Classifier Test Doğruluğu: %{tangent_acc:.2f}", flush=True)

    # 5. Teğet Uzayı Vektörleştirme ve Mesafe Matrisi
    global_mean = RiemannianGeometryEngine.frechet_mean(sigmas)
    tangent_features = np.array([RiemannianGeometryEngine.tangent_space_projection(s, global_mean) for s in sigmas])

    num_vis = 20
    riemann_dist_matrix = np.zeros((num_vis, num_vis))
    for i in range(num_vis):
        for j in range(num_vis):
            riemann_dist_matrix[i, j] = RiemannianGeometryEngine.riemannian_distance(sigmas[i], sigmas[j])

    # 6. Profilleme ve Teşhis Grafiği
    profiler_metrics = RiemannProfilleyici.profille(
        num_channels=num_channels,
        mdm_acc=mdm_acc,
        tangent_svm_acc=tangent_acc,
        extraction_time_ms=elapsed_time * 1000.0 / len(y)
    )

    print("\n📊 Riemannian BCI Profilleme Metrikleri:", flush=True)
    print(f"  • Teğet Uzayı Vektör Boyutu:      {profiler_metrics['tangent_dim']} Boyut (C*(C+1)/2)", flush=True)
    print(f"  • Epoch Başına Kovaryans Süresi: {profiler_metrics['extraction_time_ms']:.3f} ms", flush=True)
    print(f"  • Manifold Uyum Skoru:           %{profiler_metrics['manifold_score']:.1f}", flush=True)

    sample_scms = {c: RiemannianGeometryEngine.frechet_mean([sigmas[i] for i in range(len(y)) if y[i] == c]) for c in range(3)}

    gorsellestirici = RiemannGorsellestirici()
    cikti_yolu = gorsellestirici.teshis_panelini_ciz(
        sample_eeg=x_eeg[0],
        sample_scms=sample_scms,
        tangent_features=tangent_features,
        labels=y,
        riemann_dist_matrix=riemann_dist_matrix,
        profiler_metrics=profiler_metrics
    )
    print(f"\n🖼️ 6-Panelli BCI Teşhis Grafiği Başarıyla Kaydedildi: [riemann_bci_paneli.png](file:///{os.path.abspath(cikti_yolu)})", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
