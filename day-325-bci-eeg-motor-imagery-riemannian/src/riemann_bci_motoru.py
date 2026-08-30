"""
Day 325: Brain-Computer Interface (BCI) & Riemannian Geometry on EEG
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; EEG Motor İmgelemi (Motor Imagery) sinyal simülasyonunu,
Örnek Kovaryans Matrisi (SCM) hesaplamasını, SPD Manifoldu üzerinde Affine-Invariant
Riemann Mesafesini, Frechet Ortalama hesabını ve Teğet Uzayı (Tangent Space) projeksiyonunu içerir.
"""

from typing import Tuple, Dict, Any, List, Optional
import math
import numpy as np
import scipy.linalg as la
from sklearn.linear_model import LogisticRegression


class EEGMotorImageryGenerator:
    """
    Çok Kanallı EEG Motor İmgelemi (Motor Imagery) Sinyal Simülatörü.
    Sınıflar: 0 (Sol El), 1 (Sağ El), 2 (Ayaklar)
    """
    def __init__(self, num_channels: int = 8, sampling_rate: int = 250, trial_duration_sec: float = 2.0):
        self.num_channels = num_channels
        self.sampling_rate = sampling_rate
        self.trial_duration_sec = trial_duration_sec
        self.num_samples = int(sampling_rate * trial_duration_sec)

    def uret_eeg_deneyleri(
        self, num_trials_per_class: int = 30, seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Her sınıf için (Num_Trials, Channels, Samples) boyutlu EEG sinyali üretir.
        """
        np.random.seed(seed)
        num_classes = 3
        total_trials = num_trials_per_class * num_classes

        x_eeg = np.zeros((total_trials, self.num_channels, self.num_samples), dtype=np.float32)
        y = np.zeros(total_trials, dtype=np.int32)

        t = np.linspace(0, self.trial_duration_sec, self.num_samples)
        
        # 10 Hz Mu-bandı frekansı
        mu_freq = 10.0

        for i in range(total_trials):
            label = i % num_classes
            y[i] = label
            
            # Rastgele Arka Plan Gürültüsü
            noise = np.random.randn(self.num_channels, self.num_samples) * 0.8
            
            # Sınıfa Özel Motor Korteks Desenkronizasyonu (ERD/ERS)
            cov_pattern = np.eye(self.num_channels)
            c_c3 = min(1, self.num_channels - 1)
            c_cz = min(3, self.num_channels - 1)
            c_c4 = min(5, self.num_channels - 1)

            if label == 0:  # Sol El -> Sağ Motor Korteks (C4) Genliği Düşer (ERD)
                cov_pattern[c_c4, c_c4] = 0.25
                cov_pattern[c_c3, c_c3] = 1.8
            elif label == 1:  # Sağ El -> Sol Motor Korteks (C3) Genliği Düşer (ERD)
                cov_pattern[c_c3, c_c3] = 0.25
                cov_pattern[c_c4, c_c4] = 1.8
            else:  # Ayaklar -> Santral Korteks (Cz) Genliği Düşer (ERD)
                cov_pattern[c_cz, c_cz] = 0.25
                cov_pattern[0, 0] = 1.5

            # Kovaryans Renklendirmesi
            cholesky = la.cholesky(cov_pattern, lower=True)
            signal = cholesky @ noise
            
            # Mu-bandı salınımı ekle
            for c in range(self.num_channels):
                signal[c] += np.sin(2 * np.pi * mu_freq * t) * cov_pattern[c, c]

            x_eeg[i] = signal

        return x_eeg, y


class CovarianceEstimator:
    """
    Düzenlileştirilmiş Örnek Kovaryans Matrisi (Sample Covariance Matrix - SCM) Hesaplayıcı.
    Kovaryans matrislerinin kesin Simetrik Pozitif Tanımlı (SPD in S_++^C) kalmasını sağlar.
    """
    @staticmethod
    def hesapla_scm(x_trial: np.ndarray, reg_alpha: float = 1e-4) -> np.ndarray:
        """
        Girdi: (Channels, Samples) -> Çıktı: (Channels, Channels) SCM Matrisi
        """
        channels, samples = x_trial.shape
        # Ortalama Çıkarma
        x_centered = x_trial - np.mean(x_trial, axis=1, keepdims=True)
        scm = (x_centered @ x_centered.T) / (samples - 1.0)
        
        # Sütun İzolasyonu ve Düzenlileştirme (Shrinkage Regularization)
        scm_reg = scm + reg_alpha * np.eye(channels)
        return scm_reg


class RiemannianGeometryEngine:
    """
    SPD (Symmetric Positive-Definite) Manifold Üzerinde Riemann Geometrisi Motoru.
    """
    @staticmethod
    def riemannian_distance(sigma1: np.ndarray, sigma2: np.ndarray) -> float:
        """
        Affine-Invariant Riemannian Metric (AIRM) Mesafesi:
            delta_R(Sigma1, Sigma2) = || logm(Sigma1^-1/2 * Sigma2 * Sigma1^-1/2) ||_F
        """
        # Eigenvalues of Sigma1^-1 * Sigma2
        eigvals = la.eigvalsh(sigma2, sigma1)
        eigvals = np.maximum(eigvals, 1e-9)
        log_eigvals = np.log(eigvals)
        return float(np.sqrt(np.sum(log_eigvals ** 2)))

    @staticmethod
    def frechet_mean(sigmas: List[np.ndarray], max_iter: int = 15, tol: float = 1e-6) -> np.ndarray:
        """
        Karcher / Frechet Ortalama (SPD Manifold Üzerindeki Kütle Merkezi):
            Sigma_bar = argmin_{Sigma} sum_k delta_R^2(Sigma, Sigma_k)
        """
        k = len(sigmas)
        # Öklidsel Başlangıç
        mean_sigma = np.mean(sigmas, axis=0)

        for _ in range(max_iter):
            sqrt_mean = la.sqrtm(mean_sigma)
            inv_sqrt_mean = la.inv(sqrt_mean)

            tangent_sum = np.zeros_like(mean_sigma)
            for s in sigmas:
                m_rot = inv_sqrt_mean @ s @ inv_sqrt_mean
                tangent_sum += la.logm(m_rot)

            avg_tangent = tangent_sum / k
            if np.linalg.norm(avg_tangent) < tol:
                break

            mean_sigma = sqrt_mean @ la.expm(avg_tangent) @ sqrt_mean

        return mean_sigma

    @staticmethod
    def tangent_space_projection(sigma: np.ndarray, mean_sigma: np.ndarray) -> np.ndarray:
        """
        SPD Matrisini Ortalama Noktasında Öklid Teğet Uzayına Projekte Eder ve Vektörleştirir.
        Vektör Boyutu: C * (C + 1) / 2
        """
        sqrt_mean = la.sqrtm(mean_sigma)
        inv_sqrt_mean = la.inv(sqrt_mean)

        m_rot = inv_sqrt_mean @ sigma @ inv_sqrt_mean
        tangent_matrix = la.logm(m_rot)

        channels = sigma.shape[0]
        # Üst üçgensel bileşenleri ve off-diagonal için sqrt(2) çarpanını al
        vector_components = []
        for i in range(channels):
            vector_components.append(tangent_matrix[i, i])
            for j in range(i + 1, channels):
                vector_components.append(np.sqrt(2.0) * tangent_matrix[i, j])

        return np.array(vector_components, dtype=np.float32)


class RiemannianMDMClassifier:
    """
    Riemannian Minimum Distance to Mean (MDM) Sınıflandırıcısı.
    Her sınıfın Frechet ortalamasını hesaplar ve örneği en yakın Frechet ortalamasına atar.
    """
    def __init__(self):
        self.class_means: Dict[int, np.ndarray] = {}

    def fit(self, sigmas: List[np.ndarray], y: np.ndarray):
        classes = np.unique(y)
        for c in classes:
            c_sigmas = [sigmas[i] for i in range(len(y)) if y[i] == c]
            self.class_means[c] = RiemannianGeometryEngine.frechet_mean(c_sigmas)

    def predict(self, sigmas: List[np.ndarray]) -> np.ndarray:
        preds = []
        for s in sigmas:
            distances = {c: RiemannianGeometryEngine.riemannian_distance(s, mean) for c, mean in self.class_means.items()}
            best_class = min(distances, key=distances.get)
            preds.append(best_class)
        return np.array(preds)


class TangentSpaceClassifier:
    """
    Teğet Uzayı Projeksiyonu + Lojistik Regresyon Sınıflandırıcısı.
    """
    def __init__(self):
        self.global_mean: Optional[np.ndarray] = None
        self.clf = LogisticRegression(max_iter=500)

    def fit(self, sigmas: List[np.ndarray], y: np.ndarray):
        self.global_mean = RiemannianGeometryEngine.frechet_mean(sigmas)
        tangent_features = [RiemannianGeometryEngine.tangent_space_projection(s, self.global_mean) for s in sigmas]
        self.clf.fit(tangent_features, y)

    def predict(self, sigmas: List[np.ndarray]) -> np.ndarray:
        tangent_features = [RiemannianGeometryEngine.tangent_space_projection(s, self.global_mean) for s in sigmas]
        return self.clf.predict(tangent_features)
