"""
Day 326: Intracortical Spike Sorting & LFADS Latent Dynamics
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu modül; İntrakortikal Çoklu-Elektrot (MEA) sinyal simülasyonunu, PCA+GMM ile Spike Ayrıştırma (Spike Sorting)
algoritmasını ve LFADS (Latent Factor Analysis via Dynamical Systems) VAE modelini içerir.
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import scipy.signal as signal
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import torch
import torch.nn as nn
import torch.nn.functional as F


class MEAWaveformSimulator:
    """
    Çoklu Elektrot Dizilimi (Multielectrode Array - MEA / Utah Array) Ham Gerilim Sinyali Simülatörü.
    """
    def __init__(self, sampling_rate: int = 30000, duration_sec: float = 0.5):
        self.sampling_rate = sampling_rate
        self.duration_sec = duration_sec
        self.total_samples = int(sampling_rate * duration_sec)

    def uret_ham_elektrot_verisi(self, num_units: int = 3, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Ham gerilim sinyali (uV), spike zaman indeksleri ve gerçek birim (unit) etiketlerini döndürür.
        """
        np.random.seed(seed)
        t = np.linspace(0, self.duration_sec, self.total_samples)
        
        # Gürültülü Taban Sinyali (LFP + Termal Gürültü)
        voltage = np.random.randn(self.total_samples) * 5.0
        
        # 3 Farklı Birimin (Single-Unit) Tipik Aksiyon Potansiyel Dalgası
        t_wave = np.linspace(-1, 1, 48)
        wave_unit0 = -25.0 * np.exp(-t_wave**2 / 0.05) + 10.0 * np.exp(-(t_wave-0.3)**2 / 0.1)
        wave_unit1 = -40.0 * np.exp(-t_wave**2 / 0.02) + 15.0 * np.exp(-(t_wave-0.2)**2 / 0.05)
        wave_unit2 = -18.0 * np.exp(-t_wave**2 / 0.08) + 5.0 * np.exp(-(t_wave-0.4)**2 / 0.12)
        
        unit_waves = [wave_unit0, wave_unit1, wave_unit2]

        spike_indices = []
        true_labels = []

        # Spike ateşlemeleri
        num_spikes = 80
        for _ in range(num_spikes):
            unit_id = np.random.randint(0, num_units)
            idx = np.random.randint(50, self.total_samples - 50)
            
            w = unit_waves[unit_id]
            voltage[idx - 24 : idx + 24] += w
            spike_indices.append(idx)
            true_labels.append(unit_id)

        return voltage, np.array(spike_indices), np.array(true_labels)


class SpikeSorter:
    """
    Spike Algılama (Threshold Detection), PCA Öznitelik Çıkarımı ve GMM İntrakortikal Ayrıştırma.
    """
    @staticmethod
    def bant_geciren_filtre(voltage: np.ndarray, lowcut: float = 300.0, highcut: float = 3000.0, fs: int = 30000) -> np.ndarray:
        """
        Ham gerilimden yüksek frekanslı spike'ları ayırmak için 300Hz-3000Hz Butterworth filtre.
        """
        nyq = 0.5 * fs
        b, a = signal.butter(3, [lowcut / nyq, highcut / nyq], btype="band")
        return signal.filtfilt(b, a, voltage)

    @staticmethod
    def spike_tespit_et(voltage_filtered: np.ndarray, th_multiplier: float = 4.0) -> np.ndarray:
        """
        Negatif pik eşiği: V_th = - th_multiplier * std(V)
        """
        sigma_n = np.median(np.abs(voltage_filtered)) / 0.6745
        threshold = -th_multiplier * sigma_n

        peaks = []
        for i in range(25, len(voltage_filtered) - 25):
            if voltage_filtered[i] < threshold and voltage_filtered[i] < voltage_filtered[i-1] and voltage_filtered[i] < voltage_filtered[i+1]:
                peaks.append(i)
        return np.array(peaks, dtype=np.int32)

    @staticmethod
    def dalga_formu_cikar(voltage_filtered: np.ndarray, spike_indices: np.ndarray, window_size: int = 48) -> np.ndarray:
        half_w = window_size // 2
        waveforms = []
        for idx in spike_indices:
            if half_w <= idx < len(voltage_filtered) - half_w:
                waveforms.append(voltage_filtered[idx - half_w : idx + half_w])
        return np.array(waveforms, dtype=np.float32)

    @staticmethod
    def sort_spikes_pca_gmm(waveforms: np.ndarray, n_clusters: int = 3) -> Tuple[np.ndarray, np.ndarray, PCA, GaussianMixture]:
        """
        Waveform PCA İndirgemesi (2D) + Gaussian Mixture Model (GMM) Kümeleme.
        """
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(waveforms)

        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        cluster_labels = gmm.fit_predict(features_2d)

        return features_2d, cluster_labels, pca, gmm


class LFADSRecurrentGenerator(nn.Module):
    """
    LFADS (Latent Factor Analysis via Dynamical Systems) Variational Autoencoder (VAE) Modeli.
    
    Mimari:
        - Spike Counts Y in R^(B x T x N_neurons)
        - GRU Encoder -> Latent Space z_0 ~ N(mu, sigma)
        - GRU Generator -> Latent Factors g(t) in R^(B x T x d_latent)
        - Linear Readout -> Firing Rates lambda(t) = exp(W_rate * g(t) + b)
    """
    def __init__(
        self,
        num_neurons: int = 20,
        latent_dim: int = 16,
        hidden_dim: int = 64
    ):
        super().__init__()
        self.num_neurons = num_neurons
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Encoder GRU
        self.encoder_gru = nn.GRU(input_size=num_neurons, hidden_size=hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Generator GRU
        self.generator_gru = nn.GRU(input_size=latent_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc_factors = nn.Linear(hidden_dim, latent_dim)
        self.fc_rates = nn.Linear(latent_dim, num_neurons)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_spikes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Girdi: (Batch, Time, Neurons) -> Çıktı: log_rates, factors, mu, logvar
        """
        b_size, t_steps, _ = x_spikes.shape
        
        # Encoder
        _, h_n = self.encoder_gru(x_spikes)  # (1, B, Hidden)
        h_last = h_n.squeeze(0)
        
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        z_0 = self.reparameterize(mu, logvar)  # (B, Latent)

        # Generator
        gen_in = z_0.unsqueeze(1).repeat(1, t_steps, 1)  # (B, Time, Latent)
        gen_out, _ = self.generator_gru(gen_in)          # (B, Time, Hidden)

        factors = self.fc_factors(gen_out)               # (B, Time, Latent)
        log_rates = self.fc_rates(factors)               # (B, Time, Neurons)

        return log_rates, factors, mu, logvar

    @staticmethod
    def compute_poisson_loss(
        x_spikes: torch.Tensor,
        log_rates: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kl_weight: float = 1e-4
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Poisson Negative Log-Likelihood Loss + KL Divergence.
        L_pois = sum(exp(log_rate) - x_spikes * log_rate)
        """
        rates = torch.exp(log_rates)
        poisson_nll = torch.mean(rates - x_spikes * log_rates)
        
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = poisson_nll + kl_weight * kl_div
        return total_loss, poisson_nll, kl_div
