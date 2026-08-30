"""
Day 326: Intracortical Spike Sorting & LFADS Latent Dynamics
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Birim Test Paketi (PyTest Suite)
"""

import sys
import os
import pytest
import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.lfads_spike_motoru import (
    MEAWaveformSimulator,
    SpikeSorter,
    LFADSRecurrentGenerator,
)


def test_mea_waveform_simulator():
    """
    MEA Ham Gerilim Simülatörü Boyut Doğrulaması.
    """
    sim = MEAWaveformSimulator(sampling_rate=30000, duration_sec=0.1)
    voltage, spike_indices, true_labels = sim.uret_ham_elektrot_verisi(num_units=3, seed=42)
    
    assert voltage.shape == (3000,)
    assert len(spike_indices) > 0
    assert len(spike_indices) == len(true_labels)


def test_spike_sorter_threshold_detection():
    """
    Butterworth Filtresi ve Eşik Tespiti Testi.
    """
    sim = MEAWaveformSimulator(sampling_rate=30000, duration_sec=0.2)
    voltage, _, _ = sim.uret_ham_elektrot_verisi(num_units=3, seed=42)
    
    v_filt = SpikeSorter.bant_geciren_filtre(voltage)
    detected_spikes = SpikeSorter.spike_tespit_et(v_filt, th_multiplier=3.5)
    
    assert len(v_filt) == len(voltage)
    assert len(detected_spikes) > 0


def test_spike_sorter_pca_gmm_clustering():
    """
    Spike Waveform PCA 2D İndirgemesi ve GMM Kümeleme Testi.
    """
    sim = MEAWaveformSimulator(sampling_rate=30000, duration_sec=0.3)
    voltage, _, _ = sim.uret_ham_elektrot_verisi(num_units=3, seed=42)
    v_filt = SpikeSorter.bant_geciren_filtre(voltage)
    detected_spikes = SpikeSorter.spike_tespit_et(v_filt, th_multiplier=3.5)
    
    waveforms = SpikeSorter.dalga_formu_cikar(v_filt, detected_spikes, window_size=48)
    features_2d, labels, pca, gmm = SpikeSorter.sort_spikes_pca_gmm(waveforms, n_clusters=3)
    
    assert features_2d.shape == (len(waveforms), 2)
    assert len(labels) == len(waveforms)
    assert len(np.unique(labels)) <= 3


def test_lfads_recurrent_generator_forward():
    """
    LFADS VAE Modeli İleri Besleme (Forward Pass) Tensor Boyutu Doğrulaması.
    """
    batch_size = 4
    t_steps = 30
    n_neurons = 10
    latent_dim = 8
    
    model = LFADSRecurrentGenerator(num_neurons=n_neurons, latent_dim=latent_dim, hidden_dim=32)
    x_spikes = torch.poisson(torch.ones(batch_size, t_steps, n_neurons) * 0.5)
    
    log_rates, factors, mu, logvar = model(x_spikes)
    
    assert log_rates.shape == (batch_size, t_steps, n_neurons)
    assert factors.shape == (batch_size, t_steps, latent_dim)
    assert mu.shape == (batch_size, latent_dim)
    assert logvar.shape == (batch_size, latent_dim)


def test_lfads_poisson_loss():
    """
    LFADS Poisson Negative Log-Likelihood ve KL Divergence Kayıp Fonksiyonu Testi.
    """
    batch_size = 4
    t_steps = 20
    n_neurons = 8
    
    model = LFADSRecurrentGenerator(num_neurons=n_neurons, latent_dim=8, hidden_dim=32)
    x_spikes = torch.poisson(torch.ones(batch_size, t_steps, n_neurons) * 0.3)
    
    log_rates, factors, mu, logvar = model(x_spikes)
    total_loss, pois_nll, kl_div = LFADSRecurrentGenerator.compute_poisson_loss(x_spikes, log_rates, mu, logvar)
    
    assert torch.isfinite(total_loss)
    assert total_loss.item() > 0.0
