"""
Day 337: Non-Invasive BCI P300 Speller & Error-Related Potential (ErrP) Real-Time Correction
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Birim Test Paketi (PyTest Suite)
"""

import sys
import os
import pytest
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.p300_speller_motoru import (
    P300SignalSimulator,
    P300MatrixDecoder,
    ErrPDetectorAndCorrector,
)
from src.p300_profilleyici import P300Profilleyici


def test_p300_signal_simulator_epoch_shapes():
    """
    P300 ERP Sinyal Simülatörü Şekil ve Genlik Testi.
    """
    sim = P300SignalSimulator(fs=250, n_channels=8)
    target_epoch = sim.generate_p300_epoch(is_target=True)
    nontarget_epoch = sim.generate_p300_epoch(is_target=False)
    
    assert target_epoch.shape == (8, 225)
    assert nontarget_epoch.shape == (8, 225)
    assert float(np.mean(target_epoch[:, 90:115])) > float(np.mean(nontarget_epoch[:, 90:115]))


def test_p300_matrix_decoder_character():
    """
    6x6 BCI Speller Matris Çözümleme Testi.
    """
    row_scores = np.array([0.1, 0.2, 5.0, 0.1, 0.0, 0.1])  # Row 2 (M,N,O,P,Q,R)
    col_scores = np.array([0.1, 4.5, 0.2, 0.0, 0.1, 0.1])  # Col 1 ('N')
    
    pred_char, r, c = P300MatrixDecoder.decode_target_character(row_scores, col_scores)
    assert pred_char == 'N'
    assert r == 2
    assert c == 1


def test_errp_detector_error_identification():
    """
    Hata Potansiyeli (ErrP N250) Tespiti Testi.
    """
    sim = P300SignalSimulator(fs=250, n_channels=8)
    errp_engine = ErrPDetectorAndCorrector(errp_threshold=-3.5)
    
    err_epoch = sim.generate_errp_epoch(is_error=True)
    no_err_epoch = sim.generate_errp_epoch(is_error=False)
    
    assert errp_engine.detect_error(err_epoch, sim.time_vec) == True
    assert errp_engine.detect_error(no_err_epoch, sim.time_vec) == False


def test_itr_calculation_validity():
    """
    Bilgi Transfer Hızı (ITR bits/min) Hesaplama Testi.
    """
    itr = ErrPDetectorAndCorrector.calculate_itr(n_targets=36, accuracy=0.95, trial_duration_sec=3.0)
    assert itr > 40.0  # ~50-60 bits/min olmalı
