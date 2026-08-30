"""
Day 327: Closed-Loop Neuro-Prosthetic Control & Haptic Feedback
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

from src.neuro_prosthetic_motoru import (
    MotorCortexDecoder,
    ProstheticArmPlant,
    ICMSSomatosensoryEncoder,
    ClosedLoopNeuroProstheticSimulator,
)


def test_motor_cortex_decoder_output_shape():
    """
    M1 Hız Dekoderinin 2D Vektör Çıktısını Doğrulama.
    """
    decoder = MotorCortexDecoder(num_neurons=16)
    rates = np.random.uniform(5, 40, size=16)
    v_decoded = decoder.decode_velocity(rates)
    
    assert v_decoded.shape == (2,)


def test_prosthetic_arm_plant_kinematics():
    """
    Protez Kol Kinematiği ve Dokunma Kuvveti Tespiti Testi.
    """
    arm = ProstheticArmPlant(initial_pos=(0.0, 0.0), dt=0.1)
    velocity = np.array([1.0, 0.0])
    object_pos = np.array([0.1, 0.0])
    
    new_pos, force = arm.update_step(velocity, object_pos=object_pos)
    
    assert np.allclose(new_pos, [0.1, 0.0])
    assert force > 0.0  # Temas kuvveti oluşmalı


def test_icms_somatosensory_encoder_safety():
    """
    S1 ICMS Akım Genliği Emniyet Sınırı Testi (<= 100 uA).
    """
    encoder = ICMSSomatosensoryEncoder(base_amplitude_ua=10.0)
    
    # 0 N Kuvvet
    amp0, freq0 = encoder.encode_haptic_feedback(0.0)
    assert amp0 == 0.0 and freq0 == 0.0
    
    # Yüksek Kuvvet (10 N)
    amp_high, freq_high = encoder.encode_haptic_feedback(10.0)
    assert amp_high <= 100.0  # Emniyet üst sınırı
    assert freq_high <= 250.0


def test_closed_loop_simulator_error_reduction():
    """
    Kapalı Çevrim Simülasyonunun Hedefe Ulaşma Başarımı.
    """
    sim = ClosedLoopNeuroProstheticSimulator(num_neurons=20)
    target = np.array([0.5, 0.5])
    
    res = sim.run_reaching_simulation(target_pos=target, num_steps=30, closed_loop=True)
    
    assert len(res["trajectory"]) == 31
    assert res["errors"][-1] < res["errors"][0]  # Hata azalmalı
