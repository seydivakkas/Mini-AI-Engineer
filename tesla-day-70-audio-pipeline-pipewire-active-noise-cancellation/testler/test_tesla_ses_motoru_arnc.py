"""
Tesla Ses Motoru ve ARNC Birim Testleri (PyTest)
================================================
Bu test paketi; Aktif Yol Gürültüsü Engelleme (ARNC) 180° ters faz üretimini,
desibel sönümleme oranını ve PipeWire çok bölgeli yönlendirmesini test eder.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import pytest
import numpy as np
import sys
import os

su_an_dizin = os.path.dirname(os.path.abspath(__file__))
ana_dizin = os.path.dirname(su_an_dizin)
if ana_dizin not in sys.path:
    sys.path.insert(0, ana_dizin)

from src.tesla_ses_motoru_arnc import TeslaARNCNoiseCanceller, TeslaMultiZoneAudioRouter, AudioZone


def test_anti_noise_ters_faz_dogrulama():
    """Anti-noise sinyalinin ham gürültüyü ters fazda karşıladığı test edilir."""
    canceller = TeslaARNCNoiseCanceller()
    test_sig = np.array([0.5, -0.8, 0.2], dtype=np.float32)
    anti_sig = canceller.generate_anti_noise_phase(test_sig, phase_error_rad=0.0)

    assert np.allclose(anti_sig, -test_sig)


def test_gurultu_sonumleme_orani():
    """ARNC algoritmasının en az 12 dB gürültü düşüşü sağladığı test edilir."""
    canceller = TeslaARNCNoiseCanceller()
    res = canceller.process_noise_reduction(frames=480)

    assert res["is_effective"] is True
    assert res["db_reduction"] >= 12.0
    assert res["residual_noise_power"] < res["raw_noise_power"]


def test_cok_bolgeli_ses_yonlendirme():
    """Otopilot seslerinin sürücü başlığına yönlendirildiği test edilir."""
    router = TeslaMultiZoneAudioRouter()
    zone = router.route_audio_stream("autopilot_chime")

    assert zone == AudioZone.DRIVER_HEADREST.value
