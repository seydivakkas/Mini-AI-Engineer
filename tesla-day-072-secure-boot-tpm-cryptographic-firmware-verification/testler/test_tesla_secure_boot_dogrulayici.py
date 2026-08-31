"""
Tesla Secure Boot Birim Testleri (PyTest)
=========================================
Bu test paketi; SHA-256 kriptografik özet hesaplamasını,
sabit zamanlı imza karşılaştırmasını ve güven zinciri doğrulamasını test eder.

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

from src.tesla_secure_boot_dogrulayici import TeslaSecureBootValidator


def test_sha256_hesaplama():
    """SHA-256 fonksiyonunun 64 karakterli geçerli hex ürettiği test edilir."""
    validator = TeslaSecureBootValidator()
    h = validator.compute_sha256(b"TESLA_OFFICIAL_TEST_STRING")

    assert len(h) == 64
    assert isinstance(h, str)


def test_sabit_zamanli_imza_karsilastirma():
    """Doğru imzaların onaylandığı, tahrif edilmiş imzaların reddedildiği test edilir."""
    validator = TeslaSecureBootValidator()
    h1 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    h2 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    h_fake = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"

    assert validator.verify_firmware_integrity(h1, h2) is True
    assert validator.verify_firmware_integrity(h1, h_fake) is False


def test_guven_zinciri_ve_tahrifat_engelleme():
    """Tam güven zincirinin normalde geçtiği, tahrifat durumunda reddedildiği test edilir."""
    validator = TeslaSecureBootValidator()
    res_normal = validator.validate_full_secure_boot_chain(simulate_tamper=False)
    res_tamper = validator.validate_full_secure_boot_chain(simulate_tamper=True)

    assert res_normal["chain_verified"] is True
    assert res_tamper["chain_verified"] is False
    assert res_tamper["tamper_detected"] is True
