"""
Tesla Fast-Boot Yönetici Birim Testleri (PyTest)
================================================
Bu test paketi; önyükleme aşamalarının süre analizini,
systemd blame filtreleme fonksiyonunu ve <2.0s hedefini test eder.

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

from src.tesla_fast_boot_yonetici import TeslaFastBootOptimizer


def test_boot_asama_sureleri_ve_toplam():
    """Önyükleme aşamalarının toplamının < 2.0 saniye olduğu test edilir."""
    opt = TeslaFastBootOptimizer(target_boot_limit_s=2.0)
    stages = opt.analyze_boot_stages()

    assert stages["total_boot_s"] <= 2.0
    assert stages["total_boot_ms"] == sum([
        stages["firmware_post_ms"],
        stages["kernel_init_ms"],
        stages["systemd_userspace_ms"],
        stages["ui_renderer_init_ms"]
    ])


def test_yavas_servis_filtreleme():
    """200 ms üzerindeki servislerin doğru tespit edildiği test edilir."""
    opt = TeslaFastBootOptimizer()
    services = {
        "tesla-can-gateway": 45.0,
        "tesla-ui-renderer": 320.0,
        "tesla-network": 120.0
    }
    slow = opt.find_slow_services(services, threshold_ms=200.0)

    assert len(slow) == 1
    assert "tesla-ui-renderer" in slow
    assert slow["tesla-ui-renderer"] == 320.0


def test_systemd_optimizasyon_zinciri():
    """Optimizasyon sonrasında hiçbir servisin 200 ms eşiğini aşmadığı test edilir."""
    opt = TeslaFastBootOptimizer()
    res = opt.optimize_systemd_chain()

    assert res["is_fast_boot_compliant"] is True
    assert len(res["slow_services_after"]) == 0
