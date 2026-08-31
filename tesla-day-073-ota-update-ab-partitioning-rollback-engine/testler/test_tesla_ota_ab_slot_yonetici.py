"""
Tesla OTA ve A/B Slot Birim Testleri (PyTest)
=============================================
Bu test paketi; A/B slot güncelleme mekanizmasını,
başarısız boot sayacını ve 3. hatada otomatik rollback sürecini test eder.

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

from src.tesla_ota_ab_slot_yonetici import OTABootSlotManager, SlotState


def test_ota_guncelleme_ve_slot_degisimi():
    """OTA güncellemesi pasif slota yazıldığında aktif slotun değiştiği test edilir."""
    mgr = OTABootSlotManager(initial_slot='A', initial_version="2026.4.1")
    mgr.apply_ota_update("2026.8.0", "valid_hash")

    assert mgr.active_slot == 'B'
    assert mgr.passive_slot == 'A'
    assert mgr.slot_versions['B'] == "2026.8.0"


def test_3_hatada_otomatik_rollback():
    """3 ardışık boot hatasında otomatik rollback yapıldığı test edilir."""
    mgr = OTABootSlotManager(initial_slot='A', initial_version="2026.4.1")
    mgr.apply_ota_update("2026.8.0-FAULTY", "hash")

    # Slot B aktif durumda, 3 hata üret
    r1 = mgr.on_boot_failure()
    assert r1["rollback_occurred"] is False
    assert r1["active_slot"] == 'B'

    r2 = mgr.on_boot_failure()
    assert r2["rollback_occurred"] is False

    r3 = mgr.on_boot_failure()
    assert r3["rollback_occurred"] is True
    assert r3["active_slot"] == 'A'
    assert r3["current_version"] == "2026.4.1"


def test_hatali_ota_kurtarma_simulasyonu():
    """Bozuk güncelleme senaryosunun baştan sona başarıyla kurtarıldığı test edilir."""
    mgr = OTABootSlotManager()
    res = mgr.simulate_corrupted_ota_rollback()

    assert res["rollback_success"] is True
    assert res["final_active_slot"] == 'A'
    assert res["final_version"] == "2026.4.1"
