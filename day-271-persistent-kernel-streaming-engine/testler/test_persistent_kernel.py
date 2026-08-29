"""
PyTest Birim Testleri - Day 271 (FAZ 14): Kalıcı Çekirdek (Persistent Kernel) Mimarisi.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.persistent_kernel_motoru import PersistentKernelStreamingEngine
from src.persistent_kernel_profilleyici import PersistentKernelProfilleyici
from src.gorsellestirici import PersistentKernelGorsellestirici


def test_persistent_kernel_init():
    """1. Kalıcı çekirdek motoru doğru SM ve halka tamponu boyutuyla başlamalıdır."""
    engine = PersistentKernelStreamingEngine(num_sms=108, ring_buffer_size=256)
    assert engine.num_sms == 108
    assert engine.ring_buffer_size == 256
    assert len(engine.task_queue) == 0


def test_persistent_kernel_push_task():
    """2. Kilitsiz halka tamponuna görevler başarıyla eklenmelidir."""
    engine = PersistentKernelStreamingEngine(num_sms=108)
    dummy_data = np.zeros((10, 10), dtype=np.float32)
    engine.push_task("RMSNorm_Layer_1", dummy_data, "NORM")
    assert len(engine.task_queue) == 1
    assert engine.task_queue[0]["task_name"] == "RMSNorm_Layer_1"


def test_persistent_stream_latency_speedup():
    """3. 80 katmanlı akışta hızlanma oranı en az 5.0x olmalıdır."""
    engine = PersistentKernelStreamingEngine(num_sms=108)
    stats = engine.execute_persistent_stream(num_layers=80)
    assert stats["hizlanma_orani"] > 5.0
    assert stats["toplam_mikro_kernel_sayisi"] == 320


def test_persistent_stream_step_latency():
    """4. Kalıcı çekirdek adım süresi 150 μs altında kalmalıdır."""
    engine = PersistentKernelStreamingEngine(num_sms=108)
    stats = engine.execute_persistent_stream(num_layers=80)
    assert stats["persistent_toplam_sure_us"] < 150.0
    assert stats["standart_toplam_sure_us"] > 600.0


def test_mock_pipeline_execution_shape():
    """5. Fused pipeline doğru tensör boyutunu üretmelidir."""
    x = np.random.randn(16, 32).astype(np.float32)
    w = np.random.randn(32, 32).astype(np.float32)
    out, stats = PersistentKernelStreamingEngine.execute_mock_persistent_pipeline(x, w)
    assert out.shape == (16, 32)
    assert "0.08 μs" in stats["kernel_launch_gecikmesi"]


def test_mock_pipeline_mathematical_accuracy():
    """6. Fused pipeline matematiksel olarak beklenen değerleri üretmelidir."""
    x = np.ones((4, 8), dtype=np.float32)
    w = np.eye(8, dtype=np.float32)
    out, _ = PersistentKernelStreamingEngine.execute_mock_persistent_pipeline(x, w)
    assert out.shape == (4, 8)
    assert np.all(out > 0.0)


def test_persistent_kernel_profiler_output():
    """7. PersistentKernelProfilleyici 3'lü karşılaştırma raporunu üretmelidir."""
    profil = PersistentKernelProfilleyici.basarim_profili_cikar()
    assert "Persistent_Kernel_Engine" in profil["karsilastirma"]["gecis_ek_yuku_us"]
    assert profil["karsilastirma"]["gecis_ek_yuku_us"]["Persistent_Kernel_Engine"] == 0.08


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. PersistentKernelGorsellestirici 6 panelli teşhis panosunu oluşturmalıdır."""
    cikti = str(tmp_path / "test_persistent_paneli.png")
    profil = PersistentKernelProfilleyici.basarim_profili_cikar()

    PersistentKernelGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
