"""
PyTest Birim Testleri - Day 273 (FAZ 14): NVLink ve GPUDirect RDMA.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nvlink_rdma_motoru import NVLinkCrossGPUEngine
from src.nvlink_rdma_profilleyici import NVLinkRDMAProfilleyici
from src.gorsellestirici import NVLinkRDMAGorsellestirici


def test_nvlink_engine_init():
    """1. NVLink motoru 8 GPU ve tam P2P eşleme matrisiyle başlamalıdır."""
    engine = NVLinkCrossGPUEngine(num_gpus=8, interconnect_type="NVLink-4")
    assert engine.num_gpus == 8
    assert engine.interconnect_type == "NVLink-4"
    assert engine.peer_access_matrix.shape == (8, 8)
    assert len(engine.virtual_gpu_vram) == 8


def test_peer_access_validation():
    """2. P2P erişim kontrolü geçerli ve geçersiz GPU çiftlerini doğru denetlemelidir."""
    engine = NVLinkCrossGPUEngine(num_gpus=8)
    assert engine.can_access_peer(0, 1) is True
    assert engine.can_access_peer(0, 7) is True
    assert engine.can_access_peer(0, 8) is False
    assert engine.can_access_peer(-1, 2) is False


def test_direct_p2p_transfer_execution():
    """3. Doğrudan P2P transferi sıfır CPU kopyalama ile başarıyla çalışmalıdır."""
    engine = NVLinkCrossGPUEngine(num_gpus=8, interconnect_type="NVLink-4")
    data = np.ones((1024, 1024), dtype=np.float32)  # 4 MB
    stats = engine.direct_p2p_transfer(src_gpu=1, dst_gpu=4, tensor_name="weight_matrix", tensor_data=data)
    
    assert stats["src_gpu"] == 1
    assert stats["dst_gpu"] == 4
    assert stats["size_mb"] == 4.0
    assert stats["zero_copy"] is True
    assert stats["cpu_overhead_pct"] == 0.0


def test_direct_p2p_data_integrity():
    """4. Aktarılan tensör verisi hedef VRAM'de eksiksiz ve bit düzeyinde özdeş olmalıdır."""
    res = NVLinkCrossGPUEngine.execute_mock_cross_gpu_pipeline(tensor_size_mb=16.0)
    assert res["veri_dogrulugu"] is True
    assert res["hedef_vram_boyutu"] == 16.0


def test_ring_allreduce_speedup():
    """5. 8 GPU Ring All-Reduce işleminde NVLink-4 en az 10x hızlanma sağlamalıdır."""
    engine = NVLinkCrossGPUEngine(num_gpus=8, interconnect_type="NVLink-4")
    stats = engine.execute_ring_allreduce(tensor_shape=(2048, 2048, 64))
    assert stats["hizlanma_orani"] >= 10.0
    assert stats["nvlink_cpu_overhead_pct"] == 0.0


def test_nvlink_profiler_metrics():
    """6. NVLinkRDMAProfilleyici tam karşılaştırma metriklerini üretmelidir."""
    profil = NVLinkRDMAProfilleyici.basarim_profili_cikar()
    karsilastirma = profil["karsilastirma"]
    assert karsilastirma["p2p_gecikmesi_us"]["NVLink_4_H100_RDMA"] == 1.1
    assert karsilastirma["etkin_bant_genisligi_gb_s"]["NVLink_4_H100_RDMA"] == 582.0
    assert profil["hizlanma_orani"] > 15.0


def test_cpu_overhead_zero_copy():
    """7. PCIe'de %35 CPU yükü oluşurken NVLink/RDMA'de %0.0 olmalıdır."""
    profil = NVLinkRDMAProfilleyici.basarim_profili_cikar()
    cpu_yuku = profil["karsilastirma"]["cpu_host_ek_yuku_yuzde"]
    assert cpu_yuku["Standart_PCIe_Gen4"] == 35.0
    assert cpu_yuku["NVLink_4_H100_RDMA"] == 0.0


def test_gorsellestirici_dashboard_creation(tmp_path):
    """8. NVLinkRDMAGorsellestirici 6 panelli teşhis panosunu başarıyla kaydetmelidir."""
    cikti = str(tmp_path / "test_nvlink_paneli.png")
    profil = NVLinkRDMAProfilleyici.basarim_profili_cikar()

    NVLinkRDMAGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
