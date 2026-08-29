"""
PyTest Birim Testleri - Day 196: Ray Cluster & Ray Serve Dağıtık Model Dağıtımı.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ray_serve_motoru import (
    RayClusterNode,
    RayServeModelReplica,
    RayServeRouter,
    RayServeDeploymentManager,
)
from src.ray_profilleyici import RayClusterYukProfilleyici
from src.gorsellestirici import RayServeGorsellestirici


def test_ray_dugum_gpu_tahsisi():
    """1. Ray düğümü boş GPU'ları sırayla tahsis etmeli ve kapasiteyi aşmamalıdır."""
    node = RayClusterNode("test_node", "10.0.0.1", gpu_count=2)
    g1 = node.gpu_ayir()
    g2 = node.gpu_ayir()
    g3 = node.gpu_ayir()

    assert g1 == 0
    assert g2 == 1
    assert g3 is None
    assert node.allocated_gpus == 2


def test_ray_dugum_gpu_serbest_birakma():
    """2. GPU serbest bırakıldığında tahsis edilen GPU sayısı azalmalıdır."""
    node = RayClusterNode("test_node", "10.0.0.1", gpu_count=2)
    node.gpu_ayir()
    node.gpu_serbest_birak()
    assert node.allocated_gpus == 0


def test_ray_replika_istek_isleme():
    """3. Ray Serve replikası istek işleyip pozitif bir gecikme süresi dönmelidir."""
    rep = RayServeModelReplica("rep_1", "node_1", gpu_id=0)
    res = rep.process_request(prompt_len=64, gen_tokens=16)
    assert res["replica_id"] == "rep_1"
    assert res["latency_ms"] > 0
    assert rep.processed_requests == 1


def test_ray_router_en_uygun_replika_secimi():
    """4. Ray Router aktif replikalar arasından seçim yapmalıdır."""
    rep1 = RayServeModelReplica("rep_1", "node_1", gpu_id=0)
    rep2 = RayServeModelReplica("rep_2", "node_1", gpu_id=1)
    rep1.current_queue_depth = 5
    rep2.current_queue_depth = 1

    secilen = RayServeRouter.en_uygun_replika_sec([rep1, rep2])
    assert secilen is not None
    assert secilen.replica_id in ["rep_1", "rep_2"]


def test_ray_deployment_manager_varsayilan_baslatma():
    """5. DeploymentManager minimum replika sayısı kadar başlangıç aktörü oluşturmalıdır."""
    manager = RayServeDeploymentManager(min_replicas=3, max_replicas=6)
    manager.dugum_ekle(RayClusterNode("node_1", "10.0.0.1", gpu_count=4))
    manager.baslat_varsayilan_replikalar()
    assert len([r for r in manager.replicas if r.is_active]) == 3


def test_ray_autoscaler_olcek_artirma_ve_azaltma():
    """6. Ray Autoscaler gelen yüke göre replika sayısını artırmalı ve azaltmalıdır."""
    manager = RayServeDeploymentManager(min_replicas=2, max_replicas=8, target_ongoing_requests=5)
    manager.dugum_ekle(RayClusterNode("node_1", "10.0.0.1", gpu_count=8))
    manager.baslat_varsayilan_replikalar()

    # Yüksek yük -> Replika artmalı
    aktifs_high = manager.autoscale(total_incoming_requests=35)
    assert aktifs_high >= 7

    # Düşük yük -> Replika azalmalı
    aktifs_low = manager.autoscale(total_incoming_requests=5)
    assert aktifs_low <= 3


def test_ray_profilleyici_simulasyon_raporu():
    """7. Yük profilleyicisi 3 fazlı raporu ve 12 GPU küme topolojisini doğrulamalıdır."""
    rapor = RayClusterYukProfilleyici.kume_yuk_simulasyonu_calistir()
    assert rapor["toplam_kume_gpu"] == 12
    assert len(rapor["faz_raporlari"]) == 3
    assert rapor["faz_raporlari"][2]["aktif_replika_sayisi"] == 8


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. RayServeGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_ray_paneli.png")
    rapor = RayClusterYukProfilleyici.kume_yuk_simulasyonu_calistir()

    RayServeGorsellestirici.teshis_paneli_olustur(
        simulasyon_raporu=rapor,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
