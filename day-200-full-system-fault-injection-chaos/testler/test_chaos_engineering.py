"""
PyTest Birim Testleri - Day 200: Kaos Mühendisliği ve Arıza Enjeksiyonu.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.chaos_motoru import (
    NodeState,
    GPUClusterNode,
    ChaosInjector,
    ResilientClusterManager,
)
from src.chaos_profilleyici import ChaosDeneyProfilleyici
from src.gorsellestirici import ChaosGorsellestirici


def test_gpu_cluster_node_healthy():
    """1. Sağlıklı GPU düğümü başarılı yanıt ve pozitif gecikme üretmelidir."""
    node = GPUClusterNode("node-0", gpu_id=0, base_latency_ms=20.0)
    res = node.execute_inference("Hello")
    assert res["success"] is True
    assert res["latency_ms"] > 0


def test_gpu_cluster_node_crashed():
    """2. Çökmüş düğüm başarısız yanıt dönmeli ve hata sayısını artırmalıdır."""
    node = GPUClusterNode("node-0", gpu_id=0)
    node.state = NodeState.CRASHED
    res = node.execute_inference("Hello")
    assert res["success"] is False
    assert node.failed_requests == 1


def test_chaos_injector_oom():
    """3. ChaosInjector.inject_gpu_oom düğüm durumunu OOM yapmalıdır."""
    node = GPUClusterNode("node-0", gpu_id=0)
    ChaosInjector.inject_gpu_oom(node)
    assert node.state == NodeState.OOM


def test_chaos_injector_network_latency():
    """4. ChaosInjector.inject_network_latency düğümü DEGRADED yapmalı ve gecikme eklemelidir."""
    node = GPUClusterNode("node-0", gpu_id=0)
    ChaosInjector.inject_network_latency(node, delay_ms=100.0)
    assert node.state == NodeState.DEGRADED
    assert node.extra_latency_ms == 100.0


def test_chaos_injector_node_kill():
    """5. ChaosInjector.inject_node_kill düğümü CRASHED durumuna getirmelidir."""
    node = GPUClusterNode("node-0", gpu_id=0)
    ChaosInjector.inject_node_kill(node)
    assert node.state == NodeState.CRASHED


def test_resilient_cluster_failover():
    """6. Bir düğüm arızalandığında küme yöneticisi diğer sağlıklı düğüme otomatik failover yapmalıdır."""
    cluster = ResilientClusterManager(initial_node_count=2)
    ChaosInjector.inject_node_kill(cluster.nodes[0])  # İlk düğümü öldür

    res = cluster.route_inference("Failover query", max_retries=2)
    assert res["success"] is True
    assert res["node_id"] == cluster.nodes[1].node_id


def test_resilient_cluster_self_heal():
    """7. health_check_and_heal çöken düğümleri otomatik olarak yenilemelidir."""
    cluster = ResilientClusterManager(initial_node_count=3)
    ChaosInjector.inject_node_kill(cluster.nodes[0])
    ChaosInjector.inject_gpu_oom(cluster.nodes[1])

    healed_count = cluster.health_check_and_heal()
    assert healed_count == 2
    assert len(cluster.get_healthy_nodes()) == 3


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. ChaosGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_chaos_paneli.png")
    deney_raporu = ChaosDeneyProfilleyici.tam_kaos_deneyi_yurut()

    ChaosGorsellestirici.teshis_paneli_olustur(
        deney_raporu=deney_raporu,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
