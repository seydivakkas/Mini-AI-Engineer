"""
PyTest Birim Testleri - Day 197: Kubernetes KEDA & HPA vLLM GPU Otomatik Ölçekleme.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.k8s_keda_motoru import (
    KedaMetricCollector,
    KedaScaledObjectSimulator,
    KubernetesClusterController,
)
from src.keda_profilleyici import KedaAutoscalingProfilleyici
from src.gorsellestirici import KedaGorsellestirici


def test_keda_metrik_toplayici():
    """1. KedaMetricCollector tüm Prometheus vLLM anahtarlarını eksiksiz toplamalıdır."""
    m = KedaMetricCollector.metrik_topla(num_requests_waiting=10, gpu_cache_usage_factor=0.75)
    assert "vllm:num_requests_waiting" in m
    assert "vllm:gpu_cache_usage_factor" in m
    assert m["vllm:num_requests_waiting"] == 10.0


def test_keda_scaled_object_kuyruk_tetikleyici():
    """2. Yüksek kuyruk derinliği (>20 istek) pod sayısını artırmalıdır."""
    scaled = KedaScaledObjectSimulator(min_replicas=1, max_replicas=10, target_waiting_per_pod=5.0)
    m = {"vllm:num_requests_waiting": 25.0, "vllm:gpu_cache_usage_factor": 0.40}
    hedef = scaled.hesapla_hedef_replika(m)
    assert hedef == 5


def test_keda_scaled_object_kv_cache_tetikleyici():
    """3. KV Cache doluluğu eşiği (%95) aştığında pod artışı tetiklenmelidir."""
    scaled = KedaScaledObjectSimulator(min_replicas=2, max_replicas=8, target_kv_cache_usage=0.80)
    scaled.current_replicas = 2
    m = {"vllm:num_requests_waiting": 2.0, "vllm:gpu_cache_usage_factor": 0.96}
    hedef = scaled.hesapla_hedef_replika(m)
    assert hedef >= 3


def test_keda_min_max_sinirlari():
    """4. KEDA replika sayısı min_replicas ve max_replicas sınırlarını ihlal etmemelidir."""
    scaled = KedaScaledObjectSimulator(min_replicas=2, max_replicas=6)

    # Aşırı düşük yük
    m_low = {"vllm:num_requests_waiting": 0.0, "vllm:gpu_cache_usage_factor": 0.05}
    assert scaled.hesapla_hedef_replika(m_low) >= 2

    # Aşırı yüksek yük
    m_high = {"vllm:num_requests_waiting": 500.0, "vllm:gpu_cache_usage_factor": 0.99}
    assert scaled.hesapla_hedef_replika(m_high) <= 6


def test_k8s_controller_pod_olusturma():
    """5. Controller hedef sayıya göre doğru sayıda Running pod tahsis etmelidir."""
    controller = KubernetesClusterController(cluster_total_gpus=8)
    pods = controller.podlari_senkronize_et(hedef_replika=4)
    assert len(pods) == 4
    assert pods[0]["status"] == "Running"


def test_k8s_controller_pod_azaltma():
    """6. Controller hedef azaldığında podları küçültmelidir."""
    controller = KubernetesClusterController(cluster_total_gpus=8)
    controller.podlari_senkronize_et(hedef_replika=6)
    pods_reduced = controller.podlari_senkronize_et(hedef_replika=2)
    assert len(pods_reduced) == 2


def test_keda_24_saatlik_simulasyon():
    """7. 24 saatlik simülasyon statik tahsise göre >%40 maliyet tasarrufu doğrulamalıdır."""
    rapor = KedaAutoscalingProfilleyici.yirmidort_saatlik_simulasyon_calistir()
    assert len(rapor["adimlari"]) == 6
    assert rapor["maliyet_tasarruf_yuzdesi"] > 40.0


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. KedaGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_keda_paneli.png")
    rapor = KedaAutoscalingProfilleyici.yirmidort_saatlik_simulasyon_calistir()

    KedaGorsellestirici.teshis_paneli_olustur(
        simulasyon_raporu=rapor,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
