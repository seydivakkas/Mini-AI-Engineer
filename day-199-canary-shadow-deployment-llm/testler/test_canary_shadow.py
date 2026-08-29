"""
PyTest Birim Testleri - Day 199: Canary Dağıtımı ve Shadow-Traffic.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.canary_shadow_motoru import (
    LLMModelInstance,
    ShadowTrafficMirror,
    CanaryTrafficRouter,
    CanaryCircuitBreaker,
)
from src.canary_profilleyici import CanaryGecisProfilleyici
from src.gorsellestirici import CanaryGorsellestirici


def test_llm_model_instance_predict():
    """1. LLMModelInstance başarılı yanıt ve pozitif gecikme üretmelidir."""
    model = LLMModelInstance("m1", "v1.0.0", is_canary=False, base_latency_ms=20.0)
    res = model.predict("Merhaba")
    assert res["success"] is True
    assert res["model_id"] == "m1"
    assert res["latency_ms"] > 0


def test_llm_model_instance_error():
    """2. Hata olasılığı %100 olan model instance başarısız sonuç ve hata mesajı dönmelidir."""
    model = LLMModelInstance("m_broken", "v1.0.0", is_canary=True, error_prob=1.0)
    res = model.predict("Hatalı istek")
    assert res["success"] is False
    assert model.failed_requests == 1


def test_shadow_traffic_mirror_kullanici_etkisi():
    """3. ShadowTrafficMirror kullanıcıya baseline yanıtı dönmeli ve gölge log kaydetmelidir."""
    base = LLMModelInstance("base", "v1.0.0")
    shadow = LLMModelInstance("shadow", "v2.0.0")
    mirror = ShadowTrafficMirror(base, shadow)

    res = mirror.handle_request("Test")
    assert res["version"] == "v1.0.0"
    assert len(mirror.mirror_logs) == 1
    assert mirror.mirror_logs[0]["shadow_version"] == "v2.0.0"


def test_canary_traffic_router_agirlik_ayari():
    """4. CanaryTrafficRouter ağırlıkları [0.0, 1.0] aralığında sınırlandırmalıdır."""
    base = LLMModelInstance("base", "v1.0.0")
    canary = LLMModelInstance("canary", "v2.0.0")
    router = CanaryTrafficRouter(base, canary, canary_weight=0.1)

    router.set_weight(1.5)
    assert router.canary_weight == 1.0

    router.set_weight(-0.5)
    assert router.canary_weight == 0.0


def test_canary_traffic_router_yonlendirme():
    """5. %100 Canary ağırlığında tüm istekler Canary modeline yönlendirilmelidir."""
    base = LLMModelInstance("base", "v1.0.0")
    canary = LLMModelInstance("canary", "v2.0.0")
    router = CanaryTrafficRouter(base, canary, canary_weight=1.0)

    res, is_canary = router.route_request("Deneme")
    assert is_canary is True
    assert res["version"] == "v2.0.0"


def test_canary_circuit_breaker_rollback():
    """6. Hata eşiği aşıldığında Circuit Breaker Canary ağırlığını 0'a çekmelidir."""
    base = LLMModelInstance("base", "v1.0.0")
    broken_canary = LLMModelInstance("canary", "v2.0.0", error_prob=1.0)
    router = CanaryTrafficRouter(base, broken_canary, canary_weight=1.0)
    breaker = CanaryCircuitBreaker(router, max_error_rate=0.05)

    router.route_request("İstek 1")
    tripped = breaker.check_and_enforce()

    assert tripped is True
    assert router.canary_weight == 0.0


def test_canary_gecis_profilleyici_4_asama():
    """7. Geçiş profilleyicisi 4 aşamalı raporu doğrulamalıdır."""
    rapor = CanaryGecisProfilleyici.kademeli_canary_gecis_simulasyonu()
    assert len(rapor["asama_raporlari"]) == 4
    assert rapor["durum"] == "BAŞARILI GEÇİŞ"


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. CanaryGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_canary_paneli.png")
    gecis_raporu = CanaryGecisProfilleyici.kademeli_canary_gecis_simulasyonu()
    rollback_raporu = CanaryGecisProfilleyici.anomali_ve_otomatik_rollback_simulasyonu()

    CanaryGorsellestirici.teshis_paneli_olustur(
        gecis_raporu=gecis_raporu,
        rollback_raporu=rollback_raporu,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
