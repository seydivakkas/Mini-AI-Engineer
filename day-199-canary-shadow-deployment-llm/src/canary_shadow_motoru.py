"""
Üretimde Canary Dağıtımı ve Shadow-Traffic Motoru (Day 199 - FAZ 10).
A/B Testi, Gölge Trafik Aynalama, Kademeli Geçiş (Weighted Traffic Shift) ve Otomatik Rollback.
"""

from typing import Dict, Any, List, Optional, Tuple
import random
import time


class LLMModelInstance:
    """
    Üretim veya Canary Adayı Olan Model Örneği.
    """

    def __init__(self, model_id: str, version: str, is_canary: bool = False, base_latency_ms: float = 25.0, error_prob: float = 0.0):
        self.model_id = model_id
        self.version = version
        self.is_canary = is_canary
        self.base_latency_ms = base_latency_ms
        self.error_prob = error_prob
        self.total_requests: int = 0
        self.failed_requests: int = 0

    def predict(self, prompt: str) -> Dict[str, Any]:
        """Model çıkarımı yapar ve metrik üretir."""
        self.total_requests += 1
        is_error = random.random() < self.error_prob
        if is_error:
            self.failed_requests += 1
            return {
                "success": False,
                "error": "ModelInferenceError: CUDA Out of Memory or NaN Output",
                "model_id": self.model_id,
                "version": self.version,
                "latency_ms": self.base_latency_ms * 2.0,
            }

        lat = self.base_latency_ms + random.uniform(-3.0, 5.0)
        return {
            "success": True,
            "response": f"[{self.version}] Yanıt üretildi: {prompt[:20]}...",
            "model_id": self.model_id,
            "version": self.version,
            "latency_ms": lat,
        }


class ShadowTrafficMirror:
    """
    Gölge Trafik (Shadow / Dark Launch) Aynalayıcısı.
    Canlı trafiği kullanıcıya hissettirmeden adaya asenkron kopyalar.
    """

    def __init__(self, baseline_model: LLMModelInstance, shadow_model: LLMModelInstance):
        self.baseline_model = baseline_model
        self.shadow_model = shadow_model
        self.mirror_logs: List[Dict[str, Any]] = []

    def handle_request(self, prompt: str) -> Dict[str, Any]:
        """Kullanıcıya baseline yanıtını dönerken gölge modeli asenkron olarak çalıştırır."""
        # 1. Kullanıcıya dönen birincil yanıt
        live_resp = self.baseline_model.predict(prompt)

        # 2. Gölge modele gönderilen asenkron kopya
        shadow_resp = self.shadow_model.predict(prompt)

        log_kaydi = {
            "prompt": prompt,
            "baseline_version": self.baseline_model.version,
            "baseline_latency_ms": live_resp["latency_ms"],
            "shadow_version": self.shadow_model.version,
            "shadow_latency_ms": shadow_resp["latency_ms"],
            "shadow_success": shadow_resp["success"],
            "latency_farki_ms": shadow_resp["latency_ms"] - live_resp["latency_ms"],
        }
        self.mirror_logs.append(log_kaydi)
        return live_resp


class CanaryTrafficRouter:
    """
    Kademeli Ağırlıklı Canary Yönlendiricisi (Canary Traffic Shifter).
    """

    def __init__(self, baseline: LLMModelInstance, canary: LLMModelInstance, canary_weight: float = 0.05):
        self.baseline = baseline
        self.canary = canary
        self.canary_weight = canary_weight  # 0.05 = %5 Canary, %95 Baseline

    def set_weight(self, canary_weight: float):
        """Canary trafik ağırlığını ayarlar (ör. 0.05 -> 0.20 -> 0.50 -> 1.00)."""
        self.canary_weight = max(0.0, min(1.0, canary_weight))

    def route_request(self, prompt: str) -> Tuple[Dict[str, Any], bool]:
        """Trafik ağırlığına göre isteği Baseline veya Canary modeline iletir."""
        use_canary = random.random() < self.canary_weight
        selected_model = self.canary if use_canary else self.baseline
        resp = selected_model.predict(prompt)
        return resp, use_canary


class CanaryCircuitBreaker:
    """
    Canary Otomatik Geri Alma (Automatic Rollback) Sigorta Mekanizması.
    """

    def __init__(self, router: CanaryTrafficRouter, max_error_rate: float = 0.02, max_p99_latency: float = 80.0):
        self.router = router
        self.max_error_rate = max_error_rate
        self.max_p99_latency = max_p99_latency
        self.is_tripped: bool = False

    def check_and_enforce(self) -> bool:
        """Canary modelinin hata oranını ve gecikmesini kontrol eder; aşılırsa rollback yapar."""
        canary = self.router.canary
        if canary.total_requests == 0:
            return False

        error_rate = canary.failed_requests / canary.total_requests
        if error_rate > self.max_error_rate:
            # Acil Rollback: Canary ağırlığını derhal 0'a çek
            self.router.set_weight(0.0)
            self.is_tripped = True
            return True
        return False
