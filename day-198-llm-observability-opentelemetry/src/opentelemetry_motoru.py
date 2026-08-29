"""
OpenTelemetry Dağıtık İzleme (Distributed Tracing) ve LLM Gözlemlenebilirlik Motoru (Day 198 - FAZ 10).
TTFT (Time-To-First-Token), TPOT (Time-Per-Output-Token), Kuyruk Gecikmesi ve OTel Span Mimarisi.
"""

from typing import Dict, Any, List, Optional
import time
import uuid
import random


class OTelSpan:
    """
    OpenTelemetry Dağıtık İzleme Span (Aralık) Nesnesi.
    """

    def __init__(self, name: str, trace_id: str, parent_span_id: Optional[str] = None):
        self.span_id: str = str(uuid.uuid4())[:8]
        self.trace_id: str = trace_id
        self.parent_span_id: Optional[str] = parent_span_id
        self.name: str = name
        self.start_time: float = time.perf_counter()
        self.end_time: Optional[float] = None
        self.duration_ms: float = 0.0
        self.attributes: Dict[str, Any] = {}
        self.events: List[Dict[str, Any]] = []

    def set_attribute(self, key: str, value: Any):
        """Span nesnesine metrik veya etiket (tag) ekler."""
        self.attributes[key] = value

    def add_event(self, event_name: str, attributes: Optional[Dict[str, Any]] = None):
        """Span içerisine zaman damgalı olay kaydeder (ör. token üretildi)."""
        self.events.append({
            "name": event_name,
            "timestamp_offset_ms": (time.perf_counter() - self.start_time) * 1000.0,
            "attributes": attributes or {},
        })

    def finish(self):
        """Span süresini sonlandırır ve milisaniye cinsinden süreyi hesaplar."""
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000.0


class OTelTracer:
    """
    OpenTelemetry İzleyici ve Span Oluşturucu.
    """

    @classmethod
    def start_trace(cls, name: str) -> OTelSpan:
        """Yeni bir ana iz (Root Span) başlatır."""
        trace_id = str(uuid.uuid4())[:16]
        return OTelSpan(name=name, trace_id=trace_id, parent_span_id=None)

    @classmethod
    def start_child_span(cls, name: str, parent: OTelSpan) -> OTelSpan:
        """Mevcut bir span altına bağlı alt span (Child Span) oluşturur."""
        return OTelSpan(name=name, trace_id=parent.trace_id, parent_span_id=parent.span_id)


class LLMObservabilityCollector:
    """
    LLM Çıkarım Yaşam Döngüsü OTel ve Prometheus Metrik Toplayıcısı.
    """

    @classmethod
    def inferans_izi_kaydet(
        cls,
        prompt_len: int = 128,
        gen_tokens: int = 32,
        simule_queue_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Tekil bir LLM çıkarım isteğinin tüm OTel hiyerarşisini oluşturur ve TTFT/TPOT metriklerini çıkarır.
        """
        root_span = OTelTracer.start_trace("POST /v1/chat/completions")
        root_span.set_attribute("gen_ai.system", "vllm")
        root_span.set_attribute("gen_ai.request.model", "Llama-3-70B-Instruct")
        root_span.set_attribute("gen_ai.usage.prompt_tokens", prompt_len)

        # 1. Alt Span: Kuyrukta Bekleme (Queue Wait)
        queue_span = OTelTracer.start_child_span("ContinuousBatching.QueueWait", root_span)
        queue_duration = simule_queue_ms if simule_queue_ms is not None else random.uniform(5.0, 25.0)
        time.sleep(0.001)  # Minimal simülasyon zamanı
        queue_span.finish()
        queue_span.duration_ms = queue_duration

        # 2. Alt Span: Prefill Aşaması (TTFT: Time-To-First-Token)
        prefill_span = OTelTracer.start_child_span("Prefill.FirstTokenGeneration", root_span)
        ttft_duration = 35.0 + (prompt_len * 0.15) + random.uniform(-3.0, 5.0)
        time.sleep(0.001)
        prefill_span.finish()
        prefill_span.duration_ms = ttft_duration
        prefill_span.add_event("first_token_emitted", {"token_id": 1042})

        # 3. Alt Span: Decode Döngüsü (TPOT: Time-Per-Output-Token)
        decode_span = OTelTracer.start_child_span("Decode.TokenGenerationLoop", root_span)
        tpot_avg = 14.5 + random.uniform(-1.0, 2.0)  # ms / token
        decode_total = gen_tokens * tpot_avg
        time.sleep(0.001)
        decode_span.finish()
        decode_span.duration_ms = decode_total

        # Root span sonlandır
        root_span.finish()
        root_total = queue_duration + ttft_duration + decode_total
        root_span.duration_ms = root_total
        root_span.set_attribute("gen_ai.usage.completion_tokens", gen_tokens)

        return {
            "trace_id": root_span.trace_id,
            "root_span": root_span,
            "spans": [root_span, queue_span, prefill_span, decode_span],
            "metrikler": {
                "ttft_ms": ttft_duration,
                "tpot_ms": tpot_avg,
                "queue_wait_ms": queue_duration,
                "total_latency_ms": root_total,
                "prompt_tokens": prompt_len,
                "completion_tokens": gen_tokens,
                "throughput_tok_per_sec": (gen_tokens / (decode_total / 1000.0)),
            }
        }
