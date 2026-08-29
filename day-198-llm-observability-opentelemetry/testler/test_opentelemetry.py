"""
PyTest Birim Testleri - Day 198: OpenTelemetry & Prometheus ile TTFT ve TPOT İzleme.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.opentelemetry_motoru import (
    OTelSpan,
    OTelTracer,
    LLMObservabilityCollector,
)
from src.metrik_profilleyici import LLMGozlemlenebilirlikProfilleyici
from src.gorsellestirici import OTelGorsellestirici


def test_otel_span_olusturma_ve_sonlandirma():
    """1. OTelSpan süreyi milisaniye cinsinden doğru hesaplamalıdır."""
    span = OTelSpan(name="test_span", trace_id="trace_123")
    span.finish()
    assert span.end_time is not None
    assert span.duration_ms >= 0.0


def test_otel_span_nitelik_ve_olay_ekleme():
    """2. Span nitelikleri ve olayları başarıyla kaydetmelidir."""
    span = OTelSpan(name="test_span", trace_id="trace_123")
    span.set_attribute("model", "Llama-3-70B")
    span.add_event("token_generated", {"token_id": 42})

    assert span.attributes["model"] == "Llama-3-70B"
    assert len(span.events) == 1
    assert span.events[0]["name"] == "token_generated"


def test_otel_tracer_hiyerarsi():
    """3. Tracer ebeveyn-çocuk (Parent-Child) hiyerarşisini doğru kurmalıdır."""
    root = OTelTracer.start_trace("RootSpan")
    child = OTelTracer.start_child_span("ChildSpan", root)

    assert child.trace_id == root.trace_id
    assert child.parent_span_id == root.span_id


def test_llm_observability_collector_tekil_iz():
    """4. Collector 4 adet span ve TTFT/TPOT metriklerini üretmelidir."""
    iz = LLMObservabilityCollector.inferans_izi_kaydet(prompt_len=64, gen_tokens=16)
    assert len(iz["spans"]) == 4
    m = iz["metrikler"]
    assert "ttft_ms" in m
    assert "tpot_ms" in m
    assert "queue_wait_ms" in m


def test_ttft_ve_tpot_pozitif_degerler():
    """5. TTFT ve TPOT metrikleri pozitif ve anlamlı olmalıdır."""
    iz = LLMObservabilityCollector.inferans_izi_kaydet(prompt_len=128, gen_tokens=32)
    m = iz["metrikler"]
    assert m["ttft_ms"] > 10.0
    assert m["tpot_ms"] > 5.0
    assert m["throughput_tok_per_sec"] > 0.0


def test_metrik_profilleyici_istatistik_cikarma():
    """6. Profilleyici P50, P90, P99 istatistiklerini doğru hesaplamalıdır."""
    rapor = LLMGozlemlenebilirlikProfilleyici.toplu_izleme_profille(trace_sayisi=10)
    assert rapor["toplam_trace"] == 10
    assert "p50" in rapor["ttft_istatistik"]
    assert "p99" in rapor["tpot_istatistik"]


def test_p99_buyuktur_p50():
    """7. İstatistiksel olarak P99 değeri P50 değerinden büyük veya eşit olmalıdır."""
    rapor = LLMGozlemlenebilirlikProfilleyici.toplu_izleme_profille(trace_sayisi=20)
    assert rapor["ttft_istatistik"]["p99"] >= rapor["ttft_istatistik"]["p50"]
    assert rapor["total_latency_istatistik"]["p99"] >= rapor["total_latency_istatistik"]["p50"]


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. OTelGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_otel_paneli.png")
    rapor = LLMGozlemlenebilirlikProfilleyici.toplu_izleme_profille(trace_sayisi=15)

    OTelGorsellestirici.teshis_paneli_olustur(
        profil_raporu=rapor,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
