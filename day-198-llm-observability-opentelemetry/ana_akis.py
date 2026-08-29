"""
Day 198: OpenTelemetry & Prometheus ile TTFT ve TPOT İzleme Paneli Ana Çalıştırma Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.opentelemetry_motoru import (
    OTelTracer,
    LLMObservabilityCollector,
)
from src.metrik_profilleyici import LLMGozlemlenebilirlikProfilleyici
from src.gorsellestirici import OTelGorsellestirici


def main():
    print("=" * 110)
    print(">>> Day 198 (FAZ 10): OPENTELEMETRY & PROMETHEUS LLM OBSERVABILITY (TTFT & TPOT ENGINE)")
    print("=" * 110)

    # -------------------------------------------------------------
    # ADIM 1: Tekil İstek OpenTelemetry Dağıtık İzleme (Trace) Üretimi
    # -------------------------------------------------------------
    print("\n[1/4] Tekil LLM Çıkarım İsteği İçin OpenTelemetry İzi Başlatılıyor...")
    ornek_iz = LLMObservabilityCollector.inferans_izi_kaydet(prompt_len=256, gen_tokens=48)
    m = ornek_iz["metrikler"]

    print(f"  • Trace ID                    : {ornek_iz['trace_id']}")
    print(f"  • Oluşturulan OTel Span Adedi : {len(ornek_iz['spans'])} Span (Root, Queue, Prefill, Decode)")
    print(f"  • TTFT (İlk Token Gecikmesi)  : {m['ttft_ms']:.2f} ms")
    print(f"  • TPOT (Token Başı Süre)      : {m['tpot_ms']:.2f} ms / token")
    print(f"  • Kuyruk Bekleme Süresi       : {m['queue_wait_ms']:.2f} ms")
    print(f"  • Toplam İstek Gecikmesi      : {m['total_latency_ms']:.2f} ms")
    print(f"  • Üretim Throughput Hızı      : {m['throughput_tok_per_sec']:.1f} Token / Saniye")
    print("  ✓ OpenTelemetry Hiyerarşik İzleme Başarıyla Tamamlandı!")

    # -------------------------------------------------------------
    # ADIM 2: Toplu İstatistiksel Profilleme (50 Çıkarım İzi)
    # -------------------------------------------------------------
    print("\n[2/4] Kurumsal Trafikte 50 Çıkarım İzi İstatistiksel Dağılımı Çıkarılıyor...")
    profil_raporu = LLMGozlemlenebilirlikProfilleyici.toplu_izleme_profille(trace_sayisi=50)

    ttft = profil_raporu["ttft_istatistik"]
    tpot = profil_raporu["tpot_istatistik"]
    queue = profil_raporu["queue_istatistik"]
    total = profil_raporu["total_latency_istatistik"]

    print("-" * 110)
    print(f"{'Metrik Adı':<32} | {'Ortalama':<14} | {'P50 (Medyan)':<16} | {'P90':<14} | {'P99 (Kuyruk Gecikmesi)'}")
    print("-" * 110)
    print(f"{'TTFT (Time-To-First-Token)':<32} | {ttft['ort']:>8.1f} ms    | {ttft['p50']:>10.1f} ms    | {ttft['p90']:>8.1f} ms  | {ttft['p99']:>10.1f} ms")
    print(f"{'TPOT (Time-Per-Output-Token)':<32} | {tpot['ort']:>8.1f} ms    | {tpot['p50']:>10.1f} ms    | {tpot['p90']:>8.1f} ms  | {tpot['p99']:>10.1f} ms")
    print(f"{'Kuyruk Bekleme (Queue Wait)':<32} | {queue['ort']:>8.1f} ms    | {queue['p50']:>10.1f} ms    | {queue['p90']:>8.1f} ms  | {queue['p99']:>10.1f} ms")
    print(f"{'Toplam İstek Süresi':<32} | {total['ort']:>8.1f} ms    | {total['p50']:>10.1f} ms    | {total['p90']:>8.1f} ms  | {total['p99']:>10.1f} ms")
    print("-" * 110)

    # -------------------------------------------------------------
    # ADIM 3: 6 Panelli Görsel Teşhis Panosu Üretimi
    # -------------------------------------------------------------
    print("\n[3/4] 6 Panelli OpenTelemetry & Prometheus Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "opentelemetry_paneli.png")

    OTelGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ OpenTelemetry Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 110)
    print("✓ Day 198: OPENTELEMETRY LLM OBSERVABILITY BAŞARIYLA TAMAMLANDI!")
    print("=" * 110)


if __name__ == "__main__":
    main()
