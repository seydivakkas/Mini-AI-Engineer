"""
Day 199: Üretimde Canary Dağıtımı ve Shadow-Traffic ile Sıfır Kesintili Model Güncellemesi Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.canary_shadow_motoru import (
    LLMModelInstance,
    ShadowTrafficMirror,
    CanaryTrafficRouter,
    CanaryCircuitBreaker,
)
from src.canary_profilleyici import CanaryGecisProfilleyici
from src.gorsellestirici import CanaryGorsellestirici


def main():
    print("=" * 110)
    print(">>> Day 199 (FAZ 10): CANARY & SHADOW DEPLOYMENT WITH ZERO DOWNTIME")
    print("=" * 110)

    # -------------------------------------------------------------
    # ADIM 1: Gölge Trafik (Shadow / Dark Launch) Doğrulaması
    # -------------------------------------------------------------
    print("\n[1/4] Gölge Trafik Aynalama (Shadow Traffic) Test Ediliyor...")
    baseline = LLMModelInstance("llama3-70b-v1", "v1.0.0", is_canary=False, base_latency_ms=28.0)
    shadow_candidate = LLMModelInstance("llama3-70b-v2", "v2.0.0", is_canary=True, base_latency_ms=22.0)

    shadow_mirror = ShadowTrafficMirror(baseline, shadow_candidate)
    resp = shadow_mirror.handle_request("Kullanıcı promptu: RAG sistemini açıkla.")
    log = shadow_mirror.mirror_logs[0]

    print(f"  • Kullanıcıya Dönen Yanıt     : {resp['response']}")
    print(f"  • Baseline Sürümü ve Gecikme  : {log['baseline_version']} ({log['baseline_latency_ms']:.1f} ms)")
    print(f"  • Gölge Aday Sürüm ve Gecikme : {log['shadow_version']} ({log['shadow_latency_ms']:.1f} ms)")
    print(f"  • Gölge Başarım Durumu        : {'BAŞARILI' if log['shadow_success'] else 'HATALI'}")
    print("  ✓ Kullanıcı Etkilenmeden Gölge Trafik Test Edildi!")

    # -------------------------------------------------------------
    # ADIM 2: 4 Aşamalı Kademeli Canary Geçiş Simülasyonu
    # -------------------------------------------------------------
    print("\n[2/4] 4 Aşamalı Kademeli Canary Trafik Kaydırma (%5 -> %100) Yürütülüyor...")
    gecis_raporu = CanaryGecisProfilleyici.kademeli_canary_gecis_simulasyonu()

    print("-" * 110)
    print(f"{'Geçiş Aşaması':<28} | {'Hedef Canary %':<16} | {'Canary İstek':<14} | {'Baseline İstek':<16} | {'Gerçekleşen Canary %'}")
    print("-" * 110)
    for a in gecis_raporu["asama_raporlari"]:
        print(
            f"{a['asama_adi']:<28} | "
            f"{a['hedef_canary_yuzde']:<16} | "
            f"{a['canary_istek_sayisi']:>8} İstek    | "
            f"{a['baseline_istek_sayisi']:>10} İstek      | "
            f"%{a['gerceklesen_canary_yuzde']:>6.1f}"
        )
    print("-" * 110)

    # -------------------------------------------------------------
    # ADIM 3: Anomali Enjeksiyonu ve Otomatik Rollback
    # -------------------------------------------------------------
    print("\n[3/4] Anomali Enjeksiyonu ve Devre Kesici (Circuit Breaker) Rollback Testi...")
    rollback_raporu = CanaryGecisProfilleyici.anomali_ve_otomatik_rollback_simulasyonu()

    print(f"  • Otomatik Rollback Tetiklendi mi: {rollback_raporu['rollback_tetiklendi']}")
    print(f"  • Canary Son Trafik Ağırlığı      : %{rollback_raporu['canary_son_agirlik'] * 100:.0f} (Güvenli Sıfırlama)")
    print(f"  • Toplam Canary İsteği           : {rollback_raporu['toplam_canary_istek']}")
    print(f"  • Hata Veren Canary İsteği       : {rollback_raporu['hatali_canary_istek']}")
    print("  ✓ Güvenlik Eşiği Aşılınca Anında %100 Baseline'a Dönüş Sağlandı!")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Görsel Teşhis Panosu Üretimi
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Canary & Shadow Deployment Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "canary_shadow_paneli.png")

    CanaryGorsellestirici.teshis_paneli_olustur(
        gecis_raporu=gecis_raporu,
        rollback_raporu=rollback_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Canary & Shadow Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 110)
    print("✓ Day 199: CANARY & SHADOW DEPLOYMENT BAŞARIYLA TAMAMLANDI!")
    print("=" * 110)


if __name__ == "__main__":
    main()
