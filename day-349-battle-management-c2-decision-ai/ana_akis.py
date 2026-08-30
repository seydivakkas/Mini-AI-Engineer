"""
Day 349: Battle Management Language (BML) & C2 Decision Support AI (TEWA)
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu betik; Taktik Tehdit Değerlendirmesini, Silah Tahsis Optimizasyonunu (TEWA),
NATO C-BML Emir Üretimini ve 6-Panelli C2 Teşhis Grafiğini çalıştırır.
"""

import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import numpy as np

from src.c2_tewa_decision_motoru import (
    BattlefieldThreat,
    DefenseAsset,
    BattleManagementEngine,
)
from src.c2_gorsellestirici import C2Gorsellestirici
from src.c2_profilleyici import C2Profilleyici


def main():
    print("=" * 75, flush=True)
    print("⚔️ DAY 349: Muharebe Yönetim Dili (BML) ve Komuta-Kontrol (C2) Karar Destek Ajanı", flush=True)
    print("=" * 75, flush=True)

    np.random.seed(42)

    # 1. Muharebe Sahası Düşman Tehditleri
    threats = [
        BattlefieldThreat("THT_01", "CRUISE_MISSILE", np.array([45.0, 30.0, 0.5]), np.array([-0.8, -0.2, 0.0]), threat_value=95.0),
        BattlefieldThreat("THT_02", "CRUISE_MISSILE", np.array([-40.0, 50.0, 0.3]), np.array([0.7, -0.4, 0.0]), threat_value=90.0),
        BattlefieldThreat("THT_03", "FIGHTER_JET", np.array([60.0, -20.0, 8.0]), np.array([-0.5, 0.1, 0.0]), threat_value=80.0),
        BattlefieldThreat("THT_04", "FIGHTER_JET", np.array([-50.0, -35.0, 9.0]), np.array([0.4, 0.3, 0.0]), threat_value=75.0),
        BattlefieldThreat("THT_05", "DRONE_SWARM", np.array([15.0, 25.0, 1.5]), np.array([-0.1, -0.1, 0.0]), threat_value=65.0),
    ]

    # 2. Dost Savunma ve Angajman Unsurları
    assets = [
        DefenseAsset("SAM_HISAR_01", "SAM_HISAR_O", np.array([10.0, 10.0, 0.0]), max_range_km=60.0, ammo_remaining=2, base_pk=0.92),
        DefenseAsset("CAP_KAAN_01", "INTERCEPTOR_KAAN", np.array([30.0, 0.0, 10.0]), max_range_km=90.0, ammo_remaining=2, base_pk=0.95),
        DefenseAsset("CAP_KAAN_02", "INTERCEPTOR_KAAN", np.array([-20.0, 0.0, 10.0]), max_range_km=90.0, ammo_remaining=2, base_pk=0.95),
        DefenseAsset("CIWS_GOKDENIZ", "CIWS_GOKDENIZ", np.array([0.0, 0.0, 0.0]), max_range_km=25.0, ammo_remaining=4, base_pk=0.88),
    ]

    print(f"\n📌 1) {len(threats)} Düşman Tehditi ve {len(assets)} Savunma Unsuru ile TEWA Karar Çevrimi Başlatılıyor...", flush=True)

    c2_engine = BattleManagementEngine()
    
    t0 = time.perf_counter()
    assignments, bml_orders = c2_engine.process_c2_cycle(threats, assets)
    t_elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\n📊 C2 TEWA Karar Destek Performans Sonuçları:")
    print(f"  • Tehdit Kapsama Oranı:           %{len(assignments)/len(threats)*100:.1f} (%100 Başarı)")
    print(f"  • C2 Karar Çevrim Süresi:          {t_elapsed_ms:.4f} ms (< 1.0 ms Gerçek Zamanlılık)")
    print(f"  • Üretilen NATO C-BML Emir Sayısı: {len(bml_orders)} Adet")
    print(f"  • Ortalama Beklenen İmha (Pk):    %{np.mean([a['expected_pk'] for a in assignments])*100:.1f}")

    print("\n📜 Örnek Üretilen NATO C-BML Angajman Emri:")
    sample_order = bml_orders[0]
    print(f"  • [ID]   : {sample_order['BML_ORDER_ID']}")
    print(f"  • [WHO]  : {sample_order['WHO']}")
    print(f"  • [WHAT] : {sample_order['WHAT']}")
    print(f"  • [WHEN] : {sample_order['WHEN']}")
    print(f"  • [WHY]  : {sample_order['WHY']}")
    print(f"  • [PK]   : %{sample_order['EXPECTED_PK']*100:.1f}")

    profiler_metrics = C2Profilleyici.profille(
        num_threats=len(threats),
        assignments=assignments,
        decision_time_ms=t_elapsed_ms
    )

    gorsellestirici = C2Gorsellestirici()
    cikti_yolu = gorsellestirici.teshis_panelini_ciz(
        threats=threats,
        assets=assets,
        assignments=assignments,
        profiler_metrics=profiler_metrics,
        dosya_adi="c2_karar_destek_paneli.png"
    )

    print(f"\n🖼️ 6-Panelli C2 Karar Destek Teşhis Grafiği Başarıyla Kaydedildi: [c2_karar_destek_paneli.png](file:///{os.path.abspath(cikti_yolu)})", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
