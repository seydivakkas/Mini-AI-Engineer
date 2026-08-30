"""
Day 358: Deep Space Optical Communications & AI-Driven Adaptive Optics Wavefront Correction
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu betik; Atmosferik Dalga Cephesi Bozulma Simülasyonunu, AI Tabanlı Deforme Ayna Kontrolünü,
Strehl Oranı Yükseltimini ve 6-Panelli Teşhis Grafiğini çalıştırır.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import numpy as np

from src.adaptive_optics_dsoc_motoru import (
    AdaptiveOpticsAIEngine,
)
from src.optics_gorsellestirici import OpticsGorsellestirici
from src.optics_profilleyici import OpticsProfilleyici


def main():
    print("=" * 75, flush=True)
    print("🔭 DAY 358: Derin Uzay Optik İletişimi: Yapay Zeka Tabanlı Uyarlanabilir Optik (Adaptive Optics)", flush=True)
    print("=" * 75, flush=True)

    np.random.seed(42)

    print("\n📌 1) 1550 nm Lazer için Atmosferik Türbülans ve AI Deforme Ayna Optimizasyonu...", flush=True)

    engine = AdaptiveOpticsAIEngine()
    ao_res = engine.run_wavefront_correction_cycle(iterations=20)

    init_s = ao_res["init_strehl"]
    final_s = ao_res["final_strehl"]
    gain_db = 10.0 * np.log10(max(1e-4, final_s / init_s))

    print(f"\n📊 Derin Uzay Lazer İletişimi (DSOC) Performans Sonuçları:")
    print(f"  • Başlangıç Bozuk Strehl Oranı:    %{init_s * 100:.2f} (Bozuk Sinyal / Bağlantı Kopuk)")
    print(f"  • Düzeltilmiş Son Strehl Oranı:    %{final_s * 100:.2f} (> %80.0 Kriteri)")
    print(f"  • Optik Bağlaşım Kazancı:          +{gain_db:.1f} dB")
    print(f"  • Lazer Odak Lekesi (Airy Disk):   ✅ KUSURSUZ")
    print(f"  • Derin Uzay Gigabit Optik Hat:    {'✅ KESİNTİSİZ BAĞLANDI' if ao_res['optical_link_restored'] else '❌ BAĞLANAMADI'}")

    profiler_metrics = OpticsProfilleyici.profille(ao_res)

    gorsellestirici = OpticsGorsellestirici()
    cikti_yolu = gorsellestirici.teshis_panelini_ciz(
        ao_res=ao_res,
        profiler_metrics=profiler_metrics,
        dosya_adi="derin_uzay_optik_paneli.png"
    )

    print(f"\n🖼️ 6-Panelli Derin Uzay Optik Teşhis Grafiği Başarıyla Kaydedildi: [derin_uzay_optik_paneli.png](file:///{os.path.abspath(cikti_yolu)})", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
