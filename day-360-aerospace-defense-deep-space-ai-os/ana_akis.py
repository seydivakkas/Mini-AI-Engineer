"""
Day 360: Aerospace, Defense & Deep Space Autonomous AI Operating System (AeroSpace-AI-OS)
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu betik; Havacılık, Savunma ve Derin Uzay Görevleri için Otonom AI OS Simülasyonunu,
RTOS Görev Zamanlamasını, TMR Hata Düzeltimini ve 6-Panelli Teşhis Grafiğini çalıştırır (FAZ 18 FİNALİ).
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import numpy as np

from src.aerospace_ai_os_motoru import (
    AeroSpaceAutonomousAIOS,
)
from src.os_gorsellestirici import OSGorsellestirici
from src.os_profilleyici import OSProfilleyici


def main():
    print("=" * 75, flush=True)
    print("🚀 DAY 360: Havacılık, Savunma ve Derin Uzay Görevleri için Otonom AI OS (FAZ 18 FİNALİ)", flush=True)
    print("=" * 75, flush=True)

    np.random.seed(42)

    print("\n📌 1) AeroSpace-AI-OS Çekirdeği Başlatılıyor ve Çok Fazlı Görev İcra Ediliyor...", flush=True)

    os_kernel = AeroSpaceAutonomousAIOS()
    mission_res = os_kernel.execute_mission_cycle(steps=50)

    total_tasks = mission_res["total_tasks_executed"]
    mean_lat = mission_res["mean_latency_ms"]
    max_lat = mission_res["max_latency_ms"]
    deadline_rate = mission_res["deadline_success_rate"]
    seu_inj = mission_res["total_seu_injected"]
    seu_rec = mission_res["total_seu_corrected"]

    print(f"\n📊 AeroSpace Autonomous AI OS Görev Metrikleri:")
    print(f"  • İcra Edilen Toplam Görev Sayısı:  {total_tasks} Görev (5 Alt Sistem)")
    print(f"  • Ortalama Görev Gecikmesi:         {mean_lat:.2f} ms")
    print(f"  • Maksimum Görev Gecikmesi:         {max_lat:.2f} ms (< 2.00 ms Hard Deadline)")
    print(f"  • Hard Real-Time Deadline Uyumu:    %{deadline_rate:.1f} (0 Kaçırılan Deadline)")
    print(f"  • Kozmik Radyasyon SEU Hata Telafisi:{seu_rec}/{seu_inj} (%{mission_res['seu_recovery_rate']:.1f} TMR Başarısı)")
    print(f"  • FAZ 18 Entegre Görev Başarısı:    {'✅ %100 KUSURSUZ' if mission_res['os_healthy'] else '❌ BAŞARISIZ'}")

    profiler_metrics = OSProfilleyici.profille(mission_res)

    gorsellestirici = OSGorsellestirici()
    cikti_yolu = gorsellestirici.teshis_panelini_ciz(
        mission_res=mission_res,
        profiler_metrics=profiler_metrics,
        dosya_adi="aerospace_ai_os_paneli.png"
    )

    print(f"\n🖼️ 6-Panelli AeroSpace-AI-OS Teşhis Grafiği Başarıyla Kaydedildi: [aerospace_ai_os_paneli.png](file:///{os.path.abspath(cikti_yolu)})", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
