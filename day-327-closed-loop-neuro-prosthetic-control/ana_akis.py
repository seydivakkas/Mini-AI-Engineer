"""
Day 327: Closed-Loop Neuro-Prosthetic Control & Haptic Feedback
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu betik; M1 nöronlarından 2D protez kol hızı dekodlamasını, nesne temas kuvvetini,
S1 İntrakortikal Mikrostimülasyon (ICMS) dokunsal geri bildirimini ve kapalı çevrim yörünge kontrolünü simüle eder.
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
from src.neuro_prosthetic_motoru import ClosedLoopNeuroProstheticSimulator
from src.neuro_gorsellestirici import NeuroGorsellestirici
from src.neuro_profilleyici import NeuroProfilleyici


def main():
    print("=" * 75, flush=True)
    print("🧠 DAY 327: Kapalı Çevrim Nöro-Protez Kontrolü ve S1 ICMS Dokunsal Geri Bildirim", flush=True)
    print("=" * 75, flush=True)

    target_pos = np.array([1.0, 0.8])
    object_pos = np.array([0.85, 0.65])
    num_neurons = 24
    num_steps = 60

    print(f"📌 Protez Kol Görevi: Hedef Konum {target_pos} | Engel/Nesne Konumu {object_pos}", flush=True)
    print(f"📌 M1 Nöron Popülasyonu: {num_neurons} Nöron (Preferred Directions)", flush=True)

    sim = ClosedLoopNeuroProstheticSimulator(num_neurons=num_neurons)

    # 1. Açık Çevrim Simülasyonu (Dokunsal Geri Bildirimsiz)
    print("\n⚠️ 1) Açık Çevrim (Open-Loop) Protez Kol Ulaşma Simülasyonu Çalıştırılıyor...", flush=True)
    start_time = time.time()
    open_loop_res = sim.run_reaching_simulation(
        target_pos=target_pos,
        object_pos=object_pos,
        num_steps=num_steps,
        closed_loop=False
    )
    ol_time = time.time() - start_time
    print(f"✅ Açık Çevrim Tamamlandı! Son Hata: {open_loop_res['errors'][-1]:.4f} metre", flush=True)

    # 2. Kapalı Çevrim Simülasyonu (S1 ICMS Dokunsal Geri Bildirimli)
    print("\n⚡ 2) Kapalı Çevrim (Closed-Loop + S1 ICMS Haptic) Simülasyonu Çalıştırılıyor...", flush=True)
    start_time = time.time()
    closed_loop_res = sim.run_reaching_simulation(
        target_pos=target_pos,
        object_pos=object_pos,
        num_steps=num_steps,
        closed_loop=True
    )
    cl_time = time.time() - start_time
    print(f"✅ Kapalı Çevrim Tamamlandı! Son Hata: {closed_loop_res['errors'][-1]:.4f} metre", flush=True)

    # 3. Profilleme Metrikleri
    profiler_metrics = NeuroProfilleyici.profille(
        closed_loop_res=closed_loop_res,
        open_loop_res=open_loop_res,
        latency_ms=(cl_time * 1000.0) / num_steps
    )

    print("\n📊 Kapalı Çevrim Nöro-Protez & ICMS Metrikleri:", flush=True)
    print(f"  • Açık Çevrim Son Hata:            {profiler_metrics['final_err_ol']:.4f} m", flush=True)
    print(f"  • Kapalı Çevrim Son Hata:          {profiler_metrics['final_err_cl']:.4f} m", flush=True)
    print(f"  • Hata Azaltma İyileşmesi:         %{profiler_metrics['error_reduction_pct']:.2f}", flush=True)
    print(f"  • Maksimum Temas Kuvveti:          {profiler_metrics['max_force_n']:.2f} N", flush=True)
    print(f"  • Maksimum S1 ICMS Akım Genliği:   {profiler_metrics['max_amp_ua']:.1f} uA (Güvenli Sınır <= 100 uA)", flush=True)

    # 4. 6-Panelli Görsel Teşhis Grafiği
    gorsellestirici = NeuroGorsellestirici()
    cikti_yolu = gorsellestirici.teshis_panelini_ciz(
        closed_loop_res=closed_loop_res,
        open_loop_res=open_loop_res,
        profiler_metrics=profiler_metrics
    )

    print(f"\n🖼️ 6-Panelli Nöro-Protez Teşhis Grafiği Başarıyla Kaydedildi: [neuro_protez_paneli.png](file:///{os.path.abspath(cikti_yolu)})", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
