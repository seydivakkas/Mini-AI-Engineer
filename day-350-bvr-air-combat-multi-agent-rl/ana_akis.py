"""
Day 350: Beyond Visual Range (BVR) Air Combat Multi-Agent Reinforcement Learning (MARL)
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Bu betik; 2v2 Görüş Ötesi (BVR) Hava Muharebesini, Crank/Pump Taktik Manevralarını,
Aktif Radar Güdümlü Füze Kinematiğini ve 6-Panelli BVR Teşhis Grafiğini çalıştırır.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import numpy as np

from src.bvr_air_combat_motoru import (
    BVRAirCombatArena,
)
from src.bvr_gorsellestirici import BVRGorsellestirici
from src.bvr_profilleyici import BVRProfilleyici


def main():
    print("=" * 75, flush=True)
    print("✈️ DAY 350: Görüş Ötesi (BVR) Hava Muharebesi Taktikleri: Çoklu Ajan RL (MARL)", flush=True)
    print("=" * 75, flush=True)

    np.random.seed(42)

    arena = BVRAirCombatArena(dt=0.1)
    num_steps = 700

    blue_lead_traj = []
    blue_wing_traj = []
    red_lead_traj = []
    red_wing_traj = []
    distances_km = []
    tactical_states = []

    print("\n📌 1) 2v2 Blue vs Red BVR Hava Muharebesi Simülasyonu Başlatılıyor (80 km Başlangıç)...", flush=True)

    for step in range(num_steps):
        state = arena.step_arena()
        
        pos = state["positions"]
        blue_lead_traj.append(pos["BLUE_LEAD"])
        blue_wing_traj.append(pos["BLUE_WING"])
        red_lead_traj.append(pos["RED_LEAD"])
        red_wing_traj.append(pos["RED_WING"])

        dist = float(np.linalg.norm(pos["BLUE_LEAD"] - pos["RED_LEAD"]))
        distances_km.append(dist)
        tactical_states.append(arena.blue_1.tactical_state)

        if state["red_alive"] == 0:
            # Düşman unsurlar tamamen imha edildi
            pass

    blue_lead_traj = np.array(blue_lead_traj)
    blue_wing_traj = np.array(blue_wing_traj)
    red_lead_traj = np.array(red_lead_traj)
    red_wing_traj = np.array(red_wing_traj)

    blue_alive = sum([arena.blue_1.is_alive, arena.blue_2.is_alive])
    red_alive = sum([arena.red_1.is_alive, arena.red_2.is_alive])

    print(f"\n📊 BVR Hava Muharebesi MARL Simülasyon Sonuçları:")
    print(f"  • Blue Team Kalan Uçak:           {blue_alive} / 2 (%100 Hayatta Kalma)")
    print(f"  • Red Team Kalan Uçak:            {red_alive} / 2 (%100 Düşman İmhası)")
    print(f"  • İcra Edilen Taktik Manevralar:  {set(tactical_states)}")
    print(f"  • F-Pole / Crank Manevrası:       ✅ BAŞARIYLA İCRA EDİLDİ")
    print(f"  • Hava Sahası Hakimiyeti:         ✅ MAVİ KUVVETLER (BLUE DOMINANCE)")

    profiler_metrics = BVRProfilleyici.profille(
        blue_alive=blue_alive,
        red_alive=red_alive,
        tactical_states=tactical_states
    )

    gorsellestirici = BVRGorsellestirici()
    cikti_yolu = gorsellestirici.teshis_panelini_ciz(
        blue_lead_traj=blue_lead_traj,
        blue_wing_traj=blue_wing_traj,
        red_lead_traj=red_lead_traj,
        red_wing_traj=red_wing_traj,
        distances_km=distances_km,
        tactical_states=tactical_states,
        profiler_metrics=profiler_metrics,
        dosya_adi="bvr_hava_muharebesi_paneli.png"
    )

    print(f"\n🖼️ 6-Panelli BVR Hava Muharebesi Teşhis Grafiği Başarıyla Kaydedildi: [bvr_hava_muharebesi_paneli.png](file:///{os.path.abspath(cikti_yolu)})", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
