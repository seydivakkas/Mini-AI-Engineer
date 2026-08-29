"""
Day 252: Pekiştirmeli Öğrenme ile Robotik Yürüme (RL Locomotion - PPO) Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
from src.rl_locomotion_motoru import (
    RewardShaper,
    PPOActorCritic,
    LocomotionEnvironment,
)
from src.rl_locomotion_profilleyici import RLLocomotionProfilleyici
from src.gorsellestirici import RLLocomotionGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 252 (FAZ 13): PEKİŞTİRMELİ ÖĞRENME İLE ROBOTİK YÜRÜME (QUADRUPED / HUMANOID LOCOMOTION - PPO)")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: 12-DoF Quadruped ve PPO Politikasının Kurulması
    # -------------------------------------------------------------
    print("\n[1/4] 12-DoF Quadruped Simülasyon Ortamı ve PPO Aktör-Kritik Modeli Başlatılıyor...")
    env = LocomotionEnvironment()
    policy = PPOActorCritic(obs_dim=48, act_dim=12)

    print(f"  • Robot Modeli         : 12-DoF Quadruped (4 Bacak x 3 Eklem)")
    print(f"  • PPO Gözlem Boyutu    : {policy.obs_dim}D (Hız, Yönelim, Eklem Konum/Hız, Geçmiş Eylem)")
    print(f"  • PPO Eylem Boyutu     : {policy.act_dim}D (Hedef Eklem Açısı Ötelemesi - Delta q)")

    # -------------------------------------------------------------
    # ADIM 2: Çok Bileşenli Ödül Şekillendirme Yapılandırması
    # -------------------------------------------------------------
    print("\n[2/4] Çok Bileşenli Ödül Şekillendirme (Multi-Component Reward Shaping) Devrede...")
    target_vel = np.array([1.0, 0.0, 0.0])
    print(f"  • Hedef İleri Hız      : {target_vel.tolist()} m/s")
    print("  • Aktif Ödül Terimleri : +Hız Takibi, +Hayatta Kalma, -Gövde Yüksekliği Hatası, -Tork/Enerji Cezası")

    # -------------------------------------------------------------
    # ADIM 3: Kapalı Çevrim PPO Yürüyüş Simülasyonu
    # -------------------------------------------------------------
    print("\n[3/4] Kapalı Çevrim PPO Yürüyüş Adımları İcra Ediliyor (5 Adım Rollout)...")
    for step in range(1, 6):
        obs = np.random.randn(48).astype(np.float32)
        action = policy.get_action(obs)
        step_res = env.step(action)

        torques_np = np.array(step_res["torques"])
        q_dot_np = np.array(step_res["q_dot"])
        cot = RewardShaper.compute_cost_of_transport(torques_np, q_dot_np, mass_kg=12.0, lin_vel_mag=1.0)
        odul = RewardShaper.compute_step_reward(
            lin_vel=np.array(step_res["base_vel"]),
            target_lin_vel=target_vel,
            ang_vel=0.0,
            target_ang_vel=0.0,
            joint_torques=torques_np,
            joint_acc=np.zeros(12),
            base_z=step_res["base_pos"][2],
        )

        v_x = step_res["base_vel"][0]
        pos_x = step_res["base_pos"][0]
        print(f"  [Adım {step}] Konum X: {pos_x:.3f}m | Hız: {v_x:.2f}m/s | Ödül: {odul['toplam_odul']} | COT Enerji: {cot:.2f}")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli RL Locomotion Teşhis Panosu Oluşturuluyor...")
    profil_raporu = RLLocomotionProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "rl_locomotion_paneli.png")

    RLLocomotionGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ RL Locomotion Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 252 (FAZ 13): PEKİŞTİRMELİ ÖĞRENME İLE ROBOTİK YÜRÜME (RL LOCOMOTION) MODÜLÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
