"""
Day 257: Model Predictive Control (MPC) ile Yüksek Hızlı Dinamik Engelden Kaçınma Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from src.dynamic_mpc_motoru import (
    DynamicObstacleTracker,
    NonlinearMPCController,
)
from src.dynamic_mpc_profilleyici import DynamicMPCProfilleyici
from src.gorsellestirici import DynamicMPCGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 257 (FAZ 13): MODEL PREDICTIVE CONTROL (MPC) İLE YÜKSEK HIZLI DİNAMİK ENGELDEN KAÇINMA")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Dinamik Engel Takipçisinin Kurulması
    # -------------------------------------------------------------
    print("\n[1/4] Dinamik Engel Takipçisi ve Hız Vektörü Tanımlanıyor...")
    obs1 = DynamicObstacleTracker(pos_init=(5.0, 2.0), vel_xy=(0.0, -1.2), radius=0.40)
    print(f"  • Engel Başlangıç Konumu      : {obs1.pos.tolist()} m")
    print(f"  • Engel Hız Vektörü (vx, vy)  : {obs1.vel.tolist()} m/s")
    print(f"  • Engel Güvenlik Yarıçapı     : {obs1.radius} m")

    # -------------------------------------------------------------
    # ADIM 2: Nonlinear MPC Kontrolcünün Başlatılması
    # -------------------------------------------------------------
    print("\n[2/4] Kayan Ufuklu NMPC Kontrolcüsü Başlatılıyor...")
    controller = NonlinearMPCController(horizon=15, dt=0.1, robot_radius=0.30, v_max=3.0)
    print(f"  • Kayan Ufuk Adımı (N)        : {controller.N} Adım (1.5 Saniye İleri Bakış)")
    print(f"  • Örnekleme Zamanı (dt)       : {controller.dt} s (50 Hz Çözücü)")
    print(f"  • Maksimum Robot Hızı (v_max) : {controller.v_max} m/s")

    # -------------------------------------------------------------
    # ADIM 3: Dinamik Engelden Kaçış Optimizasyon Döngüsü
    # -------------------------------------------------------------
    print("\n[3/4] Çapraz Geçiş Senaryosunda NMPC Çözümü Hesaplanıyor...")
    robot_init = np.array([0.0, 0.0, 0.0, 1.5], dtype=np.float64)
    goal = (10.0, 0.0)

    res = controller.solve_control(
        current_state=robot_init,
        goal_pos=goal,
        dynamic_obstacles=[obs1],
        v_target=2.4,
    )

    print(f"  • Hesaplanan Optimal İvme (a_0)     : {res['en_iyi_ivme_m_s2']} m/s²")
    print(f"  • Hesaplanan Açısal Hız (omega_0)  : {res['en_iyi_acisal_hiz_rad_s']} rad/s")
    print(f"  • Öngörülen Yörünge Nokta Sayısı   : {len(res['ongorulen_robot_yorungesi'])}")
    print(f"  • NMPC Çözücü Başarı Durumu         : {res['optimizasyon_basarili_mi']}")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Dinamik MPC Teşhis Panosu Oluşturuluyor...")
    profil_raporu = DynamicMPCProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "mpc_avoidance_paneli.png")

    DynamicMPCGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Dinamik MPC Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 257 (FAZ 13): DİNAMİK ENGELDEN KAÇINMA MPC MODÜLÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
