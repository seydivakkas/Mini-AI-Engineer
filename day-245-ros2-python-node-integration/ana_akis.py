"""
Day 245: ROS2 Python Node Entegrasyonu ve Sensör-Eyleyici İletişimi Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.ros2_motoru import (
    ROS2Node,
    ROS2Executor,
    RobotSensorActuatorPipeline,
)
from src.ros2_profilleyici import ROS2Profilleyici
from src.gorsellestirici import ROS2Gorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 245 (FAZ 13): ROS2 (ROBOT OPERATING SYSTEM) PYTHON ENTEGRASYONU VE SENSÖR-EYLEYİCİ İLETİŞİMİ")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: ROS 2 Düğümlerinin Başlatılması ve Kaydı
    # -------------------------------------------------------------
    print("\n[1/4] ROS 2 Düğümleri ve DDS İletişim Hattı Başlatılıyor...")
    pipeline = RobotSensorActuatorPipeline()
    for name, node in pipeline.executor.dugumler.items():
        pub_list = list(node.publishers.keys())
        sub_list = list(node.subscriptions.keys())
        print(f"  • Düğüm: [{name}] | Yayıncılar: {pub_list} | Aboneler: {sub_list}")

    # -------------------------------------------------------------
    # ADIM 2: Sensörden Eyleyiciye Canlı Konu Akışı (Topic Streaming)
    # -------------------------------------------------------------
    print("\n[2/4] Kamera Akışı ve Yapay Zeka Eklem Komutları İletiliyor...")
    pipeline.simule_et(kare_sayisi=3)
    print(f"  ✓ Toplam {len(pipeline.alinan_eylemler)} Adet 7-DoF Eklem Eylemi Eyleyiciye İletildi.")

    for i, eylem in enumerate(pipeline.alinan_eylemler, 1):
        print(f"  [Komut {i}] Frame Ref: {eylem['frame_ref']} | ΔJoints: {eylem['delta_joints'][:4]}...")

    # -------------------------------------------------------------
    # ADIM 3: Senkron RPC Servis Çağrısı (Service Request/Response)
    # -------------------------------------------------------------
    print("\n[3/4] Senkron Kavrama Servisi (/arm/grasp_planner) Tetikleniyor...")
    servis_istegi = {"target_id": "cup_yellow_01"}
    servis_yaniti = pipeline.executor.call_service("/arm/grasp_planner", servis_istegi)
    print(f"  🎯 Servis İsteği: {servis_istegi}")
    print(f"  🎯 Servis Yanıtı: {servis_yaniti}")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli ROS 2 Teşhis Panosu Oluşturuluyor...")
    profil_raporu = ROS2Profilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "ros2_paneli.png")

    ROS2Gorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ ROS 2 Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 245 (FAZ 13): ROS2 PYTHON ENTEGRASYON MODÜLÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
