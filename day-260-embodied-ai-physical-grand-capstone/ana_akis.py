"""
Day 260 (FAZ 13 BÜYÜK FİNALİ): Embodied AI Fiziksel Robotik Süiti Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from src.embodied_capstone_motoru import (
    OpenVLAEmbeddingGenerator,
    DiffusionPolicyActionGenerator,
    ROS2MiddlewareBridge,
    UnifiedEmbodiedAIEngine,
)
from src.embodied_capstone_profilleyici import EmbodiedCapstoneProfilleyici
from src.gorsellestirici import EmbodiedCapstoneGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 260 (FAZ 13 BÜYÜK FİNALİ): EMBODIED AI FİZİKSEL ROBOTİK SÜİTİ (OPENVLA + DIFFUSION POLICY + ROS2)")
    print("=" * 115)

    prompt = "Masanın üzerindeki narin şişeyi çift kolla kavra, hareketli engellerden kaçarak montaj kutusuna yerleştir"
    image_features = np.array([0.45, 0.72, 0.88, 0.12])

    # -------------------------------------------------------------
    # ADIM 1: OpenVLA Çok Modlu Şartlandırma
    # -------------------------------------------------------------
    print("\n[1/4] OpenVLA Çok Modlu Durum Kodlayıcı Başlatılıyor...")
    vla_encoder = OpenVLAEmbeddingGenerator(embed_dim=64)
    cond_vec = vla_encoder.encode(prompt, image_features)
    print(f"  • Girdi Komutu                : '{prompt}'")
    print(f"  • Üretilen Şartlandırma Vektörü: {cond_vec.shape} (Birim Norm: {np.linalg.norm(cond_vec):.2f})")

    # -------------------------------------------------------------
    # ADIM 2: Diffusion Policy 16-Adım Eylem Yığını
    # -------------------------------------------------------------
    print("\n[2/4] Diffusion Policy (DDPM) ile 16 Adımlık Eylem Yığını Üretiliyor...")
    diff_policy = DiffusionPolicyActionGenerator(chunk_size=16, action_dim=7, num_diffusion_steps=10)
    action_chunk = diff_policy.generate_action_chunk(cond_vec)
    print(f"  • Eylem Yığını Boyutu (Chunk)  : {action_chunk.shape} ([16 Adım x 7-DoF])")
    print(f"  • İlk Eylem (Pick Yaklaşımı)   : {np.round(action_chunk[0, :3], 3)} m | Tutucu: {action_chunk[0, 6]:.2f} m")
    print(f"  • Son Eylem (Place Hedefi)     : {np.round(action_chunk[-1, :3], 3)} m | Tutucu: {action_chunk[-1, 6]:.2f} m")

    # -------------------------------------------------------------
    # ADIM 3: ROS2 DDS ve E-Stop Güvenlik Dağıtımı
    # -------------------------------------------------------------
    print("\n[3/4] ROS2 DDS Middleware ve E-Stop Güvenlik Katmanı Denetleniyor...")
    ros2_bridge = ROS2MiddlewareBridge()
    ros_res = ros2_bridge.publish_command(action=action_chunk[0], min_obstacle_dist=0.40, contact_force=11.5)
    print(f"  • ROS2 İletişim Durumu         : {ros_res['durum']}")
    print(f"  • E-Stop Güvenlik İhlali        : {ros_res['guvenlik_ihlal']} (Güvenli Operasyon)")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Büyük Final Teşhis Panosu
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli FAZ 13 Büyük Final Teşhis Panosu Oluşturuluyor...")
    profil_raporu = EmbodiedCapstoneProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "embodied_capstone_paneli.png")

    EmbodiedCapstoneGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ FAZ 13 Büyük Final Teşhis Panosu Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("🎉 FAZ 13 BÜYÜK FİNALİ (GÜN 241 - GÜN 260) %100 BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
