"""
Day 248: VLM Destekli Semantik SLAM ve Doğal Dil ile Otonom Navigasyon Ana Akışı.
"""

import os
import sys
import numpy as np

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.semantic_slam_motoru import (
    OccupancyGridMap,
    VLMSemanticAnchor,
    AStarPathPlanner,
    SemanticSLAMSystem,
)
from src.slam_profilleyici import SemanticSLAMProfilleyici
from src.gorsellestirici import SemanticSLAMGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 248 (FAZ 13): VLM DESTEKLİ SEMANTİK SLAM VE DOĞAL DİL İLE OTONOM İÇ MEKAN NAVİGASYONU")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: 2D Doluluk Izgarası ve Güvenlik Şişirme Katmanı
    # -------------------------------------------------------------
    print("\n[1/4] 2D Doluluk Izgarası ve Güvenlik Maliyet Haritası (Costmap) Hesaplanıyor...")
    slam = SemanticSLAMSystem(W=50, H=50)
    costmap = slam.harita.compute_inflation_costmap(guvenlik_yaricapi=2)
    print(f"  • Harita Boyutları            : {slam.harita.W}x{slam.harita.H} hücre ({slam.harita.W*0.1}m x {slam.harita.H*0.1}m)")
    print(f"  • Robot Başlangıç Konumu      : {slam.robot_pos}")
    print(f"  • Statik Duvar/Engel Hücreleri: {int(np.sum(slam.harita.izgara > 0.5))} adet")

    # -------------------------------------------------------------
    # ADIM 2: Doğal Dil Komutunun VLM ile Semantik Ankrajı
    # -------------------------------------------------------------
    print("\n[2/4] Doğal Dil Komutu VLM ile 3D Semantik Yer İmine Eşleniyor...")
    sorgu = "masanın üzerindeki kırmızı kahve kupasını bul ve git"
    grounding = slam.vlm.ground_language_query(sorgu)
    print(f"  🗣️ Kullanıcı Komutu : '{sorgu}'")
    print(f"  🎯 Eşleşen Nesne    : '{grounding['eslesen_nesne']['etiket']}' (ID: {grounding['eslesen_nesne']['id']})")
    print(f"  📍 Hedef Koordinat  : {grounding['hedef_koordinat']}")
    print(f"  ⭐ Güven Skoru      : %{grounding['guven_skoru']*100:.1f}")

    # -------------------------------------------------------------
    # ADIM 3: A* ile Engellerden Kaçınan Güvenli Rotalama
    # -------------------------------------------------------------
    print("\n[3/4] A* Optimum Çarpışmasız Yörünge Planlanıyor...")
    nav_sonuc = slam.navigate_with_language(sorgu)
    print(f"  ✓ Üretilen Rota Nokta Sayısı: {nav_sonuc['yol_nokta_sayisi']} adım")
    print(f"  ✓ Rota Özeti: {nav_sonuc['yol_koordinatlari'][:3]} ... -> {nav_sonuc['yol_koordinatlari'][-1]}")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Semantik SLAM Teşhis Panosu Oluşturuluyor...")
    profil_raporu = SemanticSLAMProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "semantic_slam_paneli.png")

    SemanticSLAMGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Semantik SLAM Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 248 (FAZ 13): VLM SEMANTİK SLAM VE NAVİGASYON MODÜLÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
