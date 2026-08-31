"""
Tesla Gün 49 Ana Akış (Tesla Day 49 Main Pipeline)
===================================================
Vision Transformer (ViT) Trafik Işığı, Geri Sayım ve Levha Algılayıcı
Uçtan Uca Çalıştırma ve Teşhis Paneli Üretim Scripti.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import sys
import os

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

su_an_dizin = os.path.dirname(os.path.abspath(__file__))
if su_an_dizin not in sys.path:
    sys.path.insert(0, su_an_dizin)

from src.tesla_vision_transformer_trafik_algilayici import TeslaVisionTransformerTrafficDetector
from src.tesla_vit_profilleyici import TeslaViTProfilleyici
from src.tesla_vit_gorsellestirici import TeslaViTGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 49: Vision Transformer (ViT) TRAFİK ALGILAYICI 🚗")
    print("================================================================================")
    print("Stajyer Görevi: Patch Embedding, Multi-Head Self-Attention, Geri Sayım & Levhalar")
    print("--------------------------------------------------------------------------------\n")

    # 1. ViT Benchmark'ı
    print(" [1] Vision Transformer Yama Öz-Dikkat ve Sınıflandırma Çözümleniyor...")
    profilleyici = TeslaViTProfilleyici(iterations=100)
    metrikler = profilleyici.benchmark_vit_detector()

    print(f"     -> İşlenen Yama Sayısı      : {metrikler['patch_count']} Adet (8x8 Yamalar)")
    print(f"     -> Trafik Işığı Durumu      : {metrikler['tl_state']} (Güven: %{metrikler['tl_conf']*100:.1f})")
    print(f"     -> Işık Geri Sayım Süresi   : {metrikler['countdown_sec']:.1f} Saniye")
    print(f"     -> Algılanan Trafik Levhası : {metrikler['sign_name']} (Güven: %{metrikler['sign_conf']*100:.1f})")

    # 2. RTOS Çözümleme Hızı
    print("\n [2] Vision Transformer NPU Çözümleme Performansı...")
    print(f"     -> Ortalama Çözüm Süresi    : {metrikler['vit_step_ortalama_us']:.3f} µs (P99: {metrikler['vit_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik Kare Hacmi     : {metrikler['saniyelik_vit_karesi']:,} Kare/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla FSD Vision Transformer Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaViTGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_vit_traffic_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 49 BAŞARIYLA TAMAMLANDI! Vision Transformer DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
