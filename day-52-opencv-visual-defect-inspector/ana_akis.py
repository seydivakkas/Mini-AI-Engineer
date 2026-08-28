"""
Day 52: OpenCV ile Kural Tabanlı Görsel Kusur & Bulanıklık Tespiti Ana Yürütme Betiği.
"""

import os
import cv2
import numpy as np
from src.bulaniklik_analizoru import BulaniklikAnalizoru
from src.kusur_tespit_motoru import MorfolojikKusurDedektoru
from src.gorsellestirici import KusurTeftisGorsellestirici


def kusurlu_dokuma_simulasyonu(genislik: int = 400, yukseklik: int = 400) -> np.ndarray:
    """Dokuma tekstil arka planı ve üzerine eklenmiş gerçekçi kusur anomalileri üretir."""
    np.random.seed(42)

    # 1. Düzenli Dokuma Tekstil Deseni
    x = np.linspace(0, 40 * np.pi, genislik)
    y = np.linspace(0, 40 * np.pi, yukseklik)
    xx, yy = np.meshgrid(x, y)
    dokuma = (np.sin(xx) * np.cos(yy) * 35.0 + 160.0).astype(np.uint8)

    img_rgb = cv2.cvtColor(dokuma, cv2.COLOR_GRAY2RGB)

    # 2. Kusur 1: Koyu Yağ Lekesi / Delik (Dairesel)
    cv2.circle(img_rgb, (120, 150), 18, (30, 20, 20), -1)

    # 3. Kusur 2: Parlak İplik Çekiği / Çizik (Doğrusal)
    cv2.line(img_rgb, (260, 80), (320, 320), (255, 240, 220), 4)

    # 4. Kusur 3: Küçük Yüzey Aşınması
    cv2.rectangle(img_rgb, (200, 260), (235, 290), (40, 40, 50), -1)

    return img_rgb


def main():
    print("=" * 85)
    print(">>> DAY 52: OPENCV İLE KURAL TABANLI GÖRSEL KUSUR & BULANIKLIK TESPİTİ")
    print("=" * 85)

    # 1. Sentetik Dokuma ve Kusur Görüntüsü Üretimi
    img_ornek = kusurlu_dokuma_simulasyonu(400, 400)
    print(f"[+] Test Görüntüsü Hazırlandı: {img_ornek.shape[1]}x{img_ornek.shape[0]} RGB Dokuma Yüzeyi")

    # 2. Bulanıklık ve Frekans Spektrumu Analizi
    print("\n[+] 1. Adım: Laplacian Varyansı ve 2D FFT Frekans Analizi...")
    bulaniklik_sonuc = BulaniklikAnalizoru.analiz_et(img_ornek, laplacian_esigi=80.0, fft_esigi=0.05)
    print(f"    - Laplacian Varyansı (FM)   : {bulaniklik_sonuc['laplacian_varyansi']:.2f}")
    print(f"    - 2D FFT Yüksek Frekans (HFR): %{bulaniklik_sonuc['fft_yuksek_frekans_orani']:.2f}")
    print(f"    - Tenengrad Odak Skoru      : {bulaniklik_sonuc['tenengrad_skoru']:.2f}")
    print(f"    - Netlik Kararı             : {bulaniklik_sonuc['karar']}")

    # 3. Morfolojik Filtreleme ve Kontur Kusur Tespiti
    print("\n[+] 2. Adım: Morfolojik Top-Hat / Black-Hat ve Kontur Ayrıştırması...")
    dedektor = MorfolojikKusurDedektoru(min_kusur_alani=20, max_kusur_alani=15000)
    kusur_sonuc = dedektor.kusurlari_tespit_et(img_ornek, kernel_boyutu=11, esik_degeri=40)

    print(f"    - Tespit Edilen Kusur Sayısı: {kusur_sonuc['kusur_sayisi']} Adet")
    print(f"    - Toplam Kusurlu Alan       : {kusur_sonuc['toplam_kusur_alani']} piksel (%{kusur_sonuc['kusur_orani_yuzde']:.3f})")
    print(f"    - Kalite Puanı              : {kusur_sonuc['kalite_puani']:.1f} / 100")

    for k in kusur_sonuc["kusurlar"]:
        print(f"      * [Kusur #{k['id']}] Tip: {k['tip']:<18} Alan: {k['alan_px']:>4}px  Kutu: {k['kutu']}")

    # 4. 6 Panelli Teşhis Panosunun Çizilmesi
    print("\n" + "=" * 85)
    print(">>> 3. 6 PANELLİ AOI KUSUR TEFTİŞ VE BULANIKLIK PANOSUNUN ÜRETİLMESİ")
    print("=" * 85)

    cikis_yolu = KusurTeftisGorsellestirici.panel_ciz(
        bulaniklik_sonuc=bulaniklik_sonuc,
        kusur_sonuc=kusur_sonuc,
        hedef_path="day-52-opencv-visual-defect-inspector/ciktilar/kusur_teftis_paneli.png"
    )
    print(f"[+] 6 Panelli Teşhis Panosu Kaydedildi: {os.path.abspath(cikis_yolu)}")
    print("=" * 85)
    print("DAY 52: OPENCV GÖRSEL KUSUR VE BULANIKLIK TESPİTİ PAKETİ BAŞARIYLA TAMAMLANDI!")
    print("=" * 85)


if __name__ == "__main__":
    main()
