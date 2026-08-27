"""Günün Ana Çalıştırma Akışı: OpenCV Temel Görüntü İşleme Araç Seti.

Bu betik; endüstriyel elektronik devre kartı ve geometrik desenli sentetik bir görsel
üreterek Gauss filtreleme, Sobel kenar çıkarımı ve morfolojik gürültü temizleme (Açma/Kapatma)
işlemlerini baştan sona çalıştırır ve 9 panelli görsel raporu diske kaydeder.
"""

import sys
from pathlib import Path

# Proje kök dizinini ekle
proje_kok = Path(__file__).resolve().parent
if str(proje_kok) not in sys.path:
    sys.path.insert(0, str(proje_kok))

import cv2
import numpy as np
from src.filtreler import KonvolusyonFiltresi, GaussBulaniklastirici, SobelKenarTespitEdici
from src.morfoloji import MorfolojikIslemci
from src.gorsellestirici import IslemePaneliUreteci


def baslik(metin: str) -> None:
    """Rapor bölüm başlığı basar."""
    cizgi = "=" * 74
    print(f"\n{cizgi}\n>>> {metin}\n{cizgi}")


def sentetik_devre_karti_uret() -> np.ndarray:
    """Görüntü işleme filtrelerini test etmek için geometrik bir devre deseni üretir."""
    resim = np.full((256, 256), 40, dtype=np.uint8)

    # 1. Ana çip gövdesi (Dikdörtgen)
    cv2.rectangle(resim, (50, 50), (140, 140), 200, -1)

    # 2. İletken bakır hatlar (Çizgiler)
    cv2.line(resim, (140, 70), (220, 70), 180, 4)
    cv2.line(resim, (140, 120), (220, 120), 180, 4)
    cv2.line(resim, (95, 140), (95, 220), 180, 4)

    # 3. Yuvarlak lehim pedleri (Daireler)
    cv2.circle(resim, (220, 70), 10, 240, -1)
    cv2.circle(resim, (220, 120), 10, 240, -1)
    cv2.circle(resim, (95, 220), 10, 240, -1)

    # 4. Kusur 1: Çip gövdesi içinde küçük siyah delikler (Closing ile kapanacak)
    resim[70:75, 70:75] = 40
    resim[110:114, 100:104] = 40

    # 5. Kusur 2: Arka planda istenmeyen toz zerrecikleri / izole beyaz noktalar (Opening ile gidecek)
    np.random.seed(42)
    gurultu_x = np.random.randint(10, 240, size=30)
    gurultu_y = np.random.randint(10, 240, size=30)
    for x, y in zip(gurultu_x, gurultu_y):
        if resim[y, x] == 40:
            resim[y, x] = 230

    return resim


def main() -> None:
    baslik("AŞAMA 1: Sentetik Endüstriyel Görüntünün Üretimi")
    orijinal = sentetik_devre_karti_uret()
    print(f"[+] Görsel Boyutu      : {orijinal.shape} (Yükseklik x Genişlik)")
    print(f"[+] Piksel Veri Tipi   : {orijinal.dtype}")
    print(f"[+] Ortalama Parlaklık : {np.mean(orijinal):.2f}")

    baslik("AŞAMA 2: Gauss Bulanıklaştırma ve Keskinleştirme Konvolüsyonu")
    gauss_resim = GaussBulaniklastirici.bulaniklastir(orijinal, cekirdek_boyutu=(5, 5), sigma_x=1.2)

    # Keskinleştirme (Sharpening) Çekirdeği:
    #  [ 0, -1,  0]
    #  [-1,  5, -1]
    #  [ 0, -1,  0]
    keskinlik_cekirdegi = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    keskin_resim = KonvolusyonFiltresi.ozel_cekirdek_uygula(orijinal, keskinlik_cekirdegi)

    print("[V] 5x5 Gauss Yumuşatması başarıyla uygulandı (Yüksek frekanslı gürültü bastırıldı).")
    print("[V] 3x3 Özel Keskinleştirme Konvolüsyonu uygulandı.")

    baslik("AŞAMA 3: Sobel Gradyan ve Kenar Büyüklüğü Analizi")
    gx, gy, magnitut = SobelKenarTespitEdici.gradyan_hesapla(gauss_resim, cekirdek_boyutu=3)
    print(f"[+] Yatay Kenar (Gx) Max Değer    : {np.max(gx)}")
    print(f"[+] Dikey Kenar (Gy) Max Değer    : {np.max(gy)}")
    print(f"[+] Birleşik Büyüklük (G) Max     : {np.max(magnitut)}")

    baslik("AŞAMA 4: Matematiksel Morfoloji Operasyonları (Açma, Kapatma, Gradyan)")
    # İkili (Binary) eşikleme ile nesne maskesi çıkarma
    _, ikili_maske = cv2.threshold(orijinal, 120, 255, cv2.THRESH_BINARY)

    cekirdek_3x3 = MorfolojikIslemci.yapisal_element_olustur((3, 3), "dikdortgen")
    
    # 1. Açma (Opening): Arka plandaki beyaz toz zerreciklerini temizler
    acma_resim = MorfolojikIslemci.acma(ikili_maske, cekirdek_3x3)

    # 2. Kapatma (Closing): Çip gövdesi içindeki siyah oyukları tıkar
    kapatma_resim = MorfolojikIslemci.kapatma(ikili_maske, cekirdek_3x3)

    # 3. Morfolojik Gradyan: Nesnenin net dış kontur çizgisini çıkarır
    morf_gradyan = MorfolojikIslemci.morfolojik_gradyan(ikili_maske, cekirdek_3x3)

    print("[V] Açma (Opening)   : Arka plan beyaz gürültüleri yok edildi.")
    print("[V] Kapatma (Closing): Çip içi mikro siyah delikler dolduruldu.")
    print("[V] Morf. Gradyan    : Dış çevre konturları ayrıştırıldı.")

    baslik("AŞAMA 5: 9 Panelli Görsel Raporun Diske Kaydedilmesi")
    paneller = {
        "1. Orijinal Görüntü": orijinal,
        "2. Gauss Yumuşatma (5x5)": gauss_resim,
        "3. Keskinleştirilmiş": keskin_resim,
        "4. Sobel Yatay (Gx)": gx,
        "5. Sobel Dikey (Gy)": gy,
        "6. Sobel Büyüklüğü (G)": magnitut,
        "7. Gürültülü İkili Maske": ikili_maske,
        "8. Açma (Gürültü Silindi)": acma_resim,
        "9. Kapatma (Delikler Tıkandı)": kapatma_resim,
    }

    cikti_yolu = proje_kok / "ciktilar" / "goruntu_isleme_paneli.png"
    kaydedilen = IslemePaneliUreteci.panel_olustur_ve_kaydet(paneller, cikti_yolu)
    print(f"[V] Görsel panel başarıyla üretildi: {kaydedilen.name}")
    print(f"[V] Kayıt Konumu: {kaydedilen}")
    print("\n[V] Day 8: OpenCV Tabanlı Temel Görüntü İşleme Araç Seti tamamlandı.")


if __name__ == "__main__":
    main()
