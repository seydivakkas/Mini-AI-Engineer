"""Günün Ana Çalıştırma Akışı: K-Means ile Baskın Renk Paleti Çıkarımı.

Bu betik; geometrik renk bloklarına sahip sentetik bir tekstil/halı deseni üreterek
K-Means ile en baskın 5 rengi ve yüzdesel ağırlıklarını çıkarır, görüntüyü kuantize eder
ve orantısal renk paleti şeridini disk üzerine PNG olarak kaydeder.
"""

import sys
from pathlib import Path

# Proje kök dizinini ekle
proje_kok = Path(__file__).resolve().parent
if str(proje_kok) not in sys.path:
    sys.path.insert(0, str(proje_kok))

import cv2
import numpy as np
from src.renk_kumeleyici import BaskinRenkCikarici
from src.palet_gorsellestirici import PaletGorsellestirici


def baslik(metin: str) -> None:
    """Rapor bölüm başlığı basar."""
    cizgi = "=" * 74
    print(f"\n{cizgi}\n>>> {metin}\n{cizgi}")


def sentetik_tekstil_deseni_uret() -> np.ndarray:
    """Net orantısal renk blokları içeren 300x300 BGR sentetik halı deseni üretir."""
    # 1. Taban Zemin: Gece Mavisi (BGR: 80, 30, 20) -> Yaklaşık %40
    resim = np.full((300, 300, 3), (80, 30, 20), dtype=np.uint8)

    # 2. Hardal Sarısı Büyük Daire (BGR: 30, 180, 220) -> Yaklaşık %25
    cv2.circle(resim, (150, 150), 95, (30, 180, 220), -1)

    # 3. Kiremit Kırmızısı Şeritler (BGR: 40, 60, 200) -> Yaklaşık %18
    cv2.rectangle(resim, (0, 0), (60, 300), (40, 60, 200), -1)
    cv2.rectangle(resim, (240, 0), (300, 300), (40, 60, 200), -1)

    # 4. Adaçayı Yeşili İç Motif (BGR: 60, 140, 80) -> Yaklaşık %12
    cv2.rectangle(resim, (100, 100), (200, 200), (60, 140, 80), -1)

    # 5. Krem / Fildişi Çizgiler ve Noktalar (BGR: 220, 240, 245) -> Yaklaşık %5
    cv2.circle(resim, (150, 150), 20, (220, 240, 245), -1)
    cv2.line(resim, (60, 150), (100, 150), (220, 240, 245), 4)
    cv2.line(resim, (200, 150), (240, 150), (220, 240, 245), 4)

    return resim


def main() -> None:
    baslik("AŞAMA 1: Sentetik Tekstil Görselinin Üretimi")
    desen_bgr = sentetik_tekstil_deseni_uret()
    h, w, c = desen_bgr.shape
    toplam_piksel = h * w
    print(f"[+] Çözünürlük         : {h} x {w} ({toplam_piksel:,} piksel)")
    print(f"[+] Kanal Sayısı       : {c} (BGR)")

    baslik("AŞAMA 2: K-Means (K=5) ile Baskın Renklerin Kümelenmesi")
    cikarici = BaskinRenkCikarici(k_kume_sayisi=5, rastgele_durum=42)
    palet = cikarici.paleti_cikar(desen_bgr)

    print(f"{'Sıra':<5} | {'HEX Kodu':<10} | {'RGB Değeri':<18} | {'Yüzde (%)':<10} | {'Piksel Adedi'}")
    print("-" * 74)
    toplam_yuzde = 0.0
    for idx, renk in enumerate(palet, 1):
        rgb_str = f"({renk.rgb[0]}, {renk.rgb[1]}, {renk.rgb[2]})"
        print(f"#{idx:<4} | {renk.hex_kodu:<10} | {rgb_str:<18} | %{renk.yuzde:<9.2f} | {renk.piksel_adedi:,}")
        toplam_yuzde += renk.yuzde
    print("-" * 74)
    print(f"[V] Toplam Ağırlık Tutarlılığı: %{toplam_yuzde:.2f}")

    baslik("AŞAMA 3: Görüntü Renk Kuantizasyonu (Color Quantization)")
    quantize_gorsel = cikarici.goruntuyu_quantize_et(desen_bgr)
    benzersiz_orijinal = len(np.unique(desen_bgr.reshape(-1, 3), axis=0))
    benzersiz_quantize = len(np.unique(quantize_gorsel.reshape(-1, 3), axis=0))
    print(f"[+] Orijinal Benzersiz Renk Adedi : {benzersiz_orijinal}")
    print(f"[+] Kuantize Benzersiz Renk Adedi : {benzersiz_quantize} (Tam 5 baskın renge indirgendi!)")

    baslik("AŞAMA 4: Görsel Rapor ve Renk Şeridinin Kaydedilmesi")
    cikti_yolu = proje_kok / "ciktilar" / "baskin_renk_paleti.png"
    kaydedilen = PaletGorsellestirici.palet_raporu_ciz(
        orijinal_bgr=desen_bgr,
        quantize_bgr=quantize_gorsel,
        palet=palet,
        dosya_yolu=cikti_yolu
    )
    print(f"[V] Renk paleti çizelgesi başarıyla kaydedildi: {kaydedilen.name}")
    print(f"[V] Kayıt Konumu: {kaydedilen}")
    print("\n[V] Day 11: Baskın Renk Paletinin Çıkarılması başarıyla tamamlandı.")


if __name__ == "__main__":
    main()
