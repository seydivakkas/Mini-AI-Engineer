"""
Day 54: Dijital Adli Bilişim, Error Level Analysis (ELA) ve Görsel Manipülasyon Tespiti Ana Yürütme Betiği.
"""

import os
import io
import cv2
import numpy as np
from PIL import Image
from src.ela_analizoru import ErrorLevelAnalizoru
from src.gurultu_adli_analizor import GurultuAdliAnalizoru
from src.adli_teftis_motoru import AdliTeftisMotoru
from src.gorsellestirici import AdliTeftisGorsellestirici


def manipule_edilmis_belge_simulasyonu(h: int = 400, w: int = 400) -> np.ndarray:
    """Homojen JPEG sıkıştırma dengesine sahip taban belgeye harici bir sahte damga ekler (Splicing/Tampering)."""
    np.random.seed(42)

    # 1. Taban Doğal Gradyanlı Arka Plan
    x = np.linspace(0, 10 * np.pi, w)
    y = np.linspace(0, 10 * np.pi, h)
    xx, yy = np.meshgrid(x, y)
    arka_plan = (np.sin(xx) * 30.0 + np.cos(yy) * 30.0 + 170.0).astype(np.uint8)
    taban_rgb = cv2.cvtColor(arka_plan, cv2.COLOR_GRAY2RGB)

    # Sensör Gürültüsü Simülasyonu
    gurultu = np.random.normal(0, 4, taban_rgb.shape)
    taban_gurultulu = np.clip(taban_rgb + gurultu, 0, 255).astype(np.uint8)

    # Taban görseli JPEG (Quality=80) olarak kaydedip dengeye getiriyoruz
    pil_taban = Image.fromarray(taban_gurultulu)
    tampon = io.BytesIO()
    pil_taban.save(tampon, format="JPEG", quality=80)
    tampon.seek(0)
    taban_jpeg = np.array(Image.open(tampon).convert("RGB"))

    # 2. Sahte Eklenmiş Bölge (Spliced Patch / Forged Stamp)
    # Bu bölge taban JPEG geçmişine sahip değildir (farklı sıkıştırma seviyesi)
    tampered_img = taban_jpeg.copy()

    # Sahte Damga 1: Yüksek kontrastlı mühür/imza
    cv2.circle(tampered_img, (260, 240), 38, (190, 40, 40), 3)
    cv2.putText(
        tampered_img, "APPROVED", (228, 245),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 40, 40), 1, cv2.LINE_AA
    )

    # Sahte Rakam 2: Tahrif edilmiş fatura tutarı
    cv2.rectangle(tampered_img, (100, 120), (190, 155), (255, 255, 255), -1)
    cv2.putText(
        tampered_img, "$950,000", (105, 145),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 30), 2, cv2.LINE_AA
    )

    return tampered_img


def main():
    print("=" * 85)
    print(">>> DAY 54: DİJİTAL ADLİ BİLİŞİM, ERROR LEVEL ANALYSIS (ELA) VE MANİPÜLASYON TESPİTİ")
    print("=" * 85)

    # 1. Sentetik Manipüle Edilmiş Adli Belge Üretimi
    print("\n[+] 1. Adım: Manipüle Edilmiş Belge (JPEG Splicing & Forgery) Hazırlanıyor...")
    aday_gorsel = manipule_edilmis_belge_simulasyonu(400, 400)
    print(f"    - Aday Görsel Boyutu: {aday_gorsel.shape[1]}x{aday_gorsel.shape[0]} px (RGB)")

    # 2. Adli Teftiş Motorunun Koşturulması
    print("\n[+] 2. Adım: Error Level Analysis (ELA) ve Sensör Gürültü Tutarsızlığı İnceleniyor...")
    motor = AdliTeftisMotoru(ela_kalite=90, z_esigi=2.2)
    teftis_sonuc = motor.teftis_et(aday_gorsel)

    print(f"    - ELA Ortalama Hata       : {teftis_sonuc['ela_istatistik']['ortalama_hata']:.2f} px")
    print(f"    - ELA Maksimum Hata       : {teftis_sonuc['ela_istatistik']['maks_hata']:.2f} px")
    print(f"    - Gürültü Tutarsızlık (CV): {teftis_sonuc['gurultu_tutarsizlik_cv']:.3f}")
    print(f"    - Tespit Edilen Şüpheli Bölge: {len(teftis_sonuc['supheli_bolgeler'])} Adet (%{teftis_sonuc['supheli_alan_orani']:.3f})")

    for b in teftis_sonuc["supheli_bolgeler"]:
        print(f"      * [Bölge #{b['id']}] Kutu: {b['kutu']} | Alan: {b['alan_px']}px | Bölge ELA: {b['bolge_ela_ort']}")

    print(f"\n    >>> MANİPÜLASYON GÜVEN SKORU: %{teftis_sonuc['manipulasyon_skoru']:.1f}")
    print(f"    >>> ADLİ BİLİŞİM KARARI     : {teftis_sonuc['karar']} (Risk: {teftis_sonuc['risk_seviyesi']})")
    print(f"    >>> AÇIKLAMA                : {teftis_sonuc['aciklama']}")

    # 3. 6 Panelli Teşhis Panosunun Çizilmesi
    print("\n" + "=" * 85)
    print(">>> 3. 6 PANELLİ ADLİ BİLİŞİM VE ELA TEŞHİS PANOSUNUN ÜRETİLMESİ")
    print("=" * 85)

    cikis_yolu = AdliTeftisGorsellestirici.panel_ciz(
        teftis_sonuc=teftis_sonuc,
        hedef_path="day-54-image-forensics-inspector/ciktilar/adli_teftis_paneli.png"
    )
    print(f"[+] 6 Panelli Teşhis Panosu Kaydedildi: {os.path.abspath(cikis_yolu)}")
    print("=" * 85)
    print("DAY 54: DİJİTAL ADLİ BİLİŞİM VE ELA ANALİZÖRÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 85)


if __name__ == "__main__":
    main()
