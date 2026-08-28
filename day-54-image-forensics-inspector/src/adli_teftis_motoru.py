"""
Dijital Adli Bilişim ve Manipülasyon Tespit Motoru (Forensic Multi-Metric Decision Engine).
"""

from typing import Dict, Any, List
import cv2
import numpy as np
from .ela_analizoru import ErrorLevelAnalizoru
from .gurultu_adli_analizor import GurultuAdliAnalizoru


class AdliTeftisMotoru:
    """ELA ve Sensör Gürültü Tutarsızlığını birleştirerek sahtecilik ve görsel manipülasyonu tespit eder."""

    def __init__(self, ela_kalite: int = 90, z_esigi: float = 2.5):
        self.ela_kalite = ela_kalite
        self.z_esigi = z_esigi

    def teftis_et(self, img_rgb: np.ndarray) -> Dict[str, Any]:
        """Uçtan uca adli bilişim analizi yaparak şüpheli bölgeleri kutular ve manipülasyon skorunu üretir."""
        h, w = img_rgb.shape[:2]

        # 1. ELA Analizi
        ela_rgb, fark_gri, ela_istatistik = ErrorLevelAnalizoru.ela_hesapla(
            img_rgb, kalite=self.ela_kalite, olcek_carpani=15.0
        )
        anomali_maskesi = ErrorLevelAnalizoru.anomali_maskesi_uret(fark_gri, z_esik=self.z_esigi)

        # 2. Sensör Gürültü Kalıntısı ve Lokal Varyans Analizi
        kalinti, kalinti_norm = GurultuAdliAnalizoru.gurultu_kalintisi_hesapla(img_rgb, filtre_ksize=3)
        lokal_varyans, gürültü_cv = GurultuAdliAnalizoru.lokal_gurultu_varyansi_haritasi(kalinti, blok_boyutu=16)

        # 3. Morfolojik Maske Temizleme ve Kontur Çıkarma
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        temiz_maske = cv2.morphologyEx(anomali_maskesi, cv2.MORPH_CLOSE, kernel)
        temiz_maske = cv2.morphologyEx(temiz_maske, cv2.MORPH_OPEN, kernel)

        konturlar, _ = cv2.findContours(temiz_maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        supheli_bolgeler: List[Dict[str, Any]] = []
        anotasyonlu_img = img_rgb.copy()
        toplam_supheli_alan = 0

        for idx, cnt in enumerate(konturlar):
            alan = cv2.contourArea(cnt)
            if alan >= 40:  # Minimum şüpheli alan eşiği
                x, y, bw, bh = cv2.boundingRect(cnt)
                toplam_supheli_alan += int(alan)

                # Bölge içi lokal ELA ve gürültü ortalaması
                bolge_ela = float(np.mean(fark_gri[y:y+bh, x:x+bw]))
                bolge_varyans = float(np.mean(lokal_varyans[y:y+bh, x:x+bw]))

                supheli_bolgeler.append({
                    "id": idx + 1,
                    "kutu": (x, y, bw, bh),
                    "alan_px": int(alan),
                    "bolge_ela_ort": round(bolge_ela, 2),
                    "bolge_varyans_ort": round(bolge_varyans, 2)
                })

                # Görsel üzerine kırmızı kutu ve etiket çizimi
                cv2.rectangle(anotasyonlu_img, (x, y), (x + bw, y + bh), (231, 76, 60), 2)
                cv2.putText(
                    anotasyonlu_img, f"FORGERY:{int(alan)}px", (x, max(y - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (231, 76, 60), 1, cv2.LINE_AA
                )

        supheli_alan_orani = (toplam_supheli_alan / (h * w)) * 100.0

        # 4. Sahtecilik Güven Skoru Hesabı (0 - 100)
        # Formül: ELA Z-skor anomali oranı + Gürültü tutarsızlık katsayısı + Şüpheli bölge sayısı
        skor_bilesen_alan = min(40.0, supheli_alan_orani * 15.0)
        skor_bilesen_gurultu = min(35.0, max(0.0, (gürültü_cv - 0.45) * 60.0))
        skor_bilesen_kontur = min(25.0, len(supheli_bolgeler) * 8.0)

        manipulasyon_skoru = float(round(min(100.0, skor_bilesen_alan + skor_bilesen_gurultu + skor_bilesen_kontur), 1))

        # 5. Adli Karar
        if manipulasyon_skoru < 25.0 and len(supheli_bolgeler) == 0:
            karar = "ORİJİNAL (AUTHENTIC)"
            risk_seviyesi = "LOW"
            karar_renk = "#2ecc71"
            aciklama = "Doğal homojen JPEG sıkıştırma dengesi ve sensör gürültü sürekliliği tespit edildi."
        elif manipulasyon_skoru < 60.0:
            karar = "ŞÜPHELİ (SUSPICIOUS)"
            risk_seviyesi = "WARNING"
            karar_renk = "#f39c12"
            aciklama = "Hafif ELA dengesizliği veya bölgesel gürültü varyans kayması tespit edildi."
        else:
            karar = "MANİPÜLE EDİLMİŞ (TAMPERED)"
            risk_seviyesi = "CRITICAL_REJECT"
            karar_renk = "#e74c3c"
            aciklama = "Belirgin ELA hata farkı ve kopyala-yapıştır / splicing manipülasyonu tespit edildi."

        return {
            "manipulasyon_skoru": manipulasyon_skoru,
            "karar": karar,
            "risk_seviyesi": risk_seviyesi,
            "karar_renk": karar_renk,
            "aciklama": aciklama,
            "supheli_bolgeler": supheli_bolgeler,
            "supheli_alan_orani": round(supheli_alan_orani, 3),
            "gurultu_tutarsizlik_cv": gürültü_cv,
            "ela_istatistik": ela_istatistik,
            "ela_rgb": ela_rgb,
            "fark_gri": fark_gri,
            "kalinti_norm": kalinti_norm,
            "lokal_varyans": lokal_varyans,
            "ikili_maske": temiz_maske,
            "anotasyonlu_gorsel": anotasyonlu_img
        }
