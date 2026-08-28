"""
OpenCV Morfolojik Filtreler ve Kontur Tabanlı Kusur Tespit Motoru (Rule-Based Defect Detector).
"""

from typing import Dict, Any, List, Tuple
import cv2
import numpy as np


class MorfolojikKusurDedektoru:
    """Yüzey lekeleri, iplik çekikleri, delikler ve çizikleri morfolojik operasyonlar ve kontur analiziyle tespit eder."""

    def __init__(self, min_kusur_alani: int = 20, max_kusur_alani: int = 15000):
        self.min_kusur_alani = min_kusur_alani
        self.max_kusur_alani = max_kusur_alani

    def kusurlari_tespit_et(
        self,
        img_rgb: np.ndarray,
        kernel_boyutu: int = 9,
        esik_degeri: int = 35
    ) -> Dict[str, Any]:
        """Morfolojik Top-Hat/Black-Hat filtreleri ve kontur geometrisi ile kusurları ayrıştırır."""
        h, w = img_rgb.shape[:2]
        gri = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) if len(img_rgb.shape) == 3 else img_rgb.copy()

        # 1. Dokusal Arka Plan Gürültüsünü Bastırma
        gri_flulaştırılmış = cv2.GaussianBlur(gri, (5, 5), 0)

        # 2. Morfolojik Top-Hat (Parlak Kusurlar) ve Black-Hat (Koyu Lekeler)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_boyutu, kernel_boyutu))
        top_hat = cv2.morphologyEx(gri_flulaştırılmış, cv2.MORPH_TOPHAT, kernel)
        black_hat = cv2.morphologyEx(gri_flulaştırılmış, cv2.MORPH_BLACKHAT, kernel)

        kusur_haritasi = cv2.add(top_hat, black_hat)

        # 3. İkili (Binary) Eşikleme ve Morfolojik Kapanış (Closing)
        _, binary_mask = cv2.threshold(kusur_haritasi, esik_degeri, 255, cv2.THRESH_BINARY)
        kernel_temizlik = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_temizlik)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_temizlik)

        # 4. Kontur Tespiti ve Geometrik Sınıflandırma
        konturlar, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        tespit_edilen_kusurlar = []
        cizim_rgb = img_rgb.copy()
        toplam_kusur_alani = 0

        for idx, cnt in enumerate(konturlar):
            alan = cv2.contourArea(cnt)
            if self.min_kusur_alani <= alan <= self.max_kusur_alani:
                x, y, kw, kh = cv2.boundingRect(cnt)
                cevre = cv2.arcLength(cnt, True)
                dairesellik = (4 * np.pi * alan) / (cevre**2) if cevre > 0 else 0.0
                en_boy_orani = kw / max(kh, 1)

                # Kusur tipi çıkarımı
                if dairesellik > 0.55:
                    kusur_tipi = "LEKE_DELIK"
                    renk = (231, 76, 60)  # Kırmızı
                elif en_boy_orani > 2.5 or en_boy_orani < 0.4:
                    kusur_tipi = "CIZIK_IPLIK_CEKIGI"
                    renk = (230, 126, 34)  # Turuncu
                else:
                    kusur_tipi = "YUZEY_ANOMALISI"
                    renk = (155, 89, 182)  # Mor

                toplam_kusur_alani += int(alan)
                tespit_edilen_kusurlar.append({
                    "id": idx + 1,
                    "tip": kusur_tipi,
                    "alan_px": int(alan),
                    "kutu": (x, y, kw, kh),
                    "dairesellik": float(round(dairesellik, 3)),
                    "en_boy_orani": float(round(en_boy_orani, 3))
                })

                # Görsel üzerine kutu ve etiket çizimi
                cv2.rectangle(cizim_rgb, (x, y), (x + kw, y + kh), renk, 2)
                cv2.putText(
                    cizim_rgb, f"{kusur_tipi[:7]}:{int(alan)}", (x, max(y - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, renk, 1, cv2.LINE_AA
                )

        kusur_orani = float(round((toplam_kusur_alani / (h * w)) * 100.0, 3))
        kusurlu_mu = len(tespit_edilen_kusurlar) > 0

        # Kalite Puanı (100 üzerinden)
        kalite_puani = max(0.0, float(round(100.0 - (kusur_orani * 20.0) - (len(tespit_edilen_kusurlar) * 10.0), 1)))

        return {
            "kusurlu_mu": kusurlu_mu,
            "kusur_sayisi": len(tespit_edilen_kusurlar),
            "toplam_kusur_alani": toplam_kusur_alani,
            "kusur_orani_yuzde": kusur_orani,
            "kalite_puani": kalite_puani,
            "kusurlar": tespit_edilen_kusurlar,
            "binary_mask": binary_mask,
            "anotasyonlu_gorsel": cizim_rgb
        }
