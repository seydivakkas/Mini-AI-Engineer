"""
Çoklu Nesne Takibi Video Sahne Simülatörü.
Kesişen yörüngeler, kapanma (occlusion), gürültülü tespitler ve görsel kırpıntılar üretir.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import cv2


class VideoSahneSimulasyonu:
    """Yapay video dizisi ve gürültülü dedektör simülatörü."""

    def __init__(self, genislik: int = 512, yukseklik: int = 384, toplam_kare: int = 40, seed: int = 42):
        self.w = genislik
        self.h = yukseklik
        self.toplam_kare = toplam_kare
        self.seed = seed
        np.random.seed(seed)

    def uret_video_dizisi(self) -> List[Dict[str, Any]]:
        """
        4 farklı nesnenin hareket ettiği, ikisinin birbiriyle kesiştiği (kapanma yaşadığı)
        40 karelik video veri yapısı üretir.
        """
        # Hedef Nesne Tanımları: [id, baslangic_x, baslangic_y, vx, vy, w, h, renk_rgb]
        nesneler = [
            {"id": 1, "x": 40.0, "y": 140.0, "vx": 6.0, "vy": 0.5, "w": 30, "h": 60, "renk": (220, 50, 50)},     # Kırmızı (Soldan sağa)
            {"id": 2, "x": 440.0, "y": 150.0, "vx": -5.5, "vy": -0.2, "w": 28, "h": 58, "renk": (50, 80, 220)},   # Mavi (Sağdan sola - Çapraz geçiş)
            {"id": 3, "x": 180.0, "y": 60.0, "vx": 1.0, "vy": 4.5, "w": 32, "h": 64, "renk": (40, 200, 40)},     # Yeşil (Yukarıdan aşağı)
            {"id": 4, "x": 300.0, "y": 300.0, "vx": -1.2, "vy": -4.0, "w": 26, "h": 55, "renk": (220, 180, 40)},  # Sarı (Aşağıdan yukarı)
        ]

        video_kareleri = []

        for f_idx in range(self.toplam_kare):
            # Arka plan asfalt zemin
            kare_img = np.full((self.h, self.w, 3), 50, dtype=np.uint8)
            # Arka plan dokusu ve ızgara
            for gy in range(0, self.h, 40):
                cv2.line(kare_img, (0, gy), (self.w, gy), (60, 60, 60), 1)
            for gx in range(0, self.w, 40):
                cv2.line(kare_img, (gx, 0), (gx, self.h), (60, 60, 60), 1)

            kare_gt = []
            kare_tespitler = []
            kare_kirpintilar = []

            # Nesneleri güncelle ve çiz
            for obj in nesneler:
                obj["x"] += obj["vx"] + np.random.normal(0, 0.3)
                obj["y"] += obj["vy"] + np.random.normal(0, 0.3)

                x1 = int(obj["x"])
                y1 = int(obj["y"])
                x2 = int(obj["x"] + obj["w"])
                y2 = int(obj["y"] + obj["h"])

                # Ekran sınır kontrolü
                if x2 < 0 or x1 >= self.w or y2 < 0 or y1 >= self.h:
                    continue

                # Görsel çizimi
                cv2.rectangle(kare_img, (x1, y1), (x2, y2), obj["renk"], -1)
                cv2.circle(kare_img, (x1 + obj["w"] // 2, y1 + 10), 6, (230, 230, 230), -1)

                gt_box = [float(x1), float(y1), float(x2), float(y2)]
                kare_gt.append({"id": obj["id"], "box": gt_box})

                # Dedektör simülasyonu (%95 tespit olasılığı, hafif konum gürültüsü)
                if np.random.rand() < 0.95:
                    noisy_x1 = max(0, x1 + int(np.random.normal(0, 1.5)))
                    noisy_y1 = max(0, y1 + int(np.random.normal(0, 1.5)))
                    noisy_x2 = min(self.w - 1, x2 + int(np.random.normal(0, 1.5)))
                    noisy_y2 = min(self.h - 1, y2 + int(np.random.normal(0, 1.5)))

                    if noisy_x2 > noisy_x1 and noisy_y2 > noisy_y1:
                        crop = kare_img[noisy_y1:noisy_y2, noisy_x1:noisy_x2].copy()
                        kare_tespitler.append(np.array([noisy_x1, noisy_y1, noisy_x2, noisy_y2], dtype=np.float32))
                        kare_kirpintilar.append(crop)

            video_kareleri.append({
                "kare_no": f_idx,
                "gorsel_rgb": kare_img,
                "gt_hedefler": kare_gt,
                "tespitler": kare_tespitler,
                "kirpintilar": kare_kirpintilar
            })

        return video_kareleri
