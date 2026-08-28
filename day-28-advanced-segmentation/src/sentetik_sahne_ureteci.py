"""
Sentetik Çoklu Nesne Sahne Üreteci:
Örtüşen nesneler, semantik arka plan, örnek maskeleri ve panoptik harita üretimi.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import cv2
from .bolutleme_turleri import PanoptikDonusturucu


class SentetikSahneUreteci:
    """
    Otonom sürüş ve robotik simülasyonu için semantik arka plan (Stuff)
    ve örtüşen nesneler (Things) içeren zengin sahneler üretir.
    """

    def __init__(self, img_size: int = 256, seed: int = 42):
        self.img_size = img_size
        self.seed = seed
        np.random.seed(seed)

    def sahne_uret(self, num_instances: int = 5) -> Dict[str, Any]:
        """
        Tek bir çoklu nesneli sahne ve tüm bölütleme formatlarındaki etiketleri üretir.
        """
        h = self.img_size
        w = self.img_size

        # 1. RGB Görsel ve Semantik Zemin Başlat
        gorsel = np.zeros((h, w, 3), dtype=np.uint8)
        semantik = np.zeros((h, w), dtype=np.int64)

        # Ufuk Çizgisi: Üst taraf Gökyüzü (Stuff: 0), Alt taraf Yol (Stuff: 1)
        horizon = int(h * 0.45)
        # Gökyüzü (Mavimsi degrade)
        for y in range(horizon):
            ratio = y / horizon
            gorsel[y, :] = [int(220 - 40 * ratio), int(180 - 30 * ratio), int(120 + 50 * ratio)]
            semantik[y, :] = 0  # Gökyüzü

        # Yol (Koyu gri asfalt + şeritler)
        for y in range(horizon, h):
            gorsel[y, :] = [60, 60, 65]
            semantik[y, :] = 1  # Yol

        # Yol Şeritleri (Beyaz çizgiler)
        cv2.line(gorsel, (w // 2, horizon), (w // 2 - 40, h), (240, 240, 240), 2)
        cv2.line(gorsel, (w // 2, horizon), (w // 2 + 40, h), (240, 240, 240), 2)

        # 2. Nesne Örnekleri (Things) Ekle
        instance_masks = []
        instance_boxes = []
        instance_labels = []

        for inst_id in range(num_instances):
            obj_type = np.random.choice([2, 3, 4])  # 2: Araç, 3: Yaya, 4: Engel

            if obj_type == 2:  # Araç (Dikdörtgen gövde)
                bw = np.random.randint(40, 70)
                bh = np.random.randint(25, 45)
                x1 = np.random.randint(10, w - bw - 10)
                y1 = np.random.randint(horizon + 10, h - bh - 10)
                x2 = x1 + bw
                y2 = y1 + bh

                color = (np.random.randint(40, 230), np.random.randint(40, 230), np.random.randint(180, 255))
                cv2.rectangle(gorsel, (x1, y1), (x2, y2), color, -1)
                # Cam ve Farlar
                cv2.rectangle(gorsel, (x1 + 5, y1 + 5), (x2 - 5, y1 + 15), (200, 230, 255), -1)
                cv2.circle(gorsel, (x1 + 8, y2 - 6), 4, (30, 30, 30), -1)
                cv2.circle(gorsel, (x2 - 8, y2 - 6), 4, (30, 30, 30), -1)

                inst_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.rectangle(inst_mask, (x1, y1), (x2, y2), 1, -1)

            elif obj_type == 3:  # Yaya (Elips gövde + kafa)
                bh = np.random.randint(35, 55)
                bw = np.random.randint(16, 26)
                cx = np.random.randint(20, w - 20)
                cy = np.random.randint(horizon + 20, h - bh // 2 - 10)
                x1 = max(0, cx - bw // 2)
                x2 = min(w - 1, cx + bw // 2)
                y1 = max(0, cy - bh // 2)
                y2 = min(h - 1, cy + bh // 2)

                color = (np.random.randint(180, 240), np.random.randint(60, 140), np.random.randint(60, 140))
                cv2.ellipse(gorsel, (cx, cy + 5), (bw // 2, bh // 2 - 8), 0, 0, 360, color, -1)
                cv2.circle(gorsel, (cx, y1 + 7), 6, (220, 200, 180), -1)

                inst_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.ellipse(inst_mask, (cx, cy + 5), (bw // 2, bh // 2 - 8), 0, 0, 360, 1, -1)
                cv2.circle(inst_mask, (cx, y1 + 7), 6, 1, -1)

            else:  # Engel / Koni (Üçgen Poligon)
                bw = np.random.randint(20, 35)
                bh = np.random.randint(25, 40)
                x1 = np.random.randint(15, w - bw - 15)
                y1 = np.random.randint(horizon + 15, h - bh - 10)
                x2 = x1 + bw
                y2 = y1 + bh
                pts = np.array([[x1 + bw // 2, y1], [x1, y2], [x2, y2]], dtype=np.int32)

                color = (0, 140, 255)  # Turuncu
                cv2.fillPoly(gorsel, [pts], color)
                cv2.line(gorsel, (x1 + 4, y1 + bh // 2), (x2 - 4, y1 + bh // 2), (255, 255, 255), 3)

                inst_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(inst_mask, [pts], 1)

            # Semantik haritayı nesne sınıfıyla güncelle
            semantik[inst_mask > 0] = obj_type
            instance_masks.append(inst_mask.astype(np.float32))
            instance_boxes.append([x1, y1, x2, y2])
            instance_labels.append(obj_type)

        # 3. Panoptik Haritayı İnşa Et
        panoptik_harita, segments = PanoptikDonusturucu.birlestir_panoptik(
            semantik_harita=semantik,
            ornek_maskeleri=instance_masks,
            ornek_siniflari=instance_labels
        )

        return {
            "gorsel_rgb": gorsel,
            "semantik_harita": semantik,
            "ornek_maskeleri": instance_masks,
            "ornek_kutulari": instance_boxes,
            "ornek_siniflari": instance_labels,
            "panoptik_harita": panoptik_harita,
            "panoptik_segmentler": segments
        }
