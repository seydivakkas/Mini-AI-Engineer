"""Albumentations ile İleri Düzey Görsel Dönüştürme Modülü.

Bu modül; hızlı C++ tabanlı OpenCV çekirdeğini kullanan Albumentations kütüphanesi ile
geometrik (döndürme, ölçekleme, kaydırma), fotometrik (renk, parlaklık, kontrast) ve
bölgesel silme (CoarseDropout) dönüşümlerini yönetir.
"""

from typing import Dict, List, Optional, Tuple, Union
import albumentations as A
import cv2
import numpy as np


class AlbumentationsDonusturucu:
    """Albumentations dönüşüm boru hatlarını (pipelines) yöneten sınıf."""

    def __init__(self, hedef_boyut: Tuple[int, int] = (64, 64)) -> None:
        """Albumentations dönüştürücüsünü ilklendirir."""
        self.hedef_boyut = hedef_boyut
        H, W = hedef_boyut

        # 1. Temel Geometrik ve Fotometrik Dönüşümler
        self.temel_pipeline = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, border_mode=cv2.BORDER_REFLECT, p=0.6),
        ])

        # 2. Ağır / Endüstriyel Üretim Seviyesi Dönüşümler
        self.agir_pipeline = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=35, border_mode=cv2.BORDER_REFLECT, p=0.7),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.6),
            A.MotionBlur(blur_limit=3, p=0.3),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(4, 12),
                hole_width_range=(4, 12),
                p=0.5,
            ),
        ])

    def donustur_tekil(
        self, gorsel_rgb: np.ndarray, mod: str = "agir"
    ) -> np.ndarray:
        """Tek bir RGB görseli belirtilen modda dönüştürür.

        Args:
            gorsel_rgb: (H, W, 3) float32 [0.0, 1.0] veya uint8 [0, 255] NumPy dizisi.
            mod: 'temel' veya 'agir'.

        Returns:
            np.ndarray: Dönüştürülmüş görsel (girdiyle aynı veri tipi ve boyutta).
        """
        is_float = gorsel_rgb.dtype == np.float32 or gorsel_rgb.dtype == np.float64
        if is_float and gorsel_rgb.max() <= 1.0:
            gorsel_uint8 = np.uint8(np.clip(gorsel_rgb * 255.0, 0, 255))
        else:
            gorsel_uint8 = gorsel_rgb.astype(np.uint8)

        pipeline = self.agir_pipeline if mod == "agir" else self.temel_pipeline
        sonuc = pipeline(image=gorsel_uint8)["image"]

        if is_float:
            return (sonuc / 255.0).astype(np.float32)
        return sonuc

    def donustur_toplu(
        self, gorseller_rgb: np.ndarray, mod: str = "agir"
    ) -> np.ndarray:
        """NumPy dizisi formatındaki bir görsel yığınını (batch) dönüştürür."""
        donusturulmus = [
            self.donustur_tekil(img, mod=mod) for img in gorseller_rgb
        ]
        return np.array(donusturulmus, dtype=gorseller_rgb.dtype)
