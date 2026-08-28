"""
RGB, sRGB, HEX ve Standart CIELAB D65 Renk Uzayı Dönüştürücü (Color Space Converter).
"""

from typing import Tuple, Union
import numpy as np
from skimage import color


class RenkUzayiDonusturucu:
    """sRGB ve algısal olarak üniform standart CIELAB (D65 aydınlatıcı) uzayı arasında çift yönlü dönüşüm yapar."""

    @classmethod
    def rgb_to_cielab(cls, rgb_array: np.ndarray) -> np.ndarray:
        """[0, 255] aralığındaki RGB görüntüsünü/piksellerini standart CIELAB (L*:[0,100], a*:[-128,127], b*:[-128,127]) uzayına dönüştürür."""
        rgb_norm = np.clip(rgb_array.astype(np.float64) / 255.0, 0.0, 1.0)
        lab = color.rgb2lab(rgb_norm)
        return lab

    @classmethod
    def cielab_to_rgb(cls, lab_array: np.ndarray) -> np.ndarray:
        """Standart CIELAB koordinatlarını [0, 255] uint8 sRGB formatına geri dönüştürür."""
        lab_guvenli = lab_array.astype(np.float64)
        rgb_norm = color.lab2rgb(lab_guvenli)
        rgb_uint8 = np.clip(np.round(rgb_norm * 255.0), 0, 255).astype(np.uint8)
        return rgb_uint8

    @classmethod
    def rgb_to_hex(cls, rgb: Union[Tuple[int, int, int], np.ndarray]) -> str:
        """(R, G, B) değerlerini standart '#RRGGBB' HEX dizesine dönüştürür."""
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
        return f"#{r:02X}{g:02X}{b:02X}"

    @classmethod
    def hex_to_rgb(cls, hex_str: str) -> np.ndarray:
        """'#RRGGBB' veya 'RRGGBB' HEX dizesini [R, G, B] uint8 dizisine çevirir."""
        hex_temiz = hex_str.lstrip("#")
        if len(hex_temiz) != 6:
            raise ValueError(f"Geçersiz HEX renk formatı: {hex_str}")
        return np.array([int(hex_temiz[i:i+2], 16) for i in (0, 2, 4)], dtype=np.uint8)
