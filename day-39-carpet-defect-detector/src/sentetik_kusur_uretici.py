"""
Sentetik Kusurlu Halı Görseli ve Hata Enjektörü (Synthetic Carpet Defect Injector).
"""

from typing import Tuple, List, Dict, Any
import numpy as np
from PIL import Image, ImageDraw


class SentetikKusurluHaliUretici:
    """Temiz referans halı deseni üretip üzerine kontrollü dokuma hataları enjekte eder."""

    @classmethod
    def temiz_referans_uret(cls, genislik: int = 400, yukseklik: int = 300) -> Image.Image:
        """Kusursuz jakarlı halı referans deseni üretir."""
        img = Image.new("RGB", (genislik, yukseklik), color=(225, 215, 195))  # Krem bej
        draw = ImageDraw.Draw(img)

        # Çerçeve ve bordürler
        draw.rectangle([15, 15, genislik - 15, yukseklik - 15], outline=(140, 30, 50), width=16)
        draw.rectangle([45, 45, genislik - 45, yukseklik - 45], outline=(25, 45, 75), width=8)

        # Merkez madalyon
        cx, cy = genislik // 2, yukseklik // 2
        draw.ellipse([cx - 70, cy - 50, cx + 70, cy + 50], fill=(35, 100, 70), outline=(200, 150, 45), width=4)

        # Doğal dokuma mikro pürüzlülüğü
        np_arr = np.array(img, dtype=np.int16)
        noise = np.random.normal(0, 3, np_arr.shape).astype(np.int16)
        return Image.fromarray(np.clip(np_arr + noise, 0, 255).astype(np.uint8))

    @classmethod
    def kusurlu_test_uret(
        cls,
        referans_gorseli: Image.Image
    ) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        """Referans halı üzerine 3 farklı dokuma hatası (İplik Kopması, Yağ Lekesi, Düğüm) enjekte eder."""
        test_img = referans_gorseli.copy()
        draw = ImageDraw.Draw(test_img)

        W, H = test_img.size
        enjekte_edilenler = []

        # 1. Hata: İplik Kopması (Elongated Warp/Weft Defect)
        # Yatay ince uzun çizgi
        x1, y1 = 70, 80
        x2, y2 = 230, 84
        draw.rectangle([x1, y1, x2, y2], fill=(40, 40, 40))
        enjekte_edilenler.append({"tur": "IPLIK_KOPMASI", "konum": [x1, y1, x2-x1, y2-y1]})

        # 2. Hata: Yağ / Boya Lekesi (Oil Stain Blob)
        # Koyu dairesel leke
        lx, ly, lr = 280, 180, 22
        draw.ellipse([lx - lr, ly - lr, lx + lr, ly + lr], fill=(20, 25, 30))
        enjekte_edilenler.append({"tur": "YAG_BOYA_LEKESI", "konum": [lx-lr, ly-lr, lr*2, lr*2]})

        # 3. Hata: İplik Düğümü / Topaklanma (Small Slub / Knot)
        kx, ky, kr = 130, 220, 10
        draw.ellipse([kx - kr, ky - kr, kx + kr, ky + kr], fill=(240, 240, 240), outline=(20, 20, 20), width=2)
        enjekte_edilenler.append({"tur": "DUGUM_TOPAKLANMA", "konum": [kx-kr, ky-kr, kr*2, kr*2]})

        # Doku gürültüsü
        arr = np.array(test_img, dtype=np.int16)
        noise = np.random.normal(0, 3, arr.shape).astype(np.int16)
        test_kusurlu = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

        return test_kusurlu, enjekte_edilenler
