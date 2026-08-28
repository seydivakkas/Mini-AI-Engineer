"""
Pillow ile Hataya Toleranslı, EXIF Düzeltmeli ve Güvenli Görsel Yükleyici (Safe Image Loader).
"""

from typing import Dict, Any, Tuple, Optional, Union
import io
import os
import numpy as np
from PIL import Image, ImageOps, ImageFile

# Kesik/bozuk ağ akışlarından gelen görsellerin yüklenmesine izin ver
ImageFile.LOAD_TRUNCATED_IMAGES = True


class GuvenliGorselYukleyici:
    """Görüntüleri Decompression Bomb, bozuk header, EXIF rotasyonu ve renk uzayı uyumsuzluklarına karşı korur."""

    def __init__(self, maks_piksel_limiti: int = 25_000_000, arka_plan_rengi: Tuple[int, int, int] = (255, 255, 255)):
        self.maks_piksel_limiti = maks_piksel_limiti
        self.arka_plan_rengi = arka_plan_rengi
        Image.MAX_IMAGE_PIXELS = maks_piksel_limiti

    def guvenli_yukle(self, kaynak: Union[str, bytes, io.BytesIO]) -> Dict[str, Any]:
        """Görseli tüm güvenlik ve normalizasyon kontrollerinden geçirerek RGB olarak yükler."""
        try:
            # 1. ByteIO veya Dosya Yolu Ayrımı
            if isinstance(kaynak, bytes):
                akıs = io.BytesIO(kaynak)
            elif isinstance(kaynak, str):
                if not os.path.exists(kaynak):
                    return {"durum": "HATA", "hata_turu": "DOSYA_BULUNAMADI", "mesaj": f"Dosya mevcut değil: {kaynak}"}
                akıs = open(kaynak, "rb")
            else:
                akıs = kaynak

            # 2. Ön Doğrulama ve Boyut Kontrolü (Decompression Bomb Guard)
            with Image.open(akıs) as img_ham:
                genislik, yukseklik = img_ham.size
                piksel_sayisi = genislik * yukseklik

                if piksel_sayisi > self.maks_piksel_limiti:
                    return {
                        "durum": "HATA",
                        "hata_turu": "DECOMPRESSION_BOMB_ENGELENDI",
                        "mesaj": f"Piksel sayısı ({piksel_sayisi}) güvenlik sınırını ({self.maks_piksel_limiti}) aştı!",
                        "boyut": (genislik, yukseklik)
                    }

                ham_mod = img_ham.mode
                ham_format = img_ham.format

                # 3. EXIF Oryantasyon Düzeltmesi (Mobil/DSLR rotasyonları)
                img_duzeltilmis = ImageOps.exif_transpose(img_ham)
                if img_duzeltilmis is None:
                    img_duzeltilmis = img_ham.copy()
                else:
                    img_duzeltilmis = img_duzeltilmis.copy()

            # 4. Renk Uzayı ve Alfa Kanalı Normalizasyonu
            if img_duzeltilmis.mode in ("RGBA", "LA"):
                # Şeffaf pikseller için beyaz mat arka plan kompoziti
                arka_plan = Image.new("RGB", img_duzeltilmis.size, self.arka_plan_rengi)
                if img_duzeltilmis.mode == "RGBA":
                    arka_plan.paste(img_duzeltilmis, mask=img_duzeltilmis.split()[3])
                else:
                    arka_plan.paste(img_duzeltilmis, mask=img_duzeltilmis.split()[1])
                img_rgb = arka_plan
            elif img_duzeltilmis.mode == "CMYK":
                img_rgb = img_duzeltilmis.convert("RGB")
            elif img_duzeltilmis.mode in ("L", "P", "1"):
                img_rgb = img_duzeltilmis.convert("RGB")
            else:
                img_rgb = img_duzeltilmis.convert("RGB")

            # 5. Metadata Temizliği ve NumPy Çıktısı
            dizi_rgb = np.array(img_rgb, dtype=np.uint8)

            return {
                "durum": "BASARILI",
                "gorsel_pil": img_rgb,
                "gorsel_numpy": dizi_rgb,
                "ham_boyut": (genislik, yukseklik),
                "son_boyut": (img_rgb.width, img_rgb.height),
                "ham_mod": ham_mod,
                "son_mod": "RGB",
                "format": ham_format or "UNKNOWN",
                "bellek_mb": float(round((dizi_rgb.nbytes) / (1024 * 1024), 2))
            }

        except Exception as e:
            return {
                "durum": "HATA",
                "hata_turu": type(e).__name__,
                "mesaj": str(e)
            }
