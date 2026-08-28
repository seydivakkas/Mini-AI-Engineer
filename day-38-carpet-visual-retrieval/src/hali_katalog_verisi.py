"""
Halı Katalog Veri Seti ve Sentetik Halı Deseni Üretici.
Klasik Hereke, Modern İskandinav, Shaggy, Vintage Eskitme, İpek Çiçekli ve Jüt Halı Tipleri.
"""

from typing import List, Dict, Any
import numpy as np
from PIL import Image, ImageDraw


def sentetik_hali_deseni_olustur(hali_kodu: str, genislik: int = 300, yukseklik: int = 300) -> Image.Image:
    """Farklı halı kategorilerine özgü desen, renk paleti ve doku üretir."""
    img = Image.new("RGB", (genislik, yukseklik))
    draw = ImageDraw.Draw(img)

    if hali_kodu == "CARPET-CLASSIC-01":
        # Klasik Hereke Bordürlü Madalyon (Bordo, Krem, Lacivert)
        draw.rectangle([0, 0, genislik, yukseklik], fill=(228, 217, 198))
        draw.rectangle([15, 15, genislik - 15, yukseklik - 15], outline=(138, 28, 48), width=18)
        draw.rectangle([40, 40, genislik - 40, yukseklik - 40], outline=(24, 43, 73), width=8)
        cx, cy = genislik // 2, yukseklik // 2
        draw.ellipse([cx - 50, cy - 50, cx + 50, cy + 50], fill=(138, 28, 48), outline=(204, 154, 45), width=5)
        # Geleneksel mikro doku
        arr = np.array(img, dtype=np.int16)
        noise = np.random.normal(0, 4, arr.shape).astype(np.int16)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    elif hali_kodu == "CARPET-MODERN-02":
        # İskandinav Geometrik Vizon (Pastel Vizon, Hardal, Gri)
        draw.rectangle([0, 0, genislik, yukseklik], fill=(210, 205, 195))
        for x in range(0, genislik, 60):
            for y in range(0, yukseklik, 60):
                if (x + y) % 120 == 0:
                    draw.polygon([(x, y), (x + 60, y), (x + 30, y + 60)], fill=(204, 154, 45))
                else:
                    draw.polygon([(x, y + 60), (x + 60, y + 60), (x + 30, y)], fill=(80, 85, 95))
        arr = np.array(img, dtype=np.int16)
        noise = np.random.normal(0, 2, arr.shape).astype(np.int16)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    elif hali_kodu == "CARPET-SHAGGY-03":
        # Kabarık Yün Shaggy Antrasit (Kaba Doku, Yüksek Gürültü/Pürüzlülük)
        draw.rectangle([0, 0, genislik, yukseklik], fill=(50, 52, 58))
        arr = np.array(img, dtype=np.int16)
        # Yoğun uzun iplik dalgalanması
        for _ in range(300):
            x = np.random.randint(0, genislik)
            y = np.random.randint(0, yukseklik)
            draw.line([(x, y), (x + np.random.randint(-15, 15), y + np.random.randint(-15, 15))], fill=(85, 88, 95), width=2)
        arr = np.array(img, dtype=np.int16)
        noise = np.random.normal(0, 18, arr.shape).astype(np.int16)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    elif hali_kodu == "CARPET-VINTAGE-04":
        # Anadolu Eskitme Terracotta (Kiremit, Soluk Bej, Düzensiz Eskitme)
        draw.rectangle([0, 0, genislik, yukseklik], fill=(184, 85, 55))
        for i in range(10, genislik, 40):
            draw.line([(i, 0), (i, yukseklik)], fill=(220, 200, 180), width=3)
        # Eskitme lekeleri
        for _ in range(15):
            rx, ry = np.random.randint(20, genislik-20), np.random.randint(20, yukseklik-20)
            draw.ellipse([rx - 25, ry - 25, rx + 25, ry + 25], fill=(210, 180, 150))
        arr = np.array(img, dtype=np.int16)
        noise = np.random.normal(0, 12, arr.shape).astype(np.int16)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    elif hali_kodu == "CARPET-SILK-05":
        # Osmanlı İpek Çiçekli Zümrüt (Zümrüt Yeşili, Altın Sarısı, Krem)
        draw.rectangle([0, 0, genislik, yukseklik], fill=(32, 98, 65))
        draw.rectangle([20, 20, genislik - 20, yukseklik - 20], outline=(204, 154, 45), width=6)
        cx, cy = genislik // 2, yukseklik // 2
        for angle in range(0, 360, 45):
            rad = np.radians(angle)
            px = int(cx + 60 * np.cos(rad))
            py = int(cy + 60 * np.sin(rad))
            draw.ellipse([px - 15, py - 15, px + 15, py + 15], fill=(228, 217, 198), outline=(204, 154, 45), width=2)
        arr = np.array(img, dtype=np.int16)
        noise = np.random.normal(0, 3, arr.shape).astype(np.int16)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    else:
        # Jüt Hasır Düz Dokuma (CARPET-MINIMAL-06)
        draw.rectangle([0, 0, genislik, yukseklik], fill=(195, 175, 145))
        for y in range(0, yukseklik, 10):
            renk_val = (175, 155, 125) if (y // 10) % 2 == 0 else (210, 190, 160)
            draw.line([(0, y), (genislik, y)], fill=renk_val, width=5)
        arr = np.array(img, dtype=np.int16)
        noise = np.random.normal(0, 6, arr.shape).astype(np.int16)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)


def sentetik_katalog_uret() -> List[Dict[str, Any]]:
    """Arama motoru için zengin katalog koleksiyonunu döndürür."""
    katalog = [
        {
            "id": "CARPET-CLASSIC-01",
            "baslik": "Hereke Klasik Madalyonlu Bordo Halı",
            "kategori": "Klasik / Geleneksel",
            "iplik_tipi": "%100 Akrilik Dokuma",
            "hav_yuksekligi": "12 mm",
            "ana_renkler": ["Bordo", "Krem", "Gece Mavisi", "Hardal"]
        },
        {
            "id": "CARPET-MODERN-02",
            "baslik": "İskandinav Geometrik Triko Halı",
            "kategori": "Modern / Geometrik",
            "iplik_tipi": "Polipropilen & Şönil",
            "hav_yuksekligi": "8 mm",
            "ana_renkler": ["Vizon", "Hardal", "Antrasit"]
        },
        {
            "id": "CARPET-SHAGGY-03",
            "baslik": "Kabarık Yün Dokulu Shaggy Halı",
            "kategori": "Shaggy / Kaba Doku",
            "iplik_tipi": "Mikrofiber Yün Dokuma",
            "hav_yuksekligi": "30 mm",
            "ana_renkler": ["Antrasit Kömür Grisi"]
        },
        {
            "id": "CARPET-VINTAGE-04",
            "baslik": "Anadolu Eskitme Terracotta Halı",
            "kategori": "Vintage / Eskitme",
            "iplik_tipi": "Pamuk Tabanlı Dijital Baskı / Jakar",
            "hav_yuksekligi": "6 mm",
            "ana_renkler": ["Kiremit", "Bej", "Toprak"]
        },
        {
            "id": "CARPET-SILK-05",
            "baslik": "Osmanlı İpek Çiçekli Zümrüt Halı",
            "kategori": "Klasik / Saray",
            "iplik_tipi": "Bambu İpeği & Viskon",
            "hav_yuksekligi": "10 mm",
            "ana_renkler": ["Zümrüt Yeşili", "Altın Sarısı", "Krem"]
        },
        {
            "id": "CARPET-MINIMAL-06",
            "baslik": "Doğal Jüt Hasır Düz Dokuma Halı",
            "kategori": "Bohem / Hasır",
            "iplik_tipi": "%100 Doğal Jüt Elyafı",
            "hav_yuksekligi": "5 mm (Düz Dokuma)",
            "ana_renkler": ["Hasır Beji", "Doğal Kahve"]
        }
    ]

    for item in katalog:
        item["gorsel"] = sentetik_hali_deseni_olustur(item["id"])

    return katalog
