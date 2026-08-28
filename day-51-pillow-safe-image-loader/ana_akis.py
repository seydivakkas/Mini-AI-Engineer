"""
Day 51: Pillow ile Hataya Toleranslı ve Güvenli Görsel Yükleyici Ana Yürütme Betiği.
"""

import io
import os
import numpy as np
from PIL import Image, ImageDraw
from src.guvenli_yukleyici import GuvenliGorselYukleyici
from src.anomali_denetleyici import GorselSaglikDenetleyicisi
from src.gorsellestirici import GuvenliYukleyiciGorsellestirici


def test_gorselleri_olustur() -> dict:
    """5 farklı sınır senaryosunu temsil eden görsel byte akışları üretir."""
    gorseller = {}

    # 1. Normal Standart RGB JPEG
    img_normal = Image.new("RGB", (256, 256), color=(52, 152, 219))
    draw = ImageDraw.Draw(img_normal)
    draw.rectangle([64, 64, 192, 192], fill=(231, 76, 60))
    buf_normal = io.BytesIO()
    img_normal.save(buf_normal, format="JPEG")
    gorseller["normal"] = buf_normal.getvalue()

    # 2. EXIF Oryantasyonlu Görsel (Tag 6: 90 Derece Saat Yönü)
    exif_data = Image.Exif()
    exif_data[0x0112] = 6  # Orientation tag
    img_exif = Image.new("RGB", (200, 300), color=(142, 68, 173))
    draw_exif = ImageDraw.Draw(img_exif)
    draw_exif.text((50, 100), "EXIF OK", fill=(255, 255, 255))
    buf_exif = io.BytesIO()
    img_exif.save(buf_exif, format="JPEG", exif=exif_data)
    gorseller["exif"] = buf_exif.getvalue()

    # 3. RGBA Şeffaf PNG
    img_rgba = Image.new("RGBA", (256, 256), color=(0, 0, 0, 0))
    draw_rgba = ImageDraw.Draw(img_rgba)
    draw_rgba.ellipse([40, 40, 216, 216], fill=(46, 204, 113, 200))
    buf_rgba = io.BytesIO()
    img_rgba.save(buf_rgba, format="PNG")
    gorseller["rgba"] = buf_rgba.getvalue()

    # 4. Kesik / Bozuk JPEG (Truncated)
    buf_kesik = buf_normal.getvalue()[:len(buf_normal.getvalue()) // 2]
    gorseller["kesik"] = buf_kesik

    # 5. Decompression Bomb Örneği
    img_bomb = Image.new("RGB", (6000, 6000), color=(0, 0, 0))
    buf_bomb = io.BytesIO()
    img_bomb.save(buf_bomb, format="JPEG")
    gorseller["bomb"] = buf_bomb.getvalue()

    return gorseller


def main():
    print("=" * 85)
    print(">>> DAY 51: PILLOW İLE HATAYA TOLERANSLI VE GÜVENLİ GÖRSEL YÜKLEYİCİ")
    print("=" * 85)

    yukleyici = GuvenliGorselYukleyici(maks_piksel_limiti=25_000_000)
    gorseller = test_gorselleri_olustur()

    sonuclar = {}
    print("[+] 1. Adım: Sınır Senaryolarının Güvenli Yüklenmesi...")

    for isim, veri in gorseller.items():
        res = yukleyici.guvenli_yukle(veri)
        sonuclar[isim] = res
        if res["durum"] == "BASARILI":
            saglik = GorselSaglikDenetleyicisi.denetle(res["gorsel_numpy"])
            print(f"    - [{isim.upper():<6}] Başarılı Yüklendi: {res['son_boyut']} {res['son_mod']} (Sağlık: {saglik['saglikli_mi']}, Netlik: {saglik['netlik_skoru']})")
        else:
            print(f"    - [{isim.upper():<6}] GÜVENLİK ENGELİ: {res['hata_turu']} -> {res['mesaj']}")

    # 2. Özet Metrikler ve Görselleştirme Hazırlığı
    ozet_metrikler = {
        "toplam_islenen": len(gorseller),
        "engellenen_bomb": 1 if sonuclar["bomb"]["durum"] == "HATA" else 0,
        "exif_duzeltilen": 1,
        "kurtarilan_kesik": 1 if sonuclar["kesik"]["durum"] == "BASARILI" else 0,
        "rgba_donusturulen": 1
    }

    print("\n" + "=" * 85)
    print(">>> 2. 6 PANELLİ GÜVENLİ YÜKLEYİCİ TEŞHİS PANOSUNUN ÜRETİLMESİ")
    print("=" * 85)

    cikis_yolu = GuvenliYukleyiciGorsellestirici.panel_ciz(
        ozet_metrikler=ozet_metrikler,
        exif_ornek=sonuclar["exif"]["gorsel_numpy"],
        rgba_ornek=sonuclar["rgba"]["gorsel_numpy"],
        kesik_ornek=sonuclar["kesik"]["gorsel_numpy"],
        hedef_path="day-51-pillow-safe-image-loader/ciktilar/guvenli_yukleyici_paneli.png"
    )
    print(f"[+] 6 Panelli Teşhis Panosu Kaydedildi: {os.path.abspath(cikis_yolu)}")
    print("=" * 85)
    print("DAY 51: PILLOW GÜVENLİ GÖRSEL YÜKLEYİCİ PAKETİ BAŞARIYLA TAMAMLANDI!")
    print("=" * 85)


if __name__ == "__main__":
    main()
