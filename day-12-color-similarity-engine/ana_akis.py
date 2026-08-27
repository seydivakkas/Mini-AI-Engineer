"""Günün Ana Çalıştırma Akışı: Algısal Renk Benzerliği ve Katalog Arama.

Bu betik; farklı renk temalarına sahip 5 ürünlük sentetik bir halı kataloğu oluşturur,
kullanıcının sıcak sonbahar renkleri (Kiremit & Hardal) içeren sorgu görselini CIELAB
CIEDE2000 metriğiyle tarar, benzerlik skorlarına göre sıralar ve görsel karşılaştırma
raporunu diske kaydeder.
"""

import sys
from pathlib import Path
from typing import List

# Proje kök dizinini ekle
proje_kok = Path(__file__).resolve().parent
if str(proje_kok) not in sys.path:
    sys.path.insert(0, str(proje_kok))

import cv2
import numpy as np
from src.palet_eslestirici import PaletRengi
from src.katalog_arama import KatalogUrunu, RenkTabanliAramaMotoru
from src.gorsellestirici import AramaGorsellestirici


def baslik(metin: str) -> None:
    """Rapor bölüm başlığı basar."""
    cizgi = "=" * 74
    print(f"\n{cizgi}\n>>> {metin}\n{cizgi}")


def desenli_gorsel_uret(renkler: List[tuple], boyut: int = 180) -> np.ndarray:
    """Belirtilen BGR renkleriyle dairesel ve şeritli sentetik motif üretir."""
    resim = np.full((boyut, boyut, 3), renkler[0], dtype=np.uint8)
    if len(renkler) > 1:
        cv2.circle(resim, (boyut // 2, boyut // 2), boyut // 3, renkler[1], -1)
    if len(renkler) > 2:
        cv2.rectangle(resim, (0, 0), (boyut // 4, boyut), renkler[2], -1)
        cv2.rectangle(resim, (boyut * 3 // 4, 0), (boyut, boyut), renkler[2], -1)
    return resim


def main() -> None:
    baslik("AŞAMA 1: Sentetik Katalog Ürünlerinin ve Renk Paletlerinin Tanımlanması")

    # Ürün 1: Ege Mavisi Halı (Gece Mavisi, Gökyüzü Mavisi, Krem)
    urun_1 = KatalogUrunu(
        urun_id="HL-101",
        ad="Ege Mavisi Klasik",
        kategori="Halı",
        gorsel_bgr=desenli_gorsel_uret([(130, 40, 20), (220, 160, 50), (240, 245, 245)]),
        palet=[
            PaletRengi(rgb=(20, 40, 130), agirlik=0.50, hex_kodu="#142882"),
            PaletRengi(rgb=(50, 160, 220), agirlik=0.35, hex_kodu="#32A0DC"),
            PaletRengi(rgb=(245, 245, 240), agirlik=0.15, hex_kodu="#F5F5F0"),
        ]
    )

    # Ürün 2: Sonbahar Toprak Halı (Kiremit, Hardal, Çikolata)
    urun_2 = KatalogUrunu(
        urun_id="HL-102",
        ad="Sonbahar Toprak",
        kategori="Halı",
        gorsel_bgr=desenli_gorsel_uret([(30, 40, 190), (30, 180, 220), (20, 40, 90)]),
        palet=[
            PaletRengi(rgb=(190, 40, 30), agirlik=0.55, hex_kodu="#BE281E"),
            PaletRengi(rgb=(220, 180, 30), agirlik=0.30, hex_kodu="#DCB41E"),
            PaletRengi(rgb=(90, 40, 20), agirlik=0.15, hex_kodu="#5A2814"),
        ]
    )

    # Ürün 3: Akdeniz Güneşi (Sıcak Kiremit, Turuncu, Amber)
    urun_3 = KatalogUrunu(
        urun_id="HL-103",
        ad="Akdeniz Güneşi",
        kategori="Halı",
        gorsel_bgr=desenli_gorsel_uret([(20, 50, 210), (30, 120, 240), (20, 200, 245)]),
        palet=[
            PaletRengi(rgb=(210, 50, 20), agirlik=0.50, hex_kodu="#D23214"),
            PaletRengi(rgb=(240, 120, 30), agirlik=0.35, hex_kodu="#F0781E"),
            PaletRengi(rgb=(245, 200, 20), agirlik=0.15, hex_kodu="#F5C814"),
        ]
    )

    # Ürün 4: İskandinav Nötr (Antrasit, Açık Gri, Kırık Beyaz)
    urun_4 = KatalogUrunu(
        urun_id="HL-104",
        ad="İskandinav Minimal",
        kategori="Halı",
        gorsel_bgr=desenli_gorsel_uret([(60, 60, 60), (160, 160, 160), (230, 230, 230)]),
        palet=[
            PaletRengi(rgb=(60, 60, 60), agirlik=0.50, hex_kodu="#3C3C3C"),
            PaletRengi(rgb=(160, 160, 160), agirlik=0.35, hex_kodu="#A0A0A0"),
            PaletRengi(rgb=(230, 230, 230), agirlik=0.15, hex_kodu="#E6E6E6"),
        ]
    )

    # Ürün 5: Tropik Zümrüt (Zümrüt Yeşili, Nane Yeşili, Altın)
    urun_5 = KatalogUrunu(
        urun_id="HL-105",
        ad="Tropik Zümrüt",
        kategori="Halı",
        gorsel_bgr=desenli_gorsel_uret([(40, 110, 20), (120, 190, 80), (30, 190, 220)]),
        palet=[
            PaletRengi(rgb=(20, 110, 40), agirlik=0.50, hex_kodu="#146E28"),
            PaletRengi(rgb=(80, 190, 120), agirlik=0.35, hex_kodu="#50BE78"),
            PaletRengi(rgb=(220, 190, 30), agirlik=0.15, hex_kodu="#DCBE1E"),
        ]
    )

    katalog = [urun_1, urun_2, urun_3, urun_4, urun_5]
    print(f"[+] Kataloğa {len(katalog)} adet ürün kaydedildi.")
    for u in katalog:
        print(f"    - [{u.urun_id}] {u.ad} (Palet Rengi: {len(u.palet)} adet)")

    baslik("AŞAMA 2: Kullanıcı Sorgusunun Oluşturulması (Kiremit & Hardal Paleti)")
    sorgu_gorsel = desenli_gorsel_uret([(35, 45, 195), (35, 175, 215)], boyut=180)
    sorgu_paleti = [
        PaletRengi(rgb=(195, 45, 35), agirlik=0.65, hex_kodu="#C32D23"),  # Kiremit Kırmızısı
        PaletRengi(rgb=(215, 175, 35), agirlik=0.35, hex_kodu="#D7AF23"),  # Sıcak Hardal
    ]
    print("[+] Sorgu Paleti:")
    for r in sorgu_paleti:
        print(f"    * {r.hex_kodu} (RGB: {r.rgb}) -> Ağırlık: %{r.agirlik*100:.0f}")

    baslik("AŞAMA 3: CIEDE2000 Algısal Benzerlik Taraması ve Sıralama")
    arama_motoru = RenkTabanliAramaMotoru(metrik="ciede2000", hassasiyet_sigma=25.0)
    arama_motoru.urunleri_toplu_ekle(katalog)

    sonuclar = arama_motoru.arama_yap(sorgu_paleti, en_iyi_k=3)

    print(f"{'Sıra':<5} | {'Ürün ID':<8} | {'Ürün Adı':<20} | {'Benzerlik':<12} | {'Delta-E':<10} | {'En Yakın Eşleşme'}")
    print("-" * 74)
    for idx, s in enumerate(sonuclar, 1):
        en_yakin = s.eslesmeler[0]
        eslesme_ozet = f"{en_yakin['kaynak_hex']} -> {en_yakin['hedef_hex']} (Delta-E={en_yakin['delta_e']})"
        print(f"#{idx:<4} | {s.urun.urun_id:<8} | {s.urun.ad:<20} | %{s.benzerlik_skoru:<10.1f} | {s.delta_e_mesafesi:<10.2f} | {eslesme_ozet}")
    print("-" * 74)

    baslik("AŞAMA 4: Görsel Arama Paneli ve Benzerlik Rozetlerinin Kaydedilmesi")
    cikti_yolu = proje_kok / "ciktilar" / "renk_arama_sonuclari.png"
    kaydedilen = AramaGorsellestirici.arama_raporu_ciz(
        sorgu_gorsel_bgr=sorgu_gorsel,
        sorgu_paleti=sorgu_paleti,
        sonuclar=sonuclar,
        dosya_yolu=cikti_yolu
    )
    print(f"[V] Arama sonuçları görsel paneli kaydedildi: {kaydedilen.name}")
    print(f"[V] Kayıt Konumu: {kaydedilen}")
    print("\n[V] Day 12: Algısal Renk Benzerliği ve Arama Altyapısı başarıyla tamamlandı.")


if __name__ == "__main__":
    main()
