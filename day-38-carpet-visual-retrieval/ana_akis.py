"""
Day 38: Halı Doku ve Desenleri İçin Çoklu Özellikli Görsel Arama Ana Yürütme Betiği.
"""

import os
import numpy as np
from PIL import Image
from src.hali_katalog_verisi import sentetik_katalog_uret, sentetik_hali_deseni_olustur
from src.fuzyon_arama_motoru import CokluOzellikFuzyonAramaMotoru
from src.gorsellestirici import HaliGorselAramaGorsellestirici


def main():
    print("=" * 80)
    print(">>> ASAMA 1: Halı Katalog Veri Setinin ve Özellik İndeksinin Oluşturulması")
    print("=" * 80)

    katalog = sentetik_katalog_uret()
    print(f"[+] Katalog Verisi Üretildi: {len(katalog)} Adet Farklı Halı Kategorisi")
    for item in katalog:
        print(f"    - [{item['id']}] {item['baslik']:<38} | Kat: {item['kategori']:<20} | İplik: {item['iplik_tipi']}")

    arama_motoru = CokluOzellikFuzyonAramaMotoru(renk_agirligi=0.55, doku_agirligi=0.45)
    print("\n[+] Katalog Görselleri Renk (HSV+Moment) ve Doku (GLCM+LBP) Uzayına İndeksleniyor...")
    arama_motoru.katalog_indeksle(katalog)
    print(f"[+] İndeksleme Tamamlandı: {len(arama_motoru.indeks)} Vektör Bellekte.")

    print("\n" + "=" * 80)
    print(">>> ASAMA 2: Örnek Bir Sorgu Halısı (Query Carpet) Üretimi ve Görsel Arama")
    print("=" * 80)

    # Gerçekçi kamera çekimini simüle eden hafif gürültülü bir Klasik Hereke varyantı sorgusu
    sorgu_temel = sentetik_hali_deseni_olustur("CARPET-CLASSIC-01", genislik=300, yukseklik=300)
    np_sorgu = np.array(sorgu_temel, dtype=np.int16)
    noise = np.random.normal(0, 8, np_sorgu.shape).astype(np.int16)
    sorgu_gorseli = Image.fromarray(np.clip(np_sorgu + noise, 0, 255).astype(np.uint8))

    print("[+] Sorgu Görseli Hazırlandı: 'Klasik Bordürlü Bordo Desenli Halı (Kamera Çekimi Simülasyonu)'")

    # Çoklu Özellik Hibrit Görsel Arama (Top-4)
    arama_sonucu = arama_motoru.gorsel_ara(sorgu_gorseli, top_k=4)

    print("\n" + "=" * 80)
    print(">>> ASAMA 3: Çoklu Özellik Füzyon Arama Sonuçları (Rank & Skor Tablosu)")
    print("=" * 80)
    print(f"{'Sıra':<4} | {'Halı Kodu':<18} | {'Halı Başlığı':<32} | {'Hibrit %':<9} | {'Renk %':<8} | {'Doku %':<8} | {'Kategori'}")
    print("-" * 110)

    for i, res in enumerate(arama_sonucu["sonuclar"]):
        print(f"#{i+1:<3} | {res['id']:<18} | {res['baslik']:<32} | %{res['hibrit_skor']:<7.2f} | %{res['renk_skor']:<6.2f} | %{res['doku_skor']:<6.2f} | {res['kategori']}")

    print("\n" + "=" * 80)
    print(">>> ASAMA 4: 6 Panelli Görsel Arama Teşhis Panosunun Kaydedilmesi")
    print("=" * 80)

    cikis_resmi = HaliGorselAramaGorsellestirici.arama_paneli_ciz(
        sorgu_gorseli=sorgu_gorseli,
        arama_sonucu=arama_sonucu,
        hedef_path="day-38-carpet-visual-retrieval/ciktilar/hali_gorsel_arama_paneli.png"
    )
    print(f"[+] Teşhis Panosu Başarıyla Üretildi: {os.path.abspath(cikis_resmi)}")
    print("=" * 80)
    print("DAY 38: HALI DOKU VE DESENLERİ İÇİN GÖRSEL ARAMA BAŞARIYLA TAMAMLANDI!")
    print("=" * 80)


if __name__ == "__main__":
    main()
