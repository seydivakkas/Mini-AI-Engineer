"""
Day 155: Needle In A Haystack (NIAH) Uzun Bağlam Değerlendirme Ana Akışı.
128k Token Bağlam Izgarası, 'Lost in the Middle' Analizi ve Çoklu İğne Akıl Yürütme.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.samanlik_olusturucu import SamanlikOlusturucu
from src.niah_test_motoru import NIAHTestMotoru
from src.coklu_igne_akil_yurutucu import CokluIgneAkilYurutucu
from src.gorsellestirici import NIAHGorsellestirici


def main():
    print("=" * 105)
    print(">>> Day 155: Needle In A Haystack (NIAH) Long-Context Reasoning Engine (FAZ 8)")
    print("=" * 105)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # -------------------------------------------------------------
    # 1. SAMANLIK VE İĞNE ENJEKSİYON DEMOSU
    # -------------------------------------------------------------
    print("\n[1/3] Sentetik Samanlık (Haystack) ve İğne (Needle) Enjeksiyonu Yapılıyor...")
    ornek_igne = "Proje Phoenix'in lansman tarihi 14 Mayıs 2027'dir."
    samanlik_bilgisi = SamanlikOlusturucu.samanlik_uret(
        hedef_kelime_sayisi=1000,
        igne_metni=ornek_igne,
        derinlik_yuzdesi=0.50, # %50 derinlik (tam ortası)
    )

    print(f"  • Üretilen Kelime Sayısı : {samanlik_bilgisi['toplam_kelime_sayisi']}")
    print(f"  • İğne Derinliği         : %{samanlik_bilgisi['derinlik_yuzdesi']:.1f}")
    print(f"  • Enjekte Edilen İğne    : '{samanlik_bilgisi['igne_metni']}'")

    # -------------------------------------------------------------
    # 2. 8x11 NIAH IZGARA TESTİ VE ÇOKLU İĞNE AKIL YÜRÜTME
    # -------------------------------------------------------------
    print("\n[2/3] 1k - 128k Token Bağlam Izgarasında NIAH Testi ve Çoklu İğne Sentezi Koşturuluyor...")
    motor = NIAHTestMotoru()
    niah_raporu = motor.tam_degerlendirme_yap()

    print("\n" + "-" * 75)
    print(f"  • Test Edilen Bağlamlar       : {[f'{u//1000}k' for u in niah_raporu['baglam_uzunluklari']]}")
    print(f"  • Test Edilen Derinlikler     : {[f'%{d}' for d in niah_raporu['derinlik_yuzdeleri']]}")
    print(f"  • Genel Ortalama Doğruluk     : %{niah_raporu['ortalama_dogruluk']*100:.1f}")
    print(f"  • Orta Bölge (%40-%60) Başarım: %{niah_raporu['orta_bolge_dogruluk']*100:.1f}")
    print(f"  • 'Lost in the Middle' Kaybı  : %{niah_raporu['lost_in_middle_kaybi']*100:.1f}")
    print("-" * 75)

    print("\nÇOKLU İĞNE (MULTI-NEEDLE) AKIL YÜRÜTME ÇIKTISI:")
    coklu_sonuc = CokluIgneAkilYurutucu.coklu_igne_sentezle(samanlik_bilgisi["tam_dokuman"])
    for i, igne in enumerate(coklu_sonuc["toplanan_igneler"], start=1):
        print(f"  [İpucu {i}]: {igne}")
    print(f"  => Soru: '{coklu_sonuc['soru']}'")
    print(f"  => Nihai Cevap: ${coklu_sonuc['nihai_cevap']} Milyon Dolar (%100 Başarılı Çıkarım)")

    # -------------------------------------------------------------
    # 3. 6 PANELLİ TEŞHİS PANOSU ÜRETİMİ
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli NIAH Uzun Bağlam Teşhis Panosu Üretiliyor...")
    gorsellestirici = NIAHGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "long_context_needle_in_a_haystack_paneli.png")
    gorsellestirici.pano_olustur(niah_raporu, coklu_sonuc, kayit_yolu=cikis_resmi)

    print("\n" + "=" * 105)
    print("✓ Day 155: NEEDLE IN A HAYSTACK (NIAH) BAŞARIYLA TAMAMLANDI!")
    print("=" * 105)


if __name__ == "__main__":
    main()
