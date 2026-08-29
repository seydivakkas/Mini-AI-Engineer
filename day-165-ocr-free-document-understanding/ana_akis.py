"""
Day 165: OCR-Free Doküman ve Tablo Anlama (Donut / Nougat Mimarisi) Ana Akışı.
Doğrudan Doküman Piksellerinden LaTeX Formülü, Markdown Tablosu ve JSON Fatura Ayrıştırma.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.donut_nougat_ayristirici import DonutNougatAyristirici
from src.gorsellestirici import DokumanGorsellestirici


def main():
    print("=" * 110)
    print(">>> Day 165 (FAZ 9): OCR-FREE DOCUMENT UNDERSTANDING: DONUT / NOUGAT LATEX & MARKDOWN PARSER")
    print("=" * 110)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # -------------------------------------------------------------
    # 1. DOKÜMAN VE TABLO AYRIŞTIRMA DEĞERLENDİRMESİ
    # -------------------------------------------------------------
    print("\n[1/2] OCR-Free Doküman Senaryoları (LaTeX, Markdown, JSON) Ayrıştırılıyor...")
    rapor = DonutNougatAyristirici.dokumanlari_ayristir_ve_degerlendir()

    print("\n" + "-" * 105)
    print(f"{'Doküman Tipi':<30} | {'Başlık':<32} | {'Levenshtein':<12} | {'Edit Similarity'}")
    print("-" * 105)
    for s in rapor["senaryo_sonuclari"]:
        print(
            f"{s['dokuman_tipi']:<30} | {s['baslik']:<32} | "
            f"{s['levenshtein_mesafesi']} karakter  | %{s['edit_similarity']*100:<10.1f}"
        )
    print("-" * 105)

    ozet = rapor["genel_ozet"]
    print("\nGENEL PERFORMANS METRİKLERİ:")
    print(f"  • Toplam Ayrıştırılan Doküman : {ozet['toplam_dokuman']}")
    print(f"  • Ortalama Edit Similarity    : {ozet['ortalama_edit_similarity']}")
    print(f"  • Genel Doğruluk Yüzdesi      : %{ozet['ortalama_dogruluk_yuzdesi']} (Kusursuz OCR-Free Eşleşme)")

    # -------------------------------------------------------------
    # 2. 6 PANELLİ TEŞHİS PANOSU ÜRETİMİ
    # -------------------------------------------------------------
    print("\n[2/2] 6 Panelli OCR-Free Doküman Teşhis Panosu Üretiliyor...")
    gorsellestirici = DokumanGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "ocr_free_document_understanding_paneli.png")
    gorsellestirici.pano_olustur(rapor, kayit_yolu=cikis_resmi)

    print("\n" + "=" * 110)
    print("✓ Day 165: OCR-FREE DOCUMENT UNDERSTANDING BAŞARIYLA TAMAMLANDI!")
    print("=" * 110)


if __name__ == "__main__":
    main()
