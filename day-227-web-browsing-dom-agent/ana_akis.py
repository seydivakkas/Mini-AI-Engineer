"""
Day 227: Web Tarayıcı ve DOM Ağacı Ajanı Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.dom_tarayici_motoru import (
    DOMElement,
    DOMTreePruner,
    WebBrowsingAgent,
)
from src.dom_profilleyici import DOMProfilleyici
from src.gorsellestirici import DOMGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 227 (FAZ 12): WEB TARAYICI AJANI - HTML DOM AĞACI BUDAMA VE OTONOM GEZİNME")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Ham HTML'in Budanması ve Erişilebilirlik Ağacı
    # -------------------------------------------------------------
    print("\n[1/4] Ham HTML Sayfası Temizleniyor ve Set-of-Marks [ID] Ağacı Üretiliyor...")

    ornek_html = """
    <html>
        <head>
            <script>var x = 100; // Gürültülü Takip Kodu</script>
            <style>body { background: #000; }</style>
        </head>
        <body>
            <header><h1>Teknoloji Mağazası</h1></header>
            <section class="search-box">
                <input type="text" placeholder="Aramak istediğiniz ürün..." />
                <button class="search-btn">Ürün Bul</button>
            </section>
            <main>
                <div class="product">
                    <h2>Yapay Zeka Hızlandırıcı GPU Kartı</h2>
                    <span class="price">49.999 TL</span>
                    <button class="cart-btn">Sepete Ekle</button>
                </div>
            </main>
        </body>
    </html>
    """

    elemanlar = DOMTreePruner.html_temizle_ve_buda(ornek_html)
    erisebilirlik_agaci = DOMTreePruner.agaci_metne_donustur(elemanlar)

    print("--- [Budanmış DOM Erişilebilirlik Ağacı / Accessibility Tree] ---")
    print(erisebilirlik_agaci)
    print("  ✓ Ham HTML'den script ve stiller ayıklandı, etkileşimli elemanlara ID atandı.")

    # -------------------------------------------------------------
    # ADIM 2: Otonom Tarayıcı Ajanının Gezinmesi
    # -------------------------------------------------------------
    print("\n[2/4] Tarayıcı Ajanı Otonom Arama ve Satın Alma Eylemlerini Yürütüyor...")
    ajan = WebBrowsingAgent(baslangic_html=ornek_html)

    hedef = "GPU Kartını ara ve sepete ekle"
    eylemler = [
        "Type[1, 'Yapay Zeka Hızlandırıcı GPU Kartı']",
        "Click[2]",
        "Click[3]",
        "Finish['Yapay Zeka Hızlandırıcı GPU Kartı 49.999 TL bedelle sepete eklendi.']",
    ]

    sonuc = ajan.otonom_gezin(hedef_gorev=hedef, eylem_adimlari=eylemler)

    print(f"  • Hedef Görev       : '{hedef}'")
    print(f"  • Görev Tamamlandı  : {sonuc['tamamlandi_mi']}")
    print(f"  • Toplam Adım Sayısı: {sonuc['adim_sayisi']}")
    print(f"  • Nihai Sonuç       : '{sonuc['nihai_yanit']}'")

    print("\n--- [Gezinme ve Eylem Günlüğü] ---")
    for r in sonuc["gezinme_raporu"]:
        print("  " + r)

    # -------------------------------------------------------------
    # ADIM 3: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[3/4] 6 Panelli Web Tarayıcı Teşhis Panosu Oluşturuluyor...")
    profil_raporu = DOMProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "web_tarayici_paneli.png")

    DOMGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Web Tarayıcı Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 227 (FAZ 12): WEB TARAYICI VE DOM AĞACI AJANI BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
