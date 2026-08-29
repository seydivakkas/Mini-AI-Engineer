"""
Day 166: GUI Ajanları ve Web Gezintisi (GUI Agent & Web Navigation VLM) Ana Akışı.
Set-of-Mark (SoM) Görsel İşaretleme, Eylem Uzayı Ayrıştırma ve Otonom Görev İcrası.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.set_of_mark_isaretleyici import SetOfMarkIsaretleyici
from src.otonom_web_ajani import OtonomWebAjani
from src.gorsellestirici import GUIAjanGorsellestirici


def main():
    print("=" * 110)
    print(">>> Day 166 (FAZ 9): GUI AGENT & WEB NAVIGATION: SET-OF-MARK (SoM) & AUTONOMOUS ACTION PLANNING")
    print("=" * 110)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # -------------------------------------------------------------
    # 1. SET-OF-MARK ELEMANLARI
    # -------------------------------------------------------------
    print("\n[1/3] Ekran Görüntüsü Elemanları SoM ile Numaralandırılıyor...")
    elemanlar = SetOfMarkIsaretleyici.ornek_sayfa_elemanlarini_getir()
    for e in elemanlar:
        print(f"  • Mark [{e['id']}] : {e['eleman_tipi']:<30} | '{e['etiket']}' @ Merkez {e['merkez']}")

    # -------------------------------------------------------------
    # 2. OTONOM WEB GÖREVLERİNİN İCRASI
    # -------------------------------------------------------------
    print("\n[2/3] Çok Adımlı Web Görevleri (Arama & E-Ticaret) İcra Ediliyor...")
    rapor = OtonomWebAjani.gorevleri_yurut_ve_degerlendir()

    for g in rapor["gorev_raporlari"]:
        print(f"\n>> GÖREV: '{g['hedef']}' ({g['toplam_adim']} Adım)")
        for adim in g["adim_detaylari"]:
            print(f"   [Adım {adim['adim']}] {adim['ekran']:<25} ──> {adim['eylem_metni']:<25} [{adim['durum']}]")

    print("\n" + "-" * 80)
    print(f"{'Metrik':<35} | {'Değer'}")
    print("-" * 80)
    print(f"{'Toplam Görev':<35} | {rapor['toplam_gorev_sayisi']}")
    print(f"{'Toplam İcra Edilen Adım':<35} | {rapor['toplam_adim_sayisi']}")
    print(f"{'Adım Doğruluk Oranı':<35} | %{rapor['adim_basari_yuzdesi']}")
    print(f"{'Görev Tamamlama Oranı':<35} | %{rapor['gorev_tamamlama_orani']}")
    print("-" * 80)

    # -------------------------------------------------------------
    # 3. 6 PANELLİ TEŞHİS PANOSU ÜRETİMİ
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli GUI Ajanı Teşhis Panosu Üretiliyor...")
    gorsellestirici = GUIAjanGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "gui_agent_web_navigation_paneli.png")
    gorsellestirici.pano_olustur(rapor, kayit_yolu=cikis_resmi)

    print("\n" + "=" * 110)
    print("✓ Day 166: GUI AGENT & WEB NAVIGATION BAŞARIYLA TAMAMLANDI!")
    print("=" * 110)


if __name__ == "__main__":
    main()
