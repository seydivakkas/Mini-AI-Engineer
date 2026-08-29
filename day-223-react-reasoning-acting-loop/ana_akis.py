"""
Day 223: ReAct (Reasoning + Acting) Otonom Ajan Döngüsü Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.react_motoru import (
    ReActStep,
    ReActMemoryTrace,
    ReActAgent,
)
from src.react_profilleyici import ReActProfilleyici
from src.gorsellestirici import ReActGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 223 (FAZ 12): ReAct (REASONING + ACTING) DÜŞÜNCE-EYLEM-GÖZLEM OTONOM AJAN DÖNGÜSÜ")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: ReAct Ajanı ve Araçların Kaydı
    # -------------------------------------------------------------
    print("\n[1/4] ReAct Ajanı Başlatılıyor ve Dış Dünya Araçları Kaydediliyor...")
    ajan = ReActAgent()

    # Veritabanı / Web Arama Aracı
    ajan.arac_kaydet(
        "Arama",
        lambda sorgu: "2024 Verisi: MarsTech cirosu 250M $, LunarCorp cirosu 140M $ olarak açıklandı.",
    )
    # Matematiksel Hesaplayıcı
    ajan.arac_kaydet("Hesapla", lambda ifade: str(eval(ifade)))
    print("  ✓ Araçlar Kaydedildi: ['Arama', 'Hesapla', 'Finish']")

    # -------------------------------------------------------------
    # ADIM 2: Otonom ReAct Karar ve İcra Döngüsü
    # -------------------------------------------------------------
    print("\n[2/4] Çok Adımlı Problem İçin ReAct Döngüsü Başlatılıyor...")
    hedef = "MarsTech 2024'te LunarCorp'tan kaç milyon $ daha fazla ciro elde etti?"

    plan = [
        ("Kullanıcı ciro farkını soruyor. Önce şirketlerin 2024 cirolarını aramalıyım.", "Arama[MarsTech ve LunarCorp 2024 ciroları]"),
        ("Gözleme göre MarsTech 250M, LunarCorp 140M. Şimdi aradaki farkı hesaplayıcı ile bulacağım.", "Hesapla[250 - 140]"),
        ("Hesaplama 110 sonucunu verdi. Nihai yanıtı hazırlayıp görevi sonlandırıyorum.", "Finish[MarsTech şirketi, LunarCorp'tan 110 Milyon $ daha fazla ciro elde etti.]"),
    ]

    sonuc = ajan.otonom_coz(hedef_soru=hedef, simule_edilen_plan=plan, max_adim=5)

    print(f"  • Hedef Soru       : '{hedef}'")
    print(f"  • Toplam Adım Sayısı: {sonuc['toplam_adim']}")
    print(f"  • Görev Tamamlandı  : {sonuc['tamamlandi_mi']}")
    print(f"  • Nihai Çıktı      : '{sonuc['nihai_cevap']}'")

    print("\n--- [Ajan Bellek İzi / Memory Trace] ---")
    print(sonuc["hafiza_izi"])

    # -------------------------------------------------------------
    # ADIM 3: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("[3/4] 6 Panelli ReAct Ajan Teşhis Panosu Oluşturuluyor...")
    profil_raporu = ReActProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "react_ajan_paneli.png")

    ReActGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ ReAct Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 223 (FAZ 12): ReAct (REASONING + ACTING) OTONOM DÖNGÜSÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
