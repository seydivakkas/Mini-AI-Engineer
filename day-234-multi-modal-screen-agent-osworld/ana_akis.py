"""
Day 234: Çok Modlu Ekran Ajanı (Computer Use / OSWorld) Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.ekran_ajani_motoru import (
    ScreenElement,
    GUIAction,
    ComputerUseAgent,
)
from src.osworld_profilleyici import OSWorldProfilleyici
from src.gorsellestirici import EkranGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 234 (FAZ 12): ÇOK MODLU EKRAN AJANI (COMPUTER USE / OSWORLD) - FARE VE KLAVYE YÖNETİMİ")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Ekran Ajanı Başlatma ve Çözünürlük Ayarı
    # -------------------------------------------------------------
    print("\n[1/4] Çok Modlu Ekran Ajanı (Computer Use) Başlatılıyor...")
    agent = ComputerUseAgent(cozunurluk=(1920, 1080))
    print("  ✓ Görsel Ekran Görünümü (1920x1080) ve Piksel Eşleme Aktif.")

    # -------------------------------------------------------------
    # ADIM 2: Ekrandaki Görsel Bileşenlerin Tanımlanması
    # -------------------------------------------------------------
    print("\n[2/4] Ekran Görüntüsü Analiz Ediliyor ve Bileşen Koordinatları Çıkarılıyor...")
    agent.ekrana_bilesen_ekle(ScreenElement(1, "Excel_Ikon", 120, 850, 48, 48))
    agent.ekrana_bilesen_ekle(ScreenElement(2, "A1_Hucresi", 240, 220, 80, 25))
    agent.ekrana_bilesen_ekle(ScreenElement(3, "Kaydet_Butonu", 45, 65, 32, 32))

    for k, b in agent.aktif_bilesenler.items():
        mx, my = b.merkez_koordinati()
        print(f"  • Bileşen: [{b.etiket}] -> Kutu: (x={b.x}, y={b.y}, w={b.genislik}, h={b.yukseklik}) -> Hedef Tıklama: ({mx}, {my})")

    # -------------------------------------------------------------
    # ADIM 3: İlkel GUI Eylemlerinin İcrası
    # -------------------------------------------------------------
    print("\n[3/4] Masaüstü Görevi İcra Ediliyor (Excel Aç -> A1'e Yaz -> Kaydet)...")
    excel = agent.bilesen_bul("Excel_Ikon")
    hucre = agent.bilesen_bul("A1_Hucresi")

    gorev_plani = [
        GUIAction("DOUBLE_CLICK", x=excel.merkez_koordinati()[0], y=excel.merkez_koordinati()[1]),
        GUIAction("CLICK", x=hucre.merkez_koordinati()[0], y=hucre.merkez_koordinati()[1]),
        GUIAction("TYPE", metin="Q3 Gelir: 450.000 TL"),
        GUIAction("HOTKEY", tuslar=["Ctrl", "S"]),
    ]

    rapor = agent.gorevi_tamamla(gorev_plani)

    print("\n--- [İşletim Sistemi Seviyesi Eylem Günlüğü] ---")
    for r in rapor:
        print("  " + r["log"])

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Ekran Ajanı Teşhis Panosu Oluşturuluyor...")
    profil_raporu = OSWorldProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "ekran_ajani_paneli.png")

    EkranGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Ekran Ajanı Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 234 (FAZ 12): ÇOK MODLU EKRAN AJANI (COMPUTER USE) BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
