"""
Day 168: Gerçek Zamanlı Video Akışı Analizi ve Olay Tespiti (Streaming VLM) Ana Akışı.
Kayan Bellek Kuyruğu (Ring Buffer), Anomali Dedektörü ve Online VLM Alarm Üretimi.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.streaming_vlm_motoru import StreamingVLMMotoru
from src.gorsellestirici import StreamingGorsellestirici


def main():
    print("=" * 110)
    print(">>> Day 168 (FAZ 9): STREAMING VLM: REAL-TIME VIDEO STREAM UNDERSTANDING & ONLINE EVENT DETECTION")
    print("=" * 110)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # -------------------------------------------------------------
    # 1. CANLI AKIŞ VE OLAY TESPİTİ SİMÜLASYONU
    # -------------------------------------------------------------
    print("\n[1/2] 30 Saniyelik Canlı Güvenlik Kamera Akışı İşleniyor...")
    rapor = StreamingVLMMotoru.canli_akis_simulasyonunu_calistir()

    print(f"  • Toplam İşlenen Süre       : {rapor['toplam_islenen_saniye']} Saniye")
    print(f"  • Tetiklenen Kritik Olay    : {rapor['toplam_tetiklenen_olay']} Adet")
    print(f"  • Olay Tespit Doğruluğu     : %{rapor['dogruluk_yuzdesi']}")

    print("\nTETİKLENEN VLM GÜVENLİK ALARMLARI:")
    print("-" * 105)
    print(f"{'Zaman':<10} | {'Anomali Skoru':<15} | {'Alarm Seviyesi':<18} | {'VLM Olay Açıklaması'}")
    print("-" * 105)
    for o in rapor["olay_gunlugu"]:
        print(
            f"t={int(o['zaman_damgasi'])}s{'':<6} | {o['anomali_skoru']:<15.2f} | "
            f"{o['alarm_seviyesi']:<18} | {o['aciklama']}"
        )
    print("-" * 105)

    # -------------------------------------------------------------
    # 2. 6 PANELLİ TEŞHİS PANOSU ÜRETİMİ
    # -------------------------------------------------------------
    print("\n[2/2] 6 Panelli Streaming VLM Teşhis Panosu Üretiliyor...")
    gorsellestirici = StreamingGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "streaming_video_understanding_paneli.png")
    gorsellestirici.pano_olustur(rapor, kayit_yolu=cikis_resmi)

    print("\n" + "=" * 110)
    print("✓ Day 168: STREAMING VIDEO UNDERSTANDING BAŞARIYLA TAMAMLANDI!")
    print("=" * 110)


if __name__ == "__main__":
    main()
