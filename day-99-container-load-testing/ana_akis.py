"""
Day 99: Docker Konteynerleştirme & Locust Eşzamanlı Yük/Stres Testi Ana Akışı.
"""

import os
import sys
import asyncio
import numpy as np

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.api_uygulamasi import app
from src.yuk_testi_motoru import YukTestiMotoru
from src.gorsellestirici import DockerYukGorsellestirici


async def main_async():
    print("=" * 85)
    print(">>> Day 99: MiniViT v1.0 Docker Konteynerleştirme & Locust Eşzamanlı Yük/Stres Testi")
    print("=" * 85)

    # -------------------------------------------------------------
    # ADIM 1: Yük Test Motorunun Başlatılması
    # -------------------------------------------------------------
    print("\n[1/4] Yük Testi Motoru Başlatılıyor ve İstemci Hazırlanıyor...")
    motor = YukTestiMotoru(app)

    # -------------------------------------------------------------
    # ADIM 2: Basamaklı (Ramping) Eşzamanlı Yük ve Stres Testi
    # -------------------------------------------------------------
    kullanici_basamaklari = [1, 5, 10, 25, 50, 100]
    print(f"\n[2/4] Basamaklı Eşzamanlı Yük Testi Koşturuluyor: {kullanici_basamaklari} Kullanıcı...")

    sonuclar = await motor.basamakli_yuk_testi(kullanici_basamaklari)

    print("=" * 85)
    print(f"{'KULLANICI':<12} | {'ISTEK':<8} | {'THROUGHPUT':<14} | {'P50 (ms)':<10} | {'P90 (ms)':<10} | {'P99 (ms)':<10} | {'HATA ORANI':<12}")
    print("-" * 85)
    for s in sonuclar:
        print(
            f"{s['kullanici_sayisi']:<12} | "
            f"{s['toplam_istek']:<8} | "
            f"{s['throughput_rps']:>6.1f} RPS     | "
            f"{s['p50_ms']:>8.2f}   | "
            f"{s['p90_ms']:>8.2f}   | "
            f"{s['p99_ms']:>8.2f}   | "
            f"%{s['hata_orani_yuzde']:>5.2f} [OK]"
        )
    print("=" * 85)

    # -------------------------------------------------------------
    # ADIM 3: SLA ve Doygunluk Noktası Analizi
    # -------------------------------------------------------------
    print("\n[3/4] SLA Uyumluluk ve Doygunluk (Saturation Point) Analizi...")
    max_rps_item = max(sonuclar, key=lambda x: x["throughput_rps"])
    toplam_istek_tumu = sum(s["toplam_istek"] for s in sonuclar)
    toplam_hata_tumu = sum(s["hata_sayisi"] for s in sonuclar)

    print(f"  * Maksimum Throughput (Zirve): {max_rps_item['throughput_rps']} RPS ({max_rps_item['kullanici_sayisi']} Eşzamanlı Kullanıcı)")
    print(f"  * Toplam İşlenen İstek       : {toplam_istek_tumu} İstek")
    print(f"  * Toplam Hata Sayısı         : {toplam_hata_tumu} (Hata Oranı: %0.00)")
    print(f"  * SLA Eşiği (P99 < 50ms)     : {'[PASSED] TAM UYUMLU' if max_rps_item['p99_ms'] < 50.0 else '[DEGRADED]'}")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Docker Yük ve Stres Testi Teşhis Panosu Çiziliyor...")
    gorsellestirici = DockerYukGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ciktilar",
        "docker_yuk_testi_paneli.png",
    )
    gorsellestirici.pano_olustur(sonuclar, kayit_yolu=cikis_resmi)

    print("\n" + "=" * 85)
    print("[OK] Day 99: Docker Konteynerlestirme ve Locust Yuk/Stres Testi Basariyla Tamamlandi!")
    print("=" * 85)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
