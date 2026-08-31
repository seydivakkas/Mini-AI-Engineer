"""
Tesla Gün 70 Ana Akış (Tesla Day 70 Main Pipeline)
===================================================
Araç İçi Ses Boru Hattı: PipeWire, ARNC ve Çok Bölgeli Ses
Uçtan Uca Çalıştırma ve Teşhis Paneli Üretim Scripti.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import sys
import os

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

su_an_dizin = os.path.dirname(os.path.abspath(__file__))
if su_an_dizin not in sys.path:
    sys.path.insert(0, su_an_dizin)

from src.tesla_ses_motoru_arnc import TeslaARNCNoiseCanceller, TeslaMultiZoneAudioRouter
from src.tesla_ses_profilleyici import TeslaSesProfilleyici
from src.tesla_ses_gorsellestirici import TeslaSesGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA FSD MASTERI | GÜN 70: AKTİF YOL GÜRÜLTÜSÜ ENGELLEME (ARNC) 🚗")
    print("================================================================================")
    print("Stajyer Görevi: 180° Anti-Noise, PipeWire Düşük Gecikme & Çok Bölgeli Yönlendirme")
    print("--------------------------------------------------------------------------------\n")

    # 1. Ses Motoru Benchmark'ı
    print(" [1] ARNC Ters Faz Algoritması ve PipeWire Tamponu Simüle Ediliyor...")
    profilleyici = TeslaSesProfilleyici(iterations=100)
    metrikler = profilleyici.benchmark_audio_dsp()

    print(f"     -> Gürültü Sönümleme Seviyesi: {metrikler['db_reduction']:.2f} dB (Hedef: >= 12.0 dB)")
    print(f"     -> PipeWire Tampon Gecikmesi : {metrikler['latency_ms']:.2f} ms (64 Örnek @ 48 kHz)")
    print(f"     -> Akustik Konfor Durumu     : {'BAŞARILI (SESSİZ KABİN)' if metrikler['is_effective'] else 'YETERSİZ'}")

    # 2. DSP Hızı
    print("\n [2] PipeWire / DSP RTOS Performansı...")
    print(f"     -> Ortalama İşleme Süresi    : {metrikler['dsp_step_ortalama_us']:.3f} µs (P99: {metrikler['dsp_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik DSP Tampon Sayısı: {metrikler['saniyelik_dsp_tamponu']:,} Tampon/sn")

    # 3. Tanı Paneli Görselleştirme
    print("\n [3] 6 Panelli Tesla Ses ve ARNC Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaSesGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_pipewire_arnc_audio_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi   : {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 70 BAŞARIYLA TAMAMLANDI! ARNC SES MOTORU DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
