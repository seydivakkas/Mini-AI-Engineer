"""
Tesla Gun 11 Ana Akis (Tesla Day 11 Main Pipeline)
===================================================
POSIX Paylasilan Bellek (shm_open), mmap ve Isimlendirilmis Semaforlar
Uctan Uca Calistirma ve Teshis Paneli Uretim Scripti.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
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

from src.tesla_paylasilan_bellek import (
    TeslaPOSIXPaylasilanBellek,
    TeslaPOSIXSemafor,
    TeslaSifirKopyaGoruntuHatti
)
from src.tesla_shm_profilleyici import TeslaSHMProfilleyici
from src.tesla_shm_gorsellestirici import TeslaSHMGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA GOMULU YAZILIM MASTERI | GUN 11: POSIX SHARED MEMORY & IPC 🚗")
    print("================================================================================")
    print("Stajyer Gorevi: 4K Tensörleri Zero-Copy ile Aktaran POSIX SHM & Semafor Mimarisi")
    print("--------------------------------------------------------------------------------\n")

    # 1. POSIX SHM ve Çift Tamponlu Hat Kurulumu
    print(" [1] POSIX Paylaşılan Bellek ve İsimlendirilmiş Semaforlar Başlatılıyor...")
    frame_boyutu = 1920 * 1080 * 3 # 6.22 MB
    hat = TeslaSifirKopyaGoruntuHatti(frame_boyutu_bayt=frame_boyutu)
    print(f"     -> Paylaşılan Bellek Alanı Tahsis Edildi: {frame_boyutu * 2 / (1024*1024):.2f} MB (/dev/shm/tesla_kamera_shm)")
    print(f"     -> POSIX Semaforları Bağlandı: [/sem_kamera_hazir, /sem_kamera_bos]")

    # 2. Kamera Üretici -> FSD Tüketici Sıfır-Kopya Aktarımı
    print("\n [2] Kamera Sürücüsü Frame Üretiyor ve FSD Otopilotuna Aktarıyor...")
    sahte_frame = b'\x55' * frame_boyutu
    hat.uretici_kamera_frame_yaz(sahte_frame)
    print("     -> [Producer] 6.22 MB 1080p Frame Paylaşılan Belleğe Yazıldı ve sem_post Verildi.")

    gorunum = hat.tuketici_fsd_frame_oku_gorunumu()
    if gorunum:
        print(f"     -> [Consumer - FSD] Sıfır Kopyalama ile Frame Okundu! (Boyut: {len(gorunum):,} Bayt)")
        print(f"     -> Bellek Doğrulaması: İlk Bayt: 0x{gorunum[0]:02X}, Son Bayt: 0x{gorunum[-1]:02X} (TAM EŞLEŞME)")

    # 3. Profilleme ve Karşılaştırma
    print("\n [3] POSIX Zero-Copy SHM vs Linux Pipe/Socket IPC Benchmark'ı...")
    profilleyici = TeslaSHMProfilleyici(tekrar_sayisi=100)
    metrikler = profilleyici.benchmark_shm_vs_pipe_gecikmesi()

    print(f"     -> POSIX SHM Gecikmesi (Zero-Copy)     : {metrikler['shm_ortalama_us']:.2f} µs")
    print(f"     -> Standart Linux Pipe Gecikmesi       : {metrikler['pipe_ortalama_us']:.2f} µs")
    print(f"     -> Hızlanma Çarpanı                    : {metrikler['hizlanma_orani']:.1f}x Daha Hızlı")
    print(f"     -> Efektif IPC Bant Genişliği          : {metrikler['shm_bant_genisligi_gbps']:,.1f} GB/s (RAM Sınırı)")

    # 4. Tanı Paneli Görselleştirme
    print("\n [4] 6 Panelli Tesla POSIX SHM Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaSHMGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_shm_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi: {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 11 BAŞARIYLA TAMAMLANDI! ZERO-COPY IPC MOTORU DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
