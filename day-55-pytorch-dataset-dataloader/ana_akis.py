import os
import sys
import multiprocessing

# Windows multiprocessing ve sys.path uyumluluğu
MEVCUT_DIZIN = os.path.abspath(os.path.dirname(__file__))
if MEVCUT_DIZIN not in sys.path:
    sys.path.insert(0, MEVCUT_DIZIN)

from src.veri_seti_motoru import HizliSentetikGorselVeriSeti
from src.darbogaz_olcer import DataLoaderBenchmarkEngine
from src.gorsellestirici import DataLoaderAnalizGorsellestirici


def main():
    multiprocessing.freeze_support()
    print("=" * 85, flush=True)
    print(">>> DAY 55: İLERİ PYTORCH DATALOADER, NUM_WORKERS VE PIN_MEMORY OPTİMİZASYONU", flush=True)
    print("=" * 85, flush=True)

    # 1. Hızlı Sentetik Görsel Veri Setinin Hazırlanması
    print("\n[+] 1. Adım: C-Bitişik Bellek ve I/O Simülasyonlu Veri Seti Başlatılıyor...", flush=True)
    dataset = HizliSentetikGorselVeriSeti(
        num_samples=640,
        channels=3,
        height=64,
        width=64,
        num_classes=10,
        simule_io_ms=0.2,
        seed=42
    )
    print(f"    - Veri Seti Boyutu   : {len(dataset)} Örnek (3x64x64 Tensörler)", flush=True)
    print(f"    - Simüle Edilen I/O  : 0.2 ms / örnek", flush=True)

    # 2. 4 Ana Konfigürasyon Arasında Karşılaştırmalı Benchmark
    print("\n[+] 2. Adım: 4 Farklı DataLoader Konfigürasyonunun Kıyaslanması (Batch Size = 64)...", flush=True)
    benchmark_sonuclari = DataLoaderBenchmarkEngine.karsilastirmali_benchmark(
        dataset=dataset,
        batch_size=64,
        num_batches=8
    )

    print("\n" + "-" * 85, flush=True)
    print(f"{'KONFİGÜRASYON ADI':<38} | {'İŞLEME HIZI':<15} | {'TOPLAM SÜRE':<12} | {'HIZLANMA':<10}", flush=True)
    print("-" * 85, flush=True)
    for b in benchmark_sonuclari:
        print(f"{b['ad']:<38} | {b['isleme_hizi_ornek_sn']:>6.1f} örnek/s  | {b['toplam_sure_sn']:>6.2f} sn     | {b['hizlanma_carpani']:>5.2f}x", flush=True)
    print("-" * 85, flush=True)

    # 3. num_workers Ölçeklenme Taraması (0, 1, 2, 4 Workers)
    print("\n[+] 3. Adım: num_workers (0, 1, 2, 4) CPU-GPU Denge Eğrisi Taranıyor...", flush=True)
    worker_sonuclari = DataLoaderBenchmarkEngine.worker_olceklenme_taramasi(
        dataset=dataset,
        batch_size=64,
        worker_listesi=[0, 1, 2, 4],
        num_batches=6
    )

    for w in worker_sonuclari:
        print(f"    - Workers={w['num_workers']}: Hız = {w['isleme_hizi_ornek_sn']:>6.1f} örnek/s | Süre = {w['toplam_sure_sn']:.2f} sn", flush=True)

    # 4. 6 Panelli Teşhis Panosunun Çizilmesi
    print("\n" + "=" * 85, flush=True)
    print(">>> 4. 6 PANELLİ DATALOADER PERFORMANS VE DARBOĞAZ PANOSUNUN ÜRETİLMESİ", flush=True)
    print("=" * 85, flush=True)

    cikis_yolu = DataLoaderAnalizGorsellestirici.panel_ciz(
        benchmark_sonuclari=benchmark_sonuclari,
        worker_tarama_sonuclari=worker_sonuclari,
        hedef_path="ciktilar/dataloader_darbogaz_paneli.png"
    )
    print(f"[+] 6 Panelli Teşhis Panosu Kaydedildi: {os.path.abspath(cikis_yolu)}", flush=True)
    print("=" * 85, flush=True)
    print("DAY 55: İLERİ PYTORCH DATALOADER VE DARBOĞAZ MOTORU BAŞARIYLA TAMAMLANDI!", flush=True)
    print("=" * 85, flush=True)


if __name__ == "__main__":
    main()
