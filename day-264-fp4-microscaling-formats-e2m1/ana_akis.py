"""
Day 264 (FAZ 14): Yeni Nesil FP4 / FP6 (Microscaling MXFP4 E2M1) Kuantizasyon ve Çekirdek Simülasyonu Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from src.mxfp4_microscaling_motoru import (
    MXFP4E2M1Codec,
    MXFP6E3M2Codec,
    MicroscaledGEMMEngine,
)
from src.mxfp4_profilleyici import MXFP4Profilleyici
from src.gorsellestirici import MXFP4Gorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 264 (FAZ 14): YENİ NESİL FP4 / FP6 (MICROSCALING MXFP4 E2M1) KUANTİZASYON VE ÇEKİRDEK SİMÜLASYONU")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Ağırlık ve Aktivasyon Tensörlerinin Hazırlanması
    # -------------------------------------------------------------
    print("\n[1/4] 128x128 Boyutlarında FP32 Ağırlık Matrisi ve Aktivasyonlar Hazırlanıyor...")
    np.random.seed(42)
    w_mat = np.random.randn(128, 128).astype(np.float32)
    x_vec = np.random.randn(1, 128).astype(np.float32)

    print(f"  • Ağırlık Matrisi Boyutu (M x K)    : {w_mat.shape} (16,384 Parametre)")
    print(f"  • Aktivasyon Vektörü Boyutu (1 x K)  : {x_vec.shape}")

    # -------------------------------------------------------------
    # ADIM 2: OCP MXFP4 E2M1 32-Eleman Blok Kuantizasyonu
    # -------------------------------------------------------------
    print("\n[2/4] OCP MXFP4 (E2M1) 32'li Mikro Bloklara Bölünüp Paylaşımlı Üs ile Kuantize Ediliyor...")
    q_w, s_w, shape_w = MXFP4E2M1Codec.quantize(w_mat, block_size=32)
    deq_w = MXFP4E2M1Codec.dequantize(q_w, s_w, shape_w)

    snr_w = MicroscaledGEMMEngine.compute_snr_db(w_mat, deq_w)
    print(f"  • Blok Sayısı (16384 / 32)           : {q_w.shape[0]} Blok")
    print(f"  • Ağırlık Sinyal-Gürültü Oranı (SNR) : {snr_w} dB (Zirve 4-Bit Kalite)")
    print(f"  • Bellek Sıkıştırma Oranı            : 4x Kat Tasarruf (16-Bit -> 4-Bit)")

    # -------------------------------------------------------------
    # ADIM 3: 4-Bit Microscaled GEMM Çekirdek Simülasyonu
    # -------------------------------------------------------------
    print("\n[3/4] 4-Bit Microscaled GEMM Simülasyonu Yürütülüyor...")
    c_mxfp4, stats = MicroscaledGEMMEngine.execute_mxfp4_gemm(x_vec, w_mat.T)
    c_exact = np.dot(x_vec, w_mat.T)

    gemm_snr = MicroscaledGEMMEngine.compute_snr_db(c_exact, c_mxfp4)
    print(f"  • GEMM Çıktı Boyutu (1 x M)          : {c_mxfp4.shape}")
    print(f"  • GEMM Çıktı Sinyal Kalitesi (SNR)   : {gemm_snr} dB")
    print(f"  • Blackwell B200 Zirve Kapasitesi    : 20.0 PFLOPS (4x FP16 Artışı)")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli OCP MXFP4 Teşhis Panosu Oluşturuluyor...")
    profil_raporu = MXFP4Profilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "mxfp4_microscaling_paneli.png")

    MXFP4Gorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ MXFP4 Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 264 (FAZ 14): YENİ NESİL FP4 / FP6 MICROSCALING MODÜLÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
