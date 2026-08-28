"""
Day 97: MiniViT v1.0 Deterministik Çıkarım ve Donanımdan Bağımsız Doğrulama Ana Akışı.
Bit-level determinizmi, CPU vs GPU paritesini ve FP32/FP16/BF16 sayısal sapmasını doğrular.
"""

import os
import sys
import time
import torch
import numpy as np

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.konfigurasyon import MiniViTConfig
from src.model import MiniViTForImageClassification
from src.determinizm_yoneticisi import (
    DeterminizmOrtami,
    BitHashHesaplayici,
    DeterminizmDenetleyicisi,
)
from src.capraz_donanim_motoru import CaprazDonanimDogrulayici, HassasiyetKiyaslayici
from src.gorsellestirici import DeterminizmGorsellestirici


def main():
    print("=" * 85)
    print(">>> Day 97: MiniViT v1.0 Deterministik Çıkarım & Donanım Doğrulama Testi")
    print("=" * 85)

    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Çalışma Ortamı Cihazı: {cihaz.type.upper()}")

    # -------------------------------------------------------------
    # ADIM 1: Determinizm Ortamının Kurulması ve Seed Sabitleme
    # -------------------------------------------------------------
    print("\n[1/5] Determinizm Ortamı Yapılandırılıyor (Seed=42, CUBLAS, cuDNN)...")
    determinizm_ortami = DeterminizmOrtami(seed=42)
    determinizm_ortami.etkinlestir()

    config = MiniViTConfig(
        goruntu_boyutu=32,
        yama_boyutu=4,
        kanal_sayisi=3,
        gizli_boyut=128,
        katman_sayisi=4,
        dikkat_baslik_sayisi=4,
        ileri_besleme_boyutu=256,
        sinif_sayisi=10,
    )
    model = MiniViTForImageClassification(config).to(cihaz)
    model.eval()

    toplam_param = sum(p.numel() for p in model.parameters())
    print(f"  * Model Parametre Sayısı : {toplam_param:,}")
    print(f"  * CUBLAS Workspace Config: {os.environ.get('CUBLAS_WORKSPACE_CONFIG')}")
    print(f"  * Deterministic Algo     : {torch.are_deterministic_algorithms_enabled()}")

    # -------------------------------------------------------------
    # ADIM 2: Ardışık 100 Çıkarımda Bit-Level Determinizm Testi
    # -------------------------------------------------------------
    print("\n[2/5] Ardışık 100 Çıkarımda Bit-Level Determinizm Test Ediliyor...")
    denetleyici = DeterminizmDenetleyicisi(model)

    torch.manual_seed(42)
    test_girdisi = torch.randn(1, 3, 32, 32, device=cihaz)

    det_sonucu = denetleyici.ardil_cikarim_testi(test_girdisi, tekrar_sayisi=100)

    print("=" * 85)
    print("[-] DETERMINIZM DENETLEME RAPORU")
    print("=" * 85)
    print(f"  * Test Edilen İterasyon   : {det_sonucu['tekrar_sayisi']}")
    print(f"  * Benzersiz SHA-256 Hash  : {det_sonucu['benzersiz_hash_sayisi']}")
    print(f"  * Örnek Çıktı Hash'i      : {det_sonucu['ornek_hash']}")
    print(f"  * Global Maksimum Sapma   : {det_sonucu['global_maks_sapma']:.2e}")
    durum_str = "[PASSED] %100 BIT-LEVEL DETERMINISTIC" if det_sonucu["tam_deterministik"] else "[FAILED]"
    print(f"  * Determinizm Durumu      : {durum_str}")

    # -------------------------------------------------------------
    # ADIM 3: Çapraz Donanım Doğrulama (CPU vs GPU Paritesi)
    # -------------------------------------------------------------
    print("\n[3/5] Çapraz Donanım Paritesi Ölçülüyor (CPU vs GPU)...")
    dogrulayici = CaprazDonanimDogrulayici(model)
    parite_sonucu = dogrulayici.cpu_gpu_parite_testi(test_girdisi, tolerans_linf=1e-4)

    print("=" * 85)
    print("[-] CAPRAZ DONANIM PARITE RAPORU")
    print("=" * 85)
    print(f"  * Donanım Desteği (CUDA)  : {'Mevcut' if parite_sonucu['has_cuda'] else 'Yok (Simüle)'}")
    print(f"  * L1 Ortalama Hata        : {parite_sonucu['l1_hata']:.2e}")
    print(f"  * L2 (RMSE) Hata          : {parite_sonucu['l2_hata']:.2e}")
    print(f"  * L_inf (Maksimum) Hata   : {parite_sonucu['linf_hata']:.2e}")
    print(f"  * Kosinüs Benzerliği      : {parite_sonucu['kosinus_benzerligi']:.8f}")
    p_str = "[PASSED] PARITE UYUMLU" if parite_sonucu["parite_uyumlu"] else "[FAILED]"
    print(f"  * Parite Uyumluluk        : {p_str}")

    # -------------------------------------------------------------
    # ADIM 4: Sayısal Hassasiyet (FP32 vs FP16 vs BF16) Analizi
    # -------------------------------------------------------------
    print("\n[4/5] Sayısal Hassasiyet Sapması (Precision Drift) ve Gecikme Ölçülüyor...")
    kiyaslayici = HassasiyetKiyaslayici(model)
    hassasiyet_sonucu = kiyaslayici.hassasiyet_karsilastir(test_girdisi, iterasyon=30)

    print("=" * 85)
    print("[-] HASSASIYET KIYASLAMA VE GECIKME RAPORU")
    print("=" * 85)
    print(f"  * FP16 Maksimum Sapma (L_inf) : {hassasiyet_sonucu['linf_fp16']:.2e}")
    print(f"  * FP16 Sinyal-Gürültü Oranı   : {hassasiyet_sonucu['snr_fp16_db']:.2f} dB")
    print(f"  * BF16 Maksimum Sapma (L_inf) : {hassasiyet_sonucu['linf_bf16']:.2e}")
    print(f"  * BF16 Sinyal-Gürültü Oranı   : {hassasiyet_sonucu['snr_bf16_db']:.2f} dB")
    print("  * Gecikme Kıyaslaması (P50)   :")
    for d_name, lat in hassasiyet_sonucu["gecikmeler_ms"].items():
        print(f"    - {d_name:<6}: {lat:.2f} ms")

    # -------------------------------------------------------------
    # ADIM 5: Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[5/5] 6 Panelli Teşhis Panosu Oluşturuluyor...")
    gorsellestirici = DeterminizmGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ciktilar",
        "deterministik_cikarim_paneli.png",
    )
    gorsellestirici.pano_olustur(det_sonucu, parite_sonucu, hassasiyet_sonucu, kayit_yolu=cikis_resmi)

    print("\n" + "=" * 85)
    print("[OK] Day 97: Deterministik Cikarim ve Donanim Dogrulama Testleri Basariyla Tamamlandi!")
    print("=" * 85)


if __name__ == "__main__":
    main()
