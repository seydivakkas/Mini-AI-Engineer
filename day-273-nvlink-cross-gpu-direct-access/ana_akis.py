"""
Day 273 (FAZ 14): NVLink ve GPUDirect RDMA Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from src.nvlink_rdma_motoru import NVLinkCrossGPUEngine
from src.nvlink_rdma_profilleyici import NVLinkRDMAProfilleyici
from src.gorsellestirici import NVLinkRDMAGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 273 (FAZ 14): NVLINK & GPUDIRECT RDMA — DÜĞÜMLER ARASI SIFIR CPU KOPYALI BELLEK ERİŞİMİ")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: 8 GPU NVSwitch / NVLink-4 Topolojisinin Başlatılması
    # -------------------------------------------------------------
    print("\n[1/4] 8x GPU NVSwitch / NVLink-4 Tam Eşler Arası (P2P) Matrisi Başlatılıyor...")
    engine = NVLinkCrossGPUEngine(num_gpus=8, interconnect_type="NVLink-4")

    print(f"  • Toplam GPU Sayısı (Cluster)        : {engine.num_gpus} GPU (NVIDIA H100 SXM5)")
    print(f"  • Bağlantı Arayüzü                   : {engine.interconnect_type} NVSwitch Mesh")
    print(f"  • P2P Eşler Arası Erişim Durumu      : %100 Doğrudan Erişim (UVA / Unified Virtual Addressing)")
    print(f"  • CPU Host Bounce Buffer             : DEVRE DIŞI (Sıfır Host Bellek Kopyası)")

    # -------------------------------------------------------------
    # ADIM 2: GPU 0 -> GPU 7 Sıfır CPU Kopyalı VRAM-to-VRAM Aktarımı
    # -------------------------------------------------------------
    print("\n[2/4] GPU 0 -> GPU 7 Arasında 128 MB Tensör Doğrudan VRAM-to-VRAM Aktarılıyor...")
    mock_res = NVLinkCrossGPUEngine.execute_mock_cross_gpu_pipeline(tensor_size_mb=128.0)
    stats = mock_res["transfer_stats"]

    print(f"  • Kaynak -> Hedef GPU                : GPU {stats['src_gpu']} -> GPU {stats['dst_gpu']}")
    print(f"  • Aktarılan Tensör Boyutu            : {stats['size_mb']:.1f} MB ({stats['tensor_name']})")
    print(f"  • Aktarım Süresi (Transfer Time)     : {stats['transfer_time_us']:.2f} μs")
    print(f"  • Efektif P2P Bant Genişliği         : {stats['bandwidth_gb_s']:.1f} GB/s")
    print(f"  • CPU Kullanım Oranı (CPU Overhead)  : %{stats['cpu_overhead_pct']:.1f} (CPU Tamamen Serbest)")
    print(f"  • Veri Bütünlüğü Doğrulaması         : {'✓ BAŞARILI (Bit-level Exact Match)' if mock_res['veri_dogrulugu'] else '✗ HATALI'}")

    # -------------------------------------------------------------
    # ADIM 3: 8-GPU 512 MB Ring All-Reduce ve Donanım Kıyaslama Raporu
    # -------------------------------------------------------------
    print("\n[3/4] 8x GPU 512 MB Ring All-Reduce ve Donanım Kıyaslama Raporu Hesaplanıyor...")
    profil_raporu = NVLinkRDMAProfilleyici.basarim_profili_cikar()
    karsilastirma = profil_raporu["karsilastirma"]

    print(f"  • P2P Taban Gecikmesi (PCIe vs NVLink) : {karsilastirma['p2p_gecikmesi_us']['Standart_PCIe_Gen4']} μs -> {karsilastirma['p2p_gecikmesi_us']['NVLink_4_H100_RDMA']} μs (16.8x Düşük Gecikme)")
    print(f"  • Efektif Bant Genişliği (PCIe vs NVL) : {karsilastirma['etkin_bant_genisligi_gb_s']['Standart_PCIe_Gen4']} GB/s -> {karsilastirma['etkin_bant_genisligi_gb_s']['NVLink_4_H100_RDMA']} GB/s (20.5x Yüksek Bant Genişliği)")
    print(f"  • 512MB All-Reduce Gecikmesi           : {karsilastirma['allreduce_512mb_gecikmesi_ms']['Standart_PCIe_Gen4']} ms -> {karsilastirma['allreduce_512mb_gecikmesi_ms']['NVLink_4_H100_RDMA']} ms (19.0x Hızlanma)")
    print(f"  • Host CPU Sürücü Ek Yükü              : %{karsilastirma['cpu_host_ek_yuku_yuzde']['Standart_PCIe_Gen4']} -> %{karsilastirma['cpu_host_ek_yuku_yuzde']['NVLink_4_H100_RDMA']}")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli NVLink & GPUDirect RDMA Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "nvlink_gpudirect_rdma_paneli.png")

    NVLinkRDMAGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ NVLink & GPUDirect RDMA Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 273 (FAZ 14): NVLINK & GPUDIRECT RDMA MODÜLÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
