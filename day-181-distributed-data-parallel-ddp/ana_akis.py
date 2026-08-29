"""
Day 181 (FAZ 10 BAŞLANGICI): Distributed Data Parallel (DDP) Ana Akış Motoru.
PyTorch DDP: Ring All-Reduce İletişimi, Gradient Bucketing ve Çoklu GPU Dağıtık Eğitimi.
"""

import sys
import os
import torch
import torch.nn as nn

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Modül yolunu ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ddp_iletisim_motoru import RingAllReduceSimulasyonu, GradyanPaketleyici
from src.dagitik_egitim_dongusu import DDPVeriOrnekleyici, DDPModelSarmalayici
from src.gorsellestirici import DDPGorsellestirici


def main():
    print("=" * 110)
    print(">>> Day 181 (FAZ 10 BAŞLANGICI): DISTRIBUTED DATA PARALLEL (DDP) & RING ALL-REDUCE ENGINE")
    print("=" * 110)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # 1. Ring All-Reduce İletişim Algoritması Doğrulaması (4 GPU Rank)
    num_ranks = 4
    tensor_dim = 1000
    torch.manual_seed(42)

    print(f"\n[1/4] Ring All-Reduce İletişim Simülasyonu ({num_ranks} GPU Rank'i Arasında Başlatılıyor)...")
    # Her rank için farklı gradyan tensörleri
    rank_grads = [torch.randn(tensor_dim) * (r + 1.0) for r in range(num_ranks)]

    ring_engine = RingAllReduceSimulasyonu(num_ranks=num_ranks)
    synced_grads, ring_stats = ring_engine.all_reduce(rank_grads)

    # Matematiksel referans: Basit aritmetik ortalama
    expected_avg = sum(rank_grads) / num_ranks
    fark = torch.norm(synced_grads[0] - expected_avg).item()

    print(f"  ✓ Ring All-Reduce Başarılı! (Matematiksel Hata: {fark:.2e})")
    print(f"  • Toplam Ring Adımı     : {ring_stats['toplam_ring_adimi']} Adım ({ring_stats['scatter_reduce_adim']} Scatter + {ring_stats['all_gather_adim']} Gather)")
    print(f"  • Ring Transfer Hacmi   : {ring_stats['ring_transfer_mb']} MB")
    print(f"  • Parameter Server Hacmi: {ring_stats['ps_merkezi_transfer_mb']} MB")
    print(f"  • Avantaj               : {ring_stats['bant_genisligi_avantaji']}")

    # 2. Gradient Bucketing (25 MB Havuzlama) Analizi
    print(f"\n[2/4] Model Parametreleri İçin Gradient Bucketing (Paketleme) Analizi...")
    # Örnek derin sinir ağı (3 katmanlı MLP)
    model = nn.Sequential(
        nn.Linear(256, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 10),
    )

    params = list(model.parameters())
    bucket_stats = GradyanPaketleyici.paket_istatistikleri_hesapla(params, bucket_cap_mb=25.0)
    print(f"  • Toplam Parametre Sayısı: {bucket_stats['toplam_parametre_adedi']:,} ({bucket_stats['toplam_model_mb']} MB)")
    print(f"  • Ayrı Tensör Sayısı     : {bucket_stats['toplam_tensor_sayisi']} adet")
    print(f"  • Bucket Sayısı          : {bucket_stats['bucket_sayisi']} adet (Kapasite: {bucket_stats['bucket_kapasite_mb']} MB)")
    print(f"  • IPC Çağrı Tasarrufu    : {bucket_stats['ipc_cagri_azalmasi']} ({bucket_stats['tahmini_gecikme_iyilesmesi']})")

    # 3. DDP Dağıtık Eğitim Döngüsü ve Senkronizasyon Simülasyonu
    print(f"\n[3/4] 4 GPU Rank'i Üzerinde DDP Dağıtık Eğitim Adımı İşletiliyor...")
    ddp_wrapper = DDPModelSarmalayici(model, num_ranks=num_ranks, lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # 4 Rank için farklı batch girdileri
    rank_inputs = [torch.randn(16, 256) for _ in range(num_ranks)]
    rank_targets = [torch.randint(0, 10, (16,)) for _ in range(num_ranks)]

    egitim_res = ddp_wrapper.egitim_adimi(rank_inputs, rank_targets, criterion)
    print(f"  ✓ Dağıtık Eğitim Adımı Tamamlandı!")
    print(f"  • Ortalama Mini-Batch Kaybı : {egitim_res['ortalama_kayip']}")
    print(f"  • Rank Başına Kayıplar     : {egitim_res['rank_kayiplari']}")
    print(f"  • Ağırlık Senkron Farkı    : {egitim_res['agirlik_senkron_farki']:.2e} (Senkron: {egitim_res['senkronizasyon_basarili']})")

    # 4. Çoklu GPU Ölçeklenebilirlik Raporu & 6 Panelli Görselleştirme
    print(f"\n[4/4] 6 Panelli DDP Dağıtık Eğitim Teşhis Panosu Oluşturuluyor...")
    olcek_raporu = DDPModelSarmalayici.ornek_ddp_olceklenme_raporu()
    for item in olcek_raporu["karsilastirma"]:
        print(f"  • {item['gpu_sayisi']:>2} GPU -> {item['hiz_imgs_per_sec']:>7.1f} imgs/s (Verim: %{item['verimlilik_yuzde']:>5.1f}) | {item['tip']}")

    gorsellestirici = DDPGorsellestirici(dpi=300)
    cikti_resmi = os.path.join(cikis_dizini, "distributed_data_parallel_ddp_paneli.png")
    gorsellestirici.pano_olustur(
        olcek_raporu=olcek_raporu,
        bucket_raporu=bucket_stats,
        egitim_logu=egitim_res,
        kayit_yolu=cikti_resmi,
    )

    print("\n" + "=" * 110)
    print("✓ Day 181: DISTRIBUTED DATA PARALLEL (DDP) BAŞARIYLA TAMAMLANDI! (FAZ 10 BAŞLADI)")
    print("=" * 110)


if __name__ == "__main__":
    main()
