"""
Day 196: Ray Cluster & Ray Serve Dağıtık Model Dağıtım Ana Çalıştırma Akışı.
"""

import os
import sys
import numpy as np
import torch

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.ray_serve_motoru import (
    RayClusterNode,
    RayServeDeploymentManager,
    RayServeRouter,
)
from src.ray_profilleyici import RayClusterYukProfilleyici
from src.gorsellestirici import RayServeGorsellestirici


def main():
    print("=" * 110)
    print(">>> Day 196 (FAZ 10): RAY CLUSTER & RAY SERVE DISTRIBUTED MODEL SERVING")
    print("=" * 110)

    # -------------------------------------------------------------
    # ADIM 1: Ray Kümesi Düğüm Topolojisi Kurulumu
    # -------------------------------------------------------------
    print("\n[1/4] Ray Kümesi Düğüm Topolojisi Başlatılıyor...")
    manager = RayServeDeploymentManager(min_replicas=2, max_replicas=8, target_ongoing_requests=5)
    manager.dugum_ekle(RayClusterNode("head_node", "10.0.0.1", gpu_count=4))
    manager.dugum_ekle(RayClusterNode("worker_node_1", "10.0.0.2", gpu_count=4))
    manager.dugum_ekle(RayClusterNode("worker_node_2", "10.0.0.3", gpu_count=4))

    print(f"  • Eklenen Düğüm Sayısı        : {len(manager.nodes)} (1 Head + 2 Worker)")
    print(f"  • Toplam Küme GPU Kapasitesi  : 12x NVIDIA A100/H100 GPU")

    # -------------------------------------------------------------
    # ADIM 2: Başlangıç Replikalarının Dağıtımı
    # -------------------------------------------------------------
    print("\n[2/4] Ray Serve Model Replikaları Başlatılıyor (@serve.deployment)...")
    manager.baslat_varsayilan_replikalar()
    print(f"  • Başlangıç Aktif Replika     : {len([r for r in manager.replicas if r.is_active])} Ray Actor")
    for r in manager.replicas:
        print(f"    - Replika ID: {r.replica_id:<12} | Düğüm: {r.node_id:<14} | GPU ID: {r.gpu_id}")
    print("  ✓ Başlangıç Replikaları Başarıyla Hazırlandı!")

    # -------------------------------------------------------------
    # ADIM 3: Çok Fazlı Trafik Yükü ve Otomatik Ölçekleme (Autoscaling)
    # -------------------------------------------------------------
    print("\n[3/4] Trafik Patlaması ve Ray Autoscaler Yük Testi Simüle Ediliyor...")
    simulasyon_raporu = RayClusterYukProfilleyici.kume_yuk_simulasyonu_calistir()

    print("-" * 110)
    print(f"{'Trafik Fazı':<28} | {'İstek':<8} | {'Aktif Replika':<14} | {'P50 (ms)':<12} | {'P95 (ms)':<12} | {'P99 (ms)':<12} | {'GPU Verimi'}")
    print("-" * 110)
    for f in simulasyon_raporu["faz_raporlari"]:
        print(
            f"{f['faz_adi']:<28} | "
            f"{f['istek_sayisi']:>5}   | "
            f"{f['aktif_replika_sayisi']:>7} Aktör     | "
            f"{f['p50_gecikme_ms']:>8.1f} ms | "
            f"{f['p95_gecikme_ms']:>8.1f} ms | "
            f"{f['p99_gecikme_ms']:>8.1f} ms | "
            f"%{f['kume_gpu_kullanimi_yuzde']:>6.1f}"
        )
    print("-" * 110)

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Görsel Teşhis Panosu Üretimi
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Ray Serve Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "ray_serve_paneli.png")

    RayServeGorsellestirici.teshis_paneli_olustur(
        simulasyon_raporu=simulasyon_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Ray Serve Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 110)
    print("✓ Day 196: RAY CLUSTER & RAY SERVE BAŞARIYLA TAMAMLANDI!")
    print("=" * 110)


if __name__ == "__main__":
    main()
