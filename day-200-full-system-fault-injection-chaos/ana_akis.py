"""
Day 200: Kaos Mühendisliği - GPU Arızaları, Ağ Gecikmesi ve Kurtarma Testi Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.chaos_motoru import (
    NodeState,
    GPUClusterNode,
    ChaosInjector,
    ResilientClusterManager,
)
from src.chaos_profilleyici import ChaosDeneyProfilleyici
from src.gorsellestirici import ChaosGorsellestirici


def main():
    print("=" * 110)
    print(">>> Day 200 (FAZ 10): FULL SYSTEM FAULT INJECTION & CHAOS ENGINEERING (GPU & NETWORK RESILIENCE)")
    print("=" * 110)

    # -------------------------------------------------------------
    # ADIM 1: Küme Başlatma ve Temiz Durum Doğrulaması
    # -------------------------------------------------------------
    print("\n[1/4] Dağıtık GPU Çıkarım Kümesi (4 Düğüm) Başlatılıyor...")
    cluster = ResilientClusterManager(initial_node_count=4)
    saglikli_dugumler = cluster.get_healthy_nodes()
    print(f"  • Aktif Sağlıklı GPU Düğüm Sayısı: {len(saglikli_dugumler)} Düğüm")
    for node in saglikli_dugumler:
        print(f"    - [{node.node_id}] GPU #{node.gpu_id} | Durum: {node.state.value} | Temel Gecikme: {node.base_latency_ms:.1f} ms")

    # -------------------------------------------------------------
    # ADIM 2: 3 Dalgalı Kaos Mühendisliği Deney Yürütümü
    # -------------------------------------------------------------
    print("\n[2/4] 3 Dalgalı Kaos Mühendisliği Arıza Enjeksiyonu Başlatılıyor...")
    print("  • 1. Dalga: GPU-1'e CUDA Out of Memory (OOM) Arızası Enjeksiyonu")
    print("  • 2. Dalga: GPU-2'ye InfiniBand +120ms Ağ Gecikmesi ve Jitter Enjeksiyonu")
    print("  • 3. Dalga: GPU-3'e Hard Kill (Kernel Panic / Pod Çökmesi) ve Otomatik İyileşme (Self-Healing)")

    deney_raporu = ChaosDeneyProfilleyici.tam_kaos_deneyi_yurut()

    # -------------------------------------------------------------
    # ADIM 3: Kaos ve Dayanıklılık Metrikleri
    # -------------------------------------------------------------
    print("\n[3/4] Küme Dayanıklılık ve Kurtarma Metrikleri:")
    print("-" * 110)
    print(f"{'Metrik Adı':<35} | {'Ölçülen Değer':<25} | {'Hedef / SLA Standardı'}")
    print("-" * 110)
    print(f"{'Toplam Gönderilen Kaos İsteği':<35} | {deney_raporu['toplam_istek']:>15} İstek   | {'100 İstek'}")
    print(f"{'Başarılı İstek Sayısı':<35} | {deney_raporu['basarili_istek']:>15} İstek   | {'100 İstek (%100 Koruma)'}")
    print(f"{'Kaos Altında SLA Erişilebilirlik':<35} | %{deney_raporu['sla_orani']:>14.1f}      | {'> %99.0 SLA'}")
    print(f"{'Ortalama Kurtarma Süresi (MTTR)':<35} | {deney_raporu['mttr_ms']:>14.1f} ms     | {'< 5000 ms SLA Eşiği'}")
    print(f"{'P99 İstek Gecikmesi':<35} | {deney_raporu['p99_gecikme_ms']:>14.1f} ms     | {'< 200 ms'} ")
    print(f"{'Ortalama İstek Gecikmesi':<35} | {deney_raporu['ortalama_gecikme_ms']:>14.1f} ms     | {'< 50 ms'} ")
    print(f"{'Otomatik İyileştirilen Düğüm':<35} | {deney_raporu['iyilesen_dugum_adedi']:>15} Düğüm   | {'Self-Healing Aktif'}")
    print("-" * 110)

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Görsel Teşhis Panosu Üretimi
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Kaos Mühendisliği Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "chaos_engineering_paneli.png")

    ChaosGorsellestirici.teshis_paneli_olustur(
        deney_raporu=deney_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Kaos Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 110)
    print("✓ Day 200: KAOS MÜHENDİSLİĞİ VE SİSTEM DAYANIKLILIK TESTİ BAŞARIYLA TAMAMLANDI!")
    print("=" * 110)


if __name__ == "__main__":
    main()
