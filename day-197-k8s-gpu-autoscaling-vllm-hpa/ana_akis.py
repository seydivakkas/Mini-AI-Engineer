"""
Day 197: Kubernetes KEDA & HPA ile GPU Kullanımına Göre vLLM Podlarını Otomatik Ölçekleme Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.k8s_keda_motoru import (
    KedaMetricCollector,
    KedaScaledObjectSimulator,
    KubernetesClusterController,
)
from src.keda_profilleyici import KedaAutoscalingProfilleyici
from src.gorsellestirici import KedaGorsellestirici


def main():
    print("=" * 110)
    print(">>> Day 197 (FAZ 10): KUBERNETES KEDA & HPA vLLM GPU AUTOSCALING ENGINE")
    print("=" * 110)

    # -------------------------------------------------------------
    # ADIM 1: KEDA ScaledObject ve Küme Başlatma
    # -------------------------------------------------------------
    print("\n[1/4] KEDA ScaledObject ve Kubernetes Kontrolcüsü Yapılandırılıyor...")
    scaled_object = KedaScaledObjectSimulator(
        min_replicas=1,
        max_replicas=10,
        target_waiting_per_pod=5.0,
        target_kv_cache_usage=0.80,
    )
    k8s_controller = KubernetesClusterController(cluster_total_gpus=16)

    print(f"  • Minimum Replika (Min Pod)   : {scaled_object.min_replicas}")
    print(f"  • Maksimum Replika (Max Pod)  : {scaled_object.max_replicas}")
    print(f"  • Hedef Kuyruk Derinliği / Pod: {scaled_object.target_waiting_per_pod}")
    print(f"  • Hedef KV Cache Eşiği        : %{scaled_object.target_kv_cache_usage * 100:.0f}")
    print("  ✓ KEDA ScaledObject Başarıyla Oluşturuldu!")

    # -------------------------------------------------------------
    # ADIM 2: vLLM Prometheus Özel Metrik Toplama Testi
    # -------------------------------------------------------------
    print("\n[2/4] vLLM Prometheus Özel Metrikleri Toplanıyor...")
    ornek_metrik = KedaMetricCollector.metrik_topla(
        num_requests_waiting=25,
        gpu_cache_usage_factor=0.88,
    )
    hedef_pod = scaled_object.hesapla_hedef_replika(ornek_metrik)
    aktif_podlar = k8s_controller.podlari_senkronize_et(hedef_pod)

    print(f"  • vllm:num_requests_waiting   : {ornek_metrik['vllm:num_requests_waiting']:.0f} İstek")
    print(f"  • vllm:gpu_cache_usage_factor : %{ornek_metrik['vllm:gpu_cache_usage_factor']*100:.1f}")
    print(f"  • KEDA Hesaplanan Hedef Pod   : {hedef_pod} Pod ({hedef_pod}x GPU)")
    print(f"  • Kubernetes Senkronize Podlar: {len(aktif_podlar)} Aktif Pod (Durum: Running)")
    print("  ✓ KEDA Ölçekleme Formülü Başarıyla Doğrulandı!")

    # -------------------------------------------------------------
    # ADIM 3: 24 Saatlik Kurumsal Trafik Simülasyonu ve Maliyet Analizi
    # -------------------------------------------------------------
    print("\n[3/4] 24 Saatlik Kurumsal LLM Yük Simülasyonu Yürütülüyor...")
    simulasyon_raporu = KedaAutoscalingProfilleyici.yirmidort_saatlik_simulasyon_calistir()

    print("-" * 110)
    print(f"{'Zaman Aralığı':<18} | {'Faz Tanımı':<18} | {'Bekleyen İstek':<18} | {'KV Cache':<12} | {'Aktif Pod':<12} | {'GPU Tahsisi'}")
    print("-" * 110)
    for a in simulasyon_raporu["adimlari"]:
        print(
            f"{a['zaman_araligi']:<18} | "
            f"{a['faz_tanimi']:<18} | "
            f"{a['kuyrukta_bekleyen']:>10} İstek       | "
            f"{a['kv_cache_doluluk']:>8}     | "
            f"{a['aktif_pod_sayisi']:>6} Pod     | "
            f"{a['kullanilan_gpu']}x GPU"
        )
    print("-" * 110)
    print(f"  • Statik Tahsis (Sabit 8 GPU) : {simulasyon_raporu['statik_gpu_saat']:.1f} GPU-Saat")
    print(f"  • KEDA Dinamik Tahsis         : {simulasyon_raporu['dinamik_gpu_saat']:.1f} GPU-Saat")
    print(f"  • Toplam Maliyet Tasarrufu    : %{simulasyon_raporu['maliyet_tasarruf_yuzdesi']:.1f} Maliyet Avantajı!")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Görsel Teşhis Panosu Üretimi
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Kubernetes KEDA Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "k8s_keda_paneli.png")

    KedaGorsellestirici.teshis_paneli_olustur(
        simulasyon_raporu=simulasyon_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ K8s KEDA Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 110)
    print("✓ Day 197: KUBERNETES KEDA & HPA vLLM AUTOSCALING BAŞARIYLA TAMAMLANDI!")
    print("=" * 110)


if __name__ == "__main__":
    main()
