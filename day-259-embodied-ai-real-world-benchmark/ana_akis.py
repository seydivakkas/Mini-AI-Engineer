"""
Day 259: Robotik Başarım Paketi (Embodied AI Benchmarking Suite) Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from src.embodied_benchmark_motoru import (
    RoboticsMetricHarvester,
    FailureRootCauseAnalyzer,
    EmbodiedBenchmarkSuite,
)
from src.embodied_benchmark_profilleyici import EmbodiedBenchmarkProfilleyici
from src.gorsellestirici import EmbodiedBenchmarkGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 259 (FAZ 13): ROBOTİK BAŞARIM PAKETİ (GSR, PATH EFFICIENCY VE COLLISION RISK ANALİTİĞİ)")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Çok Boyutlu Metriklerin Hesaplanması
    # -------------------------------------------------------------
    print("\n[1/4] Çok Boyutlu Robotik Metrikler Hasat Ediliyor...")
    t = np.linspace(0, 1, 30)
    ideal_traj = np.vstack([t * 5.0, np.zeros_like(t), np.ones_like(t) * 0.5]).T
    noisy_traj = ideal_traj + np.random.randn(*ideal_traj.shape) * 0.04

    gsr = RoboticsMetricHarvester.compute_gsr(successes=493, total_trials=500)
    path_eff = RoboticsMetricHarvester.compute_path_efficiency(noisy_traj, ideal_traj[0], ideal_traj[-1])
    smoothness = RoboticsMetricHarvester.compute_curvature_smoothness(noisy_traj)
    dists = np.linspace(0.8, 1.5, 30)
    hazard = RoboticsMetricHarvester.compute_collision_risk(dists)

    print(f"  • Grasp Success Rate (GSR)      : %{gsr*100:.1f}")
    print(f"  • Rota Geodezik Verimliliği     : %{path_eff*100:.1f}")
    print(f"  • Yörünge Eğrilik Pürüzsüzlüğü  : {smoothness} (İvme Maliyeti)")
    print(f"  • Çarpışma Tehlike Endeksi      : {hazard} (Sıfıra Yakın)")

    # -------------------------------------------------------------
    # ADIM 2: Arıza Kök Neden Analizi
    # -------------------------------------------------------------
    print("\n[2/4] Telemetri Arıza Kök Neden Analizi Yapılıyor...")
    ornek_hata = FailureRootCauseAnalyzer.classify_failure(min_dist=0.01, is_singular=False, is_slip=False, is_timeout=False)
    print(f"  • Örnek Telemetri Arıza Teşhisi : {ornek_hata}")

    # -------------------------------------------------------------
    # ADIM 3: 500 Denemelik Standart Kıyaslama Raporu
    # -------------------------------------------------------------
    print("\n[3/4] 500 Standart Robotik Deneme ve %95 Wilson Güven Aralığı Hesaplanıyor...")
    bench_data = EmbodiedBenchmarkSuite.run_benchmark_trials(num_trials=500)
    ci = bench_data["wilson_guven_araligi_95"]
    print(f"  • Toplam Deneme Sayısı          : {bench_data['toplam_deneme_sayisi']}")
    print(f"  • Başarılı Görev Sayısı         : {bench_data['basarili_deneme']} (%{bench_data['global_basari_orani_yuzde']:.1f})")
    print(f"  • %95 Wilson Güven Aralığı      : [%{ci[0]*100:.2f} - %{ci[1]*100:.2f}]")
    print(f"  • Ortalama Çevrim Süresi        : {bench_data['ortalama_cevrim_suresi_s']} saniye")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Robotik Başarım Teşhis Panosu Oluşturuluyor...")
    profil_raporu = EmbodiedBenchmarkProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "embodied_benchmark_paneli.png")

    EmbodiedBenchmarkGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Robotik Başarım Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 259 (FAZ 13): ROBOTİK BAŞARIM VE KIYASLAMA MODÜLÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
