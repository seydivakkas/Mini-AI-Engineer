"""
Day 91: Canlı AI Sistemlerinde Gözlemlenebilirlik Laboratuvarı
-------------------------------------------------------------
Gecikme (Latency P50/P95/P99), Trafik Hacmi (RPS), Veri Kayması (KS-Test & PSI)
ve Anomali İzleme Uçtan Uca Simülasyonu.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
import sys
import random
import time
import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.model import VisionModelObservability
from src.gozlemci_motoru import AIObservabilityMotoru
from src.gorsellestirici import ObservabilityGorsellestirici


def tohum_belirle(tohum: int = 42):
    random.seed(tohum)
    np.random.seed(tohum)
    torch.manual_seed(tohum)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tohum)


def main():
    print("=" * 85)
    print("🚀 Day 91: Canlı AI Sistemlerinde Gözlemlenebilirlik (AI Observability) Laboratuvarı")
    print("=" * 85)

    tohum_belirle(42)
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📌 Çalışma Ortamı Cihazı: {cihaz.upper()}")

    # 1. Model ve Gözlemlenebilirlik Motoru Hazırlığı
    model = VisionModelObservability(giris_kanali=3, sinif_sayisi=10, gizli_boyut=32)
    motor = AIObservabilityMotoru(
        model=model,
        cihaz=cihaz,
        sla_gecikme_esigi_ms=25.0,
        kayan_pencere_boyutu=400,
    )

    # -------------------------------------------------------------
    # ADIM 1: Baseline (Eğitim/Referans) Profilinin Oluşturulması
    # -------------------------------------------------------------
    print("\n[1/4] Referans (Baseline) Veri Dağılım Profili Oluşturuluyor...")
    referans_veriler = torch.randn(250, 3, 32, 32)
    motor.referans_egitimi_yapilandir(referans_veriler)
    print("  ✓ 250 Örnek Üzerinden Referans Öznitelik ve Güven Skoru Dağılımları Kaydedildi.")

    # -------------------------------------------------------------
    # ADIM 2: Canlı Normal Trafik Simülasyonu (Temiz Dağılım)
    # -------------------------------------------------------------
    print("\n[2/4] FAZ 1: Normal Üretim Trafiği Simüle Ediliyor (150 İstek)...")
    for idx in range(150):
        girdi = torch.randn(1, 3, 32, 32)
        _, _, gecikme = motor.tahmin_ve_gozlemle(girdi)
        time.sleep(0.002)  # Gerçekçi geliş aralığı

    ara_ozet = motor.metrik_toplayici.ozet_rapor_uret()
    print(f"  ✓ İşlenen İstek Sayısı: {ara_ozet.toplam_istek}")
    print(f"  ✓ Ortalama Gecikme: {ara_ozet.ortalama_gecikme_ms:.2f} ms | P95: {ara_ozet.p95_gecikme_ms:.2f} ms | P99: {ara_ozet.p99_gecikme_ms:.2f} ms")
    print(f"  ✓ SLA İhlal Oranı: %{ara_ozet.sla_ihlal_orani * 100:.2f}")

    # -------------------------------------------------------------
    # ADIM 3: Veri Kayması (Drift) ve Gecikme Bozulması Enjeksiyonu
    # -------------------------------------------------------------
    print("\n[3/4] FAZ 2: Veri Kayması (Domain Shift) & Ağ Gecikme Sıçraması Enjekte Ediliyor (150 İstek)...")
    for idx in range(150):
        # Dağılımı bozulmuş (parlaklık ve gürültü kayması) veri akışı
        bozulmus_girdi = torch.randn(1, 3, 32, 32) * 1.8 + 1.2
        _, _, gecikme = motor.tahmin_ve_gozlemle(bozulmus_girdi)

        # Yapay SLA gecikme sıçraması simülasyonu
        if idx % 10 == 0:
            time.sleep(0.035)  # 35 ms SLA gecikmesi

    # -------------------------------------------------------------
    # ADIM 4: Gözlemlenebilirlik Raporu & Teşhis Panosu
    # -------------------------------------------------------------
    print("\n[4/4] Sistem Sağlık ve Drift Analiz Raporu Çıkarılıyor...")
    son_ozet = motor.metrik_toplayici.ozet_rapor_uret()
    drift_raporu = motor.kayma_raporu_al()
    saglik = motor.sistem_saglik_durumu_al()

    print("=" * 85)
    print("📊 AI OBSERVABILITY METRİK ÖZETİ (PROMETHEUS UYUMLU)")
    print("=" * 85)
    print(f"• Toplam İşlenen İstek   : {son_ozet.toplam_istek}")
    print(f"• Anlık İşlem Hacmi (RPS) : {son_ozet.anlik_rps:.1f} req/s")
    print(f"• Ortalama Gecikme        : {son_ozet.ortalama_gecikme_ms:.2f} ms")
    print(f"• Gecikme Yüzdelikleri    : P50: {son_ozet.p50_gecikme_ms:.2f} ms | P95: {son_ozet.p95_gecikme_ms:.2f} ms | P99: {son_ozet.p99_gecikme_ms:.2f} ms")
    print(f"• Maksimum Gecikme        : {son_ozet.maks_gecikme_ms:.2f} ms")
    print(f"• SLA İhlal Oranı         : %{son_ozet.sla_ihlal_orani * 100:.2f} ({son_ozet.sla_ihlal_sayisi} İstek)")
    print(f"• Genel Sistem Durumu     : {saglik['durum']}")

    if drift_raporu:
        print("\n🔍 İSTATİSTİKSEL VERİ VE TAHMİN KAYMASI (DRIFT) ANALİZİ")
        print("─" * 85)
        print(f"• Toplam İzlenen Öznitelik : {drift_raporu.toplam_oznitelik}")
        print(f"• Kayan Öznitelik Sayısı   : {drift_raporu.kayan_oznitelik_sayisi} (%{drift_raporu.sistem_drift_orani * 100:.1f})")
        print(f"• Drift Alarm Seviyesi     : {drift_raporu.genel_durum}")
        print(f"• Tahmin (Output) Kayması  : {'EVET (Alarm)' if drift_raporu.tahmin_kaymasi_var_mi else 'HAYIR (Kararlı)'} (PSI: {drift_raporu.tahmin_psi:.4f})")

    if saglik["alarmlar"]:
        print("\n⚠️ AKTİF SİSTEM ALARMLARI:")
        for alarm in saglik["alarmlar"]:
            print(f"  [!] {alarm}")

    # Görselleştirme Paneli
    gorsellestirici = ObservabilityGorsellestirici()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "gozlemlenebilirlik_paneli.png")

    ref_np = np.array(motor.drift_dedektoru._referans_oznitelikler)
    canli_np = np.array(motor._canli_oznitelik_tamponu)

    gorsellestirici.olustur_gozlemlenebilirlik_paneli(
        zaman_serisi_verisi=motor.metrik_toplayici.zaman_serisi_verisi_al(),
        metrik_ozeti=son_ozet,
        drift_raporu=drift_raporu,
        referans_ozellik=ref_np,
        canli_ozellik=canli_np,
        kayit_yolu=cikti_yolu,
    )
    print(f"\n✓ 6-Panelli Teşhis Panosu Kaydedildi: {cikti_yolu}")
    print("\n✅ Day 91: Canlı AI Gözlemlenebilirlik Laboratuvarı Başarıyla Tamamlandı!")


if __name__ == "__main__":
    main()
