"""
Day 43: Veri Kayması (Data Drift) Tespiti, KS-Testi ve Wasserstein Mesafesi Ana Yürütme Betiği.
"""

import os
import numpy as np
from src.dagilim_olcer import KSVeWassersteinHesaplayici
from src.kayma_tespitci import VeriKaymasiDedektoru
from src.gorsellestirici import VeriKaymasiGorsellestirici


def main():
    print("=" * 85)
    print(">>> DAY 43: VERİ KAYMASI (DATA DRIFT) TESPİTİ, KS-TESTİ VE WASSERSTEIN MESAFESİ")
    print("=" * 85)

    np.random.seed(42)
    n_ref = 1500
    n_prod = 500

    # 1. Model Eğitiminde Kullanılan Referans Dağılımlar
    referans_veriler = {
        "sicaklik_celsius": np.random.normal(45.0, 2.5, size=n_ref),
        "titresim_hz": np.random.gamma(shape=6.0, scale=2.0, size=n_ref),
        "basinc_bar": np.random.normal(3.2, 0.35, size=n_ref),
        "kamera_parlaklik": np.random.normal(128.0, 15.0, size=n_ref)
    }

    dedektor = VeriKaymasiDedektoru(referans_veriler, alpha=0.05, kritik_psi_esigi=0.20)
    print("[+] Model Eğitim Referans Dağılımları Kaydedildi (1500 Örneklem / 4 Öznitelik)")

    # -------------------------------------------------------------
    # Senaryo 1: Kararlı Canlı Üretim Trafiği (In-Distribution)
    # -------------------------------------------------------------
    print("\n" + "=" * 85)
    print(">>> SENARYO 1: KARARLI CANLI ÜRETİM VERİSİ (KAYMA YOK)")
    print("=" * 85)
    canli_stabil = {
        "sicaklik_celsius": np.random.normal(45.1, 2.4, size=n_prod),
        "titresim_hz": np.random.gamma(shape=6.1, scale=2.0, size=n_prod),
        "basinc_bar": np.random.normal(3.21, 0.36, size=n_prod),
        "kamera_parlaklik": np.random.normal(127.8, 14.8, size=n_prod)
    }

    rapor1 = dedektor.teftis_et(canli_stabil)
    print(f"    - Genel Durum    : {rapor1['genel_durum']}")
    print(f"    - MLOps Aksiyonu : {rapor1['mlops_aksiyonu']}")
    print(f"    - Kayan Öznitelik: {rapor1['kayan_oznitelik_sayisi']} / {rapor1['toplam_oznitelik_sayisi']}")

    # -------------------------------------------------------------
    # Senaryo 2: Kademeli Kovaryans Kayması (Hafif Sezonluk Değişim)
    # -------------------------------------------------------------
    print("\n" + "=" * 85)
    print(">>> SENARYO 2: KADEMELİ KOVARYANS KAYMASI (ORTA DÜZEY UYARI)")
    print("=" * 85)
    canli_orta_kayma = {
        "sicaklik_celsius": np.random.normal(46.2, 2.6, size=n_prod),  # 1.2 derece hafif artış
        "titresim_hz": np.random.gamma(shape=6.0, scale=2.0, size=n_prod),
        "basinc_bar": np.random.normal(3.38, 0.38, size=n_prod),     # Hafif basınç artışı
        "kamera_parlaklik": np.random.normal(128.0, 15.0, size=n_prod)
    }

    rapor2 = dedektor.teftis_et(canli_orta_kayma)
    print(f"    - Genel Durum    : {rapor2['genel_durum']}")
    print(f"    - MLOps Aksiyonu : {rapor2['mlops_aksiyonu']}")
    print(f"    - Kayma Oranı    : %{rapor2['kayma_orani']:.1f}")
    for oz, res in rapor2["oznitelikler"].items():
        print(f"      [*] {oz:<18}: KS D={res['ks_istatistigi']:.3f}, p={res['p_degeri']:.4f}, W1={res['wasserstein_mesafesi']:.3f}, PSI={res['psi_skoru']:.3f} -> {res['kayma_derecesi']}")

    # -------------------------------------------------------------
    # Senaryo 3: Şiddetli Sensör Bozulması & Kritik Veri Kayması
    # -------------------------------------------------------------
    print("\n" + "=" * 85)
    print(">>> SENARYO 3: ŞİDDETLİ SENSÖR KAYMASI & KRİTİK ALARM")
    print("=" * 85)
    canli_kritik_kayma = {
        "sicaklik_celsius": np.random.normal(54.5, 4.0, size=n_prod),  # +9.5 derece radikal sapma
        "titresim_hz": np.random.gamma(shape=12.0, scale=2.5, size=n_prod), # 2x titreşim
        "basinc_bar": np.random.normal(4.8, 0.8, size=n_prod),        # Ciddi aşırı basınç
        "kamera_parlaklik": np.random.normal(95.0, 25.0, size=n_prod) # Işık sönmesi
    }

    rapor3 = dedektor.teftis_et(canli_kritik_kayma)
    print(f"    - Genel Durum    : {rapor3['genel_durum']}")
    print(f"    - MLOps Aksiyonu : {rapor3['mlops_aksiyonu']}")
    print(f"    - Alarm Verildi  : {'EVET (KRITIK RETRAIN TETIKLENDI)' if rapor3['alarm_verildi'] else 'HAYIR'}")
    print(f"    - Kritik Kayma   : {rapor3['kritik_kayma_sayisi']} Öznitelik")

    # -------------------------------------------------------------
    # 5. 6 Panelli Teşhis Panosunun Üretilmesi
    # -------------------------------------------------------------
    print("\n" + "=" * 85)
    print(">>> 5. KONSOLİDE 6 PANELLİ VERİ KAYMASI TEŞHİS PANOSUNUN ÜRETİLMESİ")
    print("=" * 85)

    cikis_resmi = VeriKaymasiGorsellestirici.panel_ciz(
        odak_oznitelik="sicaklik_celsius",
        referans_dizi=referans_veriler["sicaklik_celsius"],
        uretim_dizi=canli_kritik_kayma["sicaklik_celsius"],
        genel_rapor=rapor3,
        hedef_path="day-43-numpy-data-drift-detector/ciktilar/veri_kaymasi_paneli.png"
    )
    print(f"[+] 6 Panelli Veri Kayması Panosu Kaydedildi: {os.path.abspath(cikis_resmi)}")
    print("=" * 85)
    print("DAY 43: VERİ KAYMASI TESPİT MOTORU BAŞARIYLA TAMAMLANDI!")
    print("=" * 85)


if __name__ == "__main__":
    main()
