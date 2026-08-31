"""
Tesla Gün 23 Ana Akış (Tesla Day 23 Main Pipeline)
===================================================
Lityum İyon / LFP Batarya Hücre Kimyası ve 2-RC Eşdeğer Devre Modeli (ECM)
Uçtan Uca Çalıştırma ve Teşhis Paneli Üretim Scripti.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import sys
import os

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

su_an_dizin = os.path.dirname(os.path.abspath(__file__))
if su_an_dizin not in sys.path:
    sys.path.insert(0, su_an_dizin)

from src.tesla_batarya_ecm_modeli import (
    TeslaBatteryECM,
    BatteryCellParameters,
    BatteryChemistry
)
from src.tesla_ecm_profilleyici import TeslaECMProfilleyici
from src.tesla_ecm_gorsellestirici import TeslaECMGorsellestirici


def ana_calistirici():
    print("================================================================================")
    print("🚗 TESLA GÖMÜLÜ YAZILIM MASTERI | GÜN 23: LFP & NMC BATARYA ECM MODELİ 🚗")
    print("================================================================================")
    print("Stajyer Görevi: 2-RC Dual Polarization, OCV-SoC Platosu, Arrhenius R0 & Terminal V")
    print("--------------------------------------------------------------------------------\n")

    params_lfp = BatteryCellParameters(chemistry=BatteryChemistry.LFP, nominal_capacity_ah=60.0)
    params_nmc = BatteryCellParameters(chemistry=BatteryChemistry.NMC, nominal_capacity_ah=75.0)

    ecm_lfp = TeslaBatteryECM(params_lfp, initial_soc=0.80, initial_temp_c=25.0)
    ecm_nmc = TeslaBatteryECM(params_nmc, initial_soc=0.80, initial_temp_c=25.0)

    # 1. 150A Ağır İvmelenme Deşarj Testi
    print(" [1] 150A Tam Gaz Deşarj Altında LFP ve NMC Voltaj Çökmesi Hesaplanıyor...")
    out_lfp = ecm_lfp.step(current_a=150.0, dt_s=1.0)
    out_nmc = ecm_nmc.step(current_a=150.0, dt_s=1.0)

    print(f"     -> LFP Hücre OCV          : {out_lfp['ocv_v']:.3f} V | Terminal: {out_lfp['v_terminal']:.3f} V (ΔV = {(out_lfp['ocv_v'] - out_lfp['v_terminal'])*1000:.1f} mV)")
    print(f"     -> NMC Hücre OCV          : {out_nmc['ocv_v']:.3f} V | Terminal: {out_nmc['v_terminal']:.3f} V (ΔV = {(out_nmc['ocv_v'] - out_nmc['v_terminal'])*1000:.1f} mV)")
    print(f"     -> Joule Kayıp Gücü (Isı) : LFP = {out_lfp['p_loss_w']:.1f} W | NMC = {out_nmc['p_loss_w']:.1f} W")

    # 2. Fren Rejenerasyonu (-80A Şarj)
    print("\n [2] -80A Tek Pedallı Sürüş (One-Pedal Drive) Rejeneratif Şarj Simülasyonu...")
    regen_lfp = ecm_lfp.step(current_a=-80.0, dt_s=1.0)
    print(f"     -> LFP Rejenerasyon Voltajı: {regen_lfp['v_terminal']:.3f} V (OCV Üzerine Çıktı: +{(regen_lfp['v_terminal'] - regen_lfp['ocv_v'])*1000:.1f} mV)")

    # 3. Soğuk Hava İç Direnç Analizi (-10°C vs +25°C)
    print("\n [3] Düşük Sıcaklıkta Arrhenius İç Direnç Artış Denetimi...")
    ecm_cold = TeslaBatteryECM(params_nmc, initial_temp_c=-10.0)
    r0_cold = ecm_cold.get_temperature_adjusted_r0()
    r0_warm = params_nmc.r0_ohmic_ohm
    print(f"     -> 25°C Nominal İç Direnç : {r0_warm*1000:.2f} mΩ")
    print(f"     -> -10°C Soğuk İç Direnç  : {r0_cold*1000:.2f} mΩ ({r0_cold/r0_warm:.1f}x Daha Yüksek İç Direnç!)")

    # 4. Dinamik Sürüş Benchmark'ı
    print("\n [4] WLTP Dinamik Sürüş Profili ve 2-RC Simülasyon Hızı Benchmark'ı...")
    profilleyici = TeslaECMProfilleyici(sim_adimlari=1000)
    metrikler = profilleyici.benchmark_batarya_ecm()

    print(f"     -> Ortalama ECM Adım Süresi   : {metrikler['ecm_step_ortalama_us']:.3f} µs (P99: {metrikler['ecm_step_p99_us']:.3f} µs)")
    print(f"     -> Saniyelik ECM Çözüm Hacmi  : {metrikler['saniyelik_ecm_adimi']:,} Adım/sn")
    print(f"     -> 1000 Adım Sonrası LFP SoC  : %{metrikler['lfp_son_soc']:.2f}")
    print(f"     -> 1000 Adım Sonrası NMC SoC  : %{metrikler['nmc_son_soc']:.2f}")

    # 5. Tanı Paneli Görselleştirme
    print("\n [5] 6 Panelli Tesla Batarya ECM Tanı Paneli Üretiliyor...")
    gorsellestirici = TeslaECMGorsellestirici(cikti_dizini=os.path.join(su_an_dizin, "ciktilar"))
    gorsel_yolu = gorsellestirici.tani_paneli_ciz(metrikler, dosya_adi="tesla_batarya_ecm_tani_paneli.png")
    print(f"     -> Tanı Paneli Kaydedildi: {gorsel_yolu}")

    print("\n================================================================================")
    print(" 🚀 GÜN 23 BAŞARIYLA TAMAMLANDI! BATARYA ECM & HÜCRE FİZİĞİ DOĞRULANDI! 🚀")
    print("================================================================================")


if __name__ == "__main__":
    ana_calistirici()
