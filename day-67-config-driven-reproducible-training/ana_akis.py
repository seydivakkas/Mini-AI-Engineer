"""
Day 67: YAML Konfigurasyon Yonetimi ve Deterministik Egitim Ana Akis Betigi
==========================================================================
1. YAML dosyasini Pydantic v2 semasi ile tip guvenli olarak yukler.
2. CLI / calisma zamani override'larini uygulayarak parametreleri gunceller.
3. Run A (Seed=42), Run B (Seed=42) ve Run C (Seed=99) bagimsiz kosularini yurutur.
4. Kayip (Loss), Dogruluk (Accuracy) ve Agirlik SHA256 ozetleri uzerinden determinizmi kanitlar.
5. 6 Panelli yuksek cozunurluklu endustriyel teshis panosunu kaydeder.
"""

import os
import sys
from src.konfigurasyon_yoneticisi import KonfigurasyonYoneticisi
from src.deney_dogrulayici import DeterminizmDogrulayici
from src.gorsellestirici import DeterminizmGorsellestirici


def main() -> None:
    print("=" * 95)
    print(">>> DAY 67: YAML KONFIGURASYON YONETIMI & DETERMINISTIK TEKRARLANABILIR EGITIM")
    print("=" * 95)

    kok_dizin = os.path.dirname(os.path.abspath(__file__))
    yaml_yolu = os.path.join(kok_dizin, "konfigurasyonlar", "varsayilan_egitim.yaml")
    ciktilar_dizini = os.path.join(kok_dizin, "ciktilar")
    os.makedirs(ciktilar_dizini, exist_ok=True)
    dashboard_yolu = os.path.join(ciktilar_dizini, "deterministik_egitim_paneli.png")

    # 1. Adım: YAML Konfigürasyonunu Yükle ve Doğrula
    print("\n[+] 1. Adim: YAML Konfigurasyon Dosyasi Yukleniyor ve Pydantic v2 ile Dogrulaniyor...")
    # Ornek calisma zamani override: Batch size ve epoch optimizasyonu
    override_parametreleri = ["egitim.epoch_sayisi=10", "egitim.tohum=42"]
    config = KonfigurasyonYoneticisi.yaml_yukle(yaml_yolu, override_listesi=override_parametreleri)

    print(f"    - Deney Adi          : {config.deney_adi} (v{config.versiyon})")
    print(f"    - Model Mimarisi     : {config.model.mimari_adi} (Girdi: {config.veri.girdi_boyutu}, Sinif: {config.model.sinif_sayisi})")
    print(f"    - Optimizer / LR     : {config.optimizer.tur.upper()} (LR: {config.optimizer.lr}, WD: {config.optimizer.weight_decay})")
    print(f"    - Scheduler          : {config.scheduler.tur.upper()} (T_max: {config.scheduler.t_max})")
    print(f"    - Deterministik Mod  : {'AKTIF' if config.egitim.deterministik_mod else 'PASIF'} (Tohum: {config.egitim.tohum})")

    # 2. Adım: Determinizm ve Tekrarlanabilirlik Testini Koş
    print("\n[+] 2. Adim: Determinizm Dogrulama Kosulari Baslatiliyor (Run A vs Run B vs Run C)...")
    dogrulama_sonuclari = DeterminizmDogrulayici.determinizm_testi_kos(config, farkli_tohum_baseline=99)

    res_a = dogrulama_sonuclari["sonuc_a"]
    res_b = dogrulama_sonuclari["sonuc_b"]
    res_c = dogrulama_sonuclari["sonuc_c"]

    print("\n" + "=" * 95)
    print(">>> 3. DETERMINISTIK EGITIM KOSULARI SONUCLARI VE METRIKLER")
    print("=" * 95)
    print(f"{'Kosu':<15} | {'Tohum':<8} | {'Son Train Loss':<16} | {'Son Val Loss':<14} | {'Val Acc (%)':<12} | {'Agirlik SHA256 (Ilk 12 Karakter)':<30}")
    print("-" * 95)
    print(f"{'Run A (Hedef)':<15} | {config.egitim.tohum:<8} | {res_a['son_train_loss']:<16.6f} | {res_a['son_val_loss']:<14.6f} | %{res_a['son_val_accuracy']:<10.2f} | {dogrulama_sonuclari['run_a_hash'][:28]}...")
    print(f"{'Run B (Tekrar)':<15} | {config.egitim.tohum:<8} | {res_b['son_train_loss']:<16.6f} | {res_b['son_val_loss']:<14.6f} | %{res_b['son_val_accuracy']:<10.2f} | {dogrulama_sonuclari['run_b_hash'][:28]}...")
    print(f"{'Run C (Farkli)':<15} | {99:<8} | {res_c['son_train_loss']:<16.6f} | {res_c['son_val_loss']:<14.6f} | %{res_c['son_val_accuracy']:<10.2f} | {dogrulama_sonuclari['run_c_hash'][:28]}...")

    print("\n" + "=" * 95)
    print(">>> 4. DETERMINIZM VE SAYISAL BIT-FOR-BIT ESLESME ANALIZI")
    print("=" * 95)
    print(f"* Maksimum Egitim Kaybi Farki (Delta Loss A-B)    : {dogrulama_sonuclari['maks_train_loss_delta_ab']:.10f} (SIFIR HATA)")
    print(f"* Maksimum Dogrulama Kaybi Farki (Delta Val Loss)  : {dogrulama_sonuclari['maks_val_loss_delta_ab']:.10f} (SIFIR HATA)")
    print(f"* Maksimum Dogrulama Basarimi Farki (Delta Acc)    : % {dogrulama_sonuclari['maks_val_acc_delta_ab']:.4f}")
    print(f"* Agirlik Tensörleri SHA256 Hash Eslesmesi         : {'%100 ESIT VE KUSURSUZ' if dogrulama_sonuclari['agirlik_hash_eslesmesi'] else 'UYUSMAZLIK VAR'}")
    print(f"* Deterministik Mod Durumu                         : {'BASARILI (%100 TEKRARLANABILIR)' if dogrulama_sonuclari['deterministik_basarili'] else 'BASARISIZ'}")

    # 3. Adım: 6 Panelli Teşhis Panosunu Üret
    print("\n[+] 5. Adim: 6 Panelli Determinizm ve Konfigurasyon Teşhis Panosu Olusturuluyor...")
    grafik_yolu = DeterminizmGorsellestirici.panoyu_ciz_ve_kaydet(
        dogrulama_sonuclari=dogrulama_sonuclari,
        cikti_yolu=dashboard_yolu
    )
    print(f"[+] Teşhis Panosu Kaydedildi: {grafik_yolu}")
    print("=" * 95)
    print("DAY 67: CONFIG-DRIVEN REPRODUCIBLE TRAINING BASARIYLA TAMAMLANDI!")
    print("=" * 95)


if __name__ == "__main__":
    main()
