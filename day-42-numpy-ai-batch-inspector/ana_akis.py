"""
Day 42: Üretim Girdi Tensörleri Doğrulama ve Anomali Teftişi (AI Batch Inspector) Ana Yürütme Betiği.
"""

import os
import numpy as np
from src.sema import TensorSemasi
from src.denetleyici import AIBatchDenetleyici
from src.temizleyici import BatchTemizleyici
from src.gorsellestirici import TensorDenetimGorsellestirici


def main():
    print("=" * 85)
    print(">>> DAY 42: FAZ 3 AÇILIŞI - ÜRETİM GİRDİ TENSÖRLERİ DOĞRULAMA VE ANOMALİ DENETÇİSİ")
    print("=" * 85)

    # 1. Üretim Modeli Tensör Şemasının Tanımlanması
    sema = TensorSemasi(
        model_adi="ResNet50-Edge-Inference",
        beklenen_sekil=(-1, 3, 224, 224),
        kanal_sirasi="NCHW",
        gecerli_tipler=[np.float32, np.float16],
        deger_araligi=(-3.0, 3.0),
        max_batch=32,
        sureklilik_sarti=True,
        max_bellek_mb=64.0
    )

    denetleyici = AIBatchDenetleyici(sema)
    temizleyici = BatchTemizleyici(sema)

    print("[+] Üretim Modeli Tensör Şeması Devreye Alındı:")
    print("    - Beklenen Şekil       : (-1, 3, 224, 224) [Dinamik Batch, NCHW]")
    print("    - Geçerli Dtype'lar    : [float32, float16]")
    print("    - Güvenli Değer Aralığı: [-3.0, +3.0]")
    print("    - Maksimum Batch Sınırı: 32")

    # -------------------------------------------------------------
    # Senaryo 1: Kusursuz Üretim Batch'i (Golden Standard)
    # -------------------------------------------------------------
    print("\n" + "=" * 85)
    print(">>> SENARYO 1: KUSURSUZ ÜRETİM BATCH'İ (GOLDEN BATCH)")
    print("=" * 85)
    np.random.seed(42)
    kusursuz_batch = np.random.normal(0.0, 0.85, size=(8, 3, 224, 224)).astype(np.float32)
    kusursuz_batch = np.clip(kusursuz_batch, -2.8, 2.8)

    rapor1 = denetleyici.denetle(kusursuz_batch)
    print(f"    - Karar          : {rapor1['karar']}")
    print(f"    - Denetim Süresi : {rapor1['denetim_suresi_ms']:.3f} ms")
    print(f"    - Şekil          : {rapor1['sekil']} | Dtype: {rapor1['dtype']} | Bellek: {rapor1['bellek_mb']} MB")
    print(f"    - Güvenli Geçiş  : {'ONAYLANDI (PASS)' if rapor1['guvenli_gecis'] else 'RED'}")

    # -------------------------------------------------------------
    # Senaryo 2: Düzeltilebilir Kusurlu Batch (NHWC & Aralık Dışı & float64)
    # -------------------------------------------------------------
    print("\n" + "=" * 85)
    print(">>> SENARYO 2: DÜZELTİLEBİLİR UYARI BATCH'İ (NHWC, float64, Outlier Değerler)")
    print("=" * 85)
    kusurlu_batch = np.random.normal(0.0, 1.2, size=(4, 224, 224, 3)).astype(np.float64)
    kusurlu_batch[0, 50, 50, 0] = 5.8  # Aralık dışı outlier
    kusurlu_batch[1, 100, 100, 1] = -4.9

    rapor2 = denetleyici.denetle(kusurlu_batch)
    print(f"    - Karar          : {rapor2['karar']}")
    print(f"    - İhlal Sayısı   : {rapor2['toplam_ihlal_sayisi']} Adet")
    for ih in rapor2["ihlaller"]:
        print(f"      [*] {ih['kod']}: {ih['mesaj']}")

    temiz_batch, temiz_rapor = temizleyici.temizle_ve_uyarla(kusurlu_batch, rapor2)
    print("\n    [+] Otomatik Temizleme (Sanitization) Sonucu:")
    print(f"      - Yeni Şekil   : {temiz_rapor['yeni_sekil']} (NHWC -> NCHW)")
    print(f"      - Yeni Dtype   : {temiz_rapor['yeni_dtype']} (float64 -> float32)")
    print(f"      - Yapılan İşlem: {temiz_rapor['yapilan_islemler']}")

    # -------------------------------------------------------------
    # Senaryo 3: Kritik Hatalı Batch (NaN / Inf Taşması)
    # -------------------------------------------------------------
    print("\n" + "=" * 85)
    print(">>> SENARYO 3: KRİTİK HATALI BATCH (NaN ve Inf Sayısal Taşma)")
    print("=" * 85)
    nan_batch = np.random.normal(0.0, 1.0, size=(2, 3, 224, 224)).astype(np.float32)
    nan_batch[0, 0, 10, 10] = np.nan
    nan_batch[0, 1, 20, 20] = np.inf

    rapor3 = denetleyici.denetle(nan_batch)
    print(f"    - Karar          : {rapor3['karar']}")
    print(f"    - Güvenli Geçiş  : {'ONAYLANDI' if rapor3['guvenli_gecis'] else 'REDDEDİLDİ (CRITICAL REJECT)'}")
    for ih in rapor3["ihlaller"]:
        print(f"      [*] {ih['kod']}: {ih['mesaj']}")

    # -------------------------------------------------------------
    # Senaryo 4: Batch Boyutu Aşımı (OOM Engelleme)
    # -------------------------------------------------------------
    print("\n" + "=" * 85)
    print(">>> SENARYO 4: GPU BELLEK KORUMA (Aşırı Büyük Batch Boyutu B=64 > 32)")
    print("=" * 85)
    buyuk_batch = np.zeros((64, 3, 224, 224), dtype=np.float32)
    rapor4 = denetleyici.denetle(buyuk_batch)
    print(f"    - Karar          : {rapor4['karar']}")
    print(f"    - Hata Mesajı    : {rapor4['ihlaller'][0]['mesaj']}")

    # -------------------------------------------------------------
    # 5. 6 Panelli Teşhis Panosunun Çizilmesi
    # -------------------------------------------------------------
    print("\n" + "=" * 85)
    print(">>> 5. KONSOLİDE 6 PANELLİ ANOMALİ TEŞHİS PANOSUNUN ÜRETİLMESİ")
    print("=" * 85)
    cikis_resmi = TensorDenetimGorsellestirici.panel_ciz(
        orijinal_tensor=kusurlu_batch,
        denetim_raporu=rapor2,
        temizlenmis_rapor=temiz_rapor,
        hedef_path="day-42-numpy-ai-batch-inspector/ciktilar/tensor_denetim_paneli.png"
    )
    print(f"[+] 6 Panelli Teftiş Panosu Kaydedildi: {os.path.abspath(cikis_resmi)}")
    print("=" * 85)
    print("DAY 42: ÜRETİM GİRDİ TENSÖRLERİ DOĞRULAMA PAKETİ BAŞARIYLA TAMAMLANDI!")
    print("=" * 85)


if __name__ == "__main__":
    main()
