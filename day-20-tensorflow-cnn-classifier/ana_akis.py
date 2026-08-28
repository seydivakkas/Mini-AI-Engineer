"""Day 20 Ana Çalıştırma Akışı: TensorFlow/Keras ile Derin Öğrenme Görsel Sınıflandırma.

Bu betik; 4 sınıflı sentetik görsel veri seti üretir,
Conv2D + MaxPooling + BatchNormalization + Dense + Dropout katmanlarından oluşan
CNN modelini derler, EarlyStopping ile eğitir, test kümesinde değerlendirir ve
öğrenme eğrileri ile test tahminlerini içeren 4 panelli rapor grafiğini kaydeder.
"""

import os
# Keras backend ayarı
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "torch"

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Proje kök dizinini ekle
proje_kok = Path(__file__).resolve().parent
if str(proje_kok) not in sys.path:
    sys.path.insert(0, str(proje_kok))

import numpy as np
from src.model_mimari import build_cnn_model
from src.veri_hazirlayici import VeriHazirlayici
from src.egitici import ModelEgitici, EgitimSonucu
from src.gorsellestirici import CNNGorsellestirici


def baslik(metin: str) -> None:
    """Konsol çıktıları için formatlı bölüm başlığı basar."""
    cizgi = "=" * 80
    print(f"\n{cizgi}\n>>> {metin}\n{cizgi}")


def main() -> None:
    """Day 20 CNN görsel sınıflandırma tam eğitim ve değerlendirme akışını koşturur."""
    baslik("AŞAMA 1: Görsel Veri Setinin Üretilmesi ve Hazırlanması")
    hazirlayici = VeriHazirlayici(hedef_boyut=(64, 64), random_state=42)
    X, y, sinif_isimleri = hazirlayici.sentetik_veri_seti_uret(sinif_basina_ornek=50)

    print(f"[+] Toplam Üretilen Görsel Sayısı: {len(X)}")
    print(f"[+] Tensör Şekli (Shape): {X.shape} (N={X.shape[0]}, H={X.shape[1]}, W={X.shape[2]}, C={X.shape[3]})")
    print(f"[+] Piksel Değer Aralığı: [{X.min():.1f}, {X.max():.1f}]")
    print(f"[+] Sınıflar: {sinif_isimleri}")

    baslik("AŞAMA 2: Stratified Veri Bölümleme (Train / Validation / Test)")
    X_train, y_train, X_val, y_val, X_test, y_test = hazirlayici.veri_bol(
        X, y, val_orani=0.15, test_orani=0.15
    )
    print(f"[+] Eğitim Kümesi (Train)       : {X_train.shape[0]} örnek")
    print(f"[+] Doğrulama Kümesi (Validation): {X_val.shape[0]} örnek")
    print(f"[+] Test Kümesi (Test / Holdout) : {X_test.shape[0]} örnek")

    baslik("AŞAMA 3: Keras CNN Model Mimarisinin Kurulması ve Derlenmesi")
    model = build_cnn_model(
        input_shape=(64, 64, 3),
        num_classes=len(sinif_isimleri),
        learning_rate=0.001,
        dropout_rate=0.3,
    )

    toplam_param = model.count_params()
    print(f"[+] Model Adı: {model.name}")
    print(f"[+] Toplam Parametre Sayısı: {toplam_param:,}")
    print("\n--- Katman Özeti (Layer Summary) ---")
    for layer in model.layers:
        print(f"  - {layer.name:<22}: Çıktı Şekli = {str(layer.output.shape):<20} | Param = {layer.count_params():,}")

    baslik("AŞAMA 4: Modelin Eğitilmesi (EarlyStopping & ReduceLROnPlateau)")
    egitici = ModelEgitici(model)

    t0 = time.perf_counter()
    tarihce = egitici.egit(
        X_train, y_train, X_val, y_val,
        epochs=35,
        batch_size=16,
        patience=10,
    )
    egitim_suresi = time.perf_counter() - t0

    tamamlanan_epoch = len(tarihce.history["loss"])
    son_train_loss = tarihce.history["loss"][-1]
    son_val_loss = tarihce.history["val_loss"][-1]
    son_val_acc = tarihce.history["val_accuracy"][-1]

    print(f"[+] Eğitim Tamamlandı: {tamamlanan_epoch} Epoch ({egitim_suresi:.2f} saniye)")
    print(f"[+] Son Eğitim Kaybı (Train Loss): {son_train_loss:.4f}")
    print(f"[+] Son Doğrulama Kaybı (Val Loss): {son_val_loss:.4f}")
    print(f"[+] Son Doğrulama Doğruluğu (Val Acc): %{son_val_acc * 100:.2f}")

    baslik("AŞAMA 5: Test Kümesinde Kapsamlı Değerlendirme")
    sonuc: EgitimSonucu = egitici.degerlendir(X_test, y_test, tarihce, egitim_suresi)

    print(f"[+] {sonuc.ozet()}")
    print("\n--- Sınıf Bazında Test Performansı ---")
    for idx, sinif_adi in enumerate(sinif_isimleri):
        sinif_mask = sonuc.y_test_gercek == idx
        sinif_toplam = int(np.sum(sinif_mask))
        sinif_dogru = int(np.sum(sonuc.y_test_tahmin[sinif_mask] == idx))
        sinif_acc = (sinif_dogru / sinif_toplam) * 100.0 if sinif_toplam > 0 else 0.0
        print(f"  - {sinif_adi:<12}: Doğruluk = %{sinif_acc:<6.1f} ({sinif_dogru}/{sinif_toplam})")

    baslik("AŞAMA 6: Görselleştirme ve Analiz Raporunun Kaydedilmesi")
    cikti_klasor = proje_kok / "ciktilar"
    cikti_klasor.mkdir(parents=True, exist_ok=True)
    rapor_dosyasi = cikti_klasor / "cnn_egitim_raporu.png"

    CNNGorsellestirici.egitim_raporu_ciz(
        sonuc=sonuc,
        sinif_isimleri=sinif_isimleri,
        X_test=X_test,
        hedef_dosya=rapor_dosyasi,
    )
    print(f"[+] CNN eğitim ve teşhis raporu başarıyla kaydedildi: {rapor_dosyasi}")

    baslik("AŞAMA 7: Mini Görev (Challenge) - Evrişim Ara Katman Aktivasyonlarının Çıkarılması")
    from src.aktivasyon_cikarici import AraKatmanAktivasyonCikarici
    cikarici = AraKatmanAktivasyonCikarici(model)
    ornek_gorsel = X_test[0]
    ornek_etiket = sinif_isimleri[y_test[0]]

    aktivasyon = cikarici.aktivasyon_haritasi_cikar(ornek_gorsel, katman_adi="conv2d_blok1")
    fig_akt = cikarici.aktivasyon_grid_ciz(
        aktivasyon,
        maks_filtre=16,
        baslik=f"Katman: conv2d_blok1 Aktivasyon Haritaları (Örnek Sınıf: {ornek_etiket})"
    )
    aktivasyon_dosyasi = cikti_klasor / "cnn_aktivasyon_haritalari.png"
    fig_akt.savefig(aktivasyon_dosyasi, bbox_inches="tight")
    print(f"[+] Ara katman aktivasyon haritaları kaydedildi: {aktivasyon_dosyasi}")
    print(f"[+] İncelenen Katman: conv2d_blok1 (32 filtre, {aktivasyon.shape[1]}x{aktivasyon.shape[2]})")


if __name__ == "__main__":
    main()
