"""Day 21 Ana Çalıştırma Akışı: PyTorch ile Derin Öğrenme Görsel Sınıflandırma.

Bu betik; 4 sınıflı sentetik görsel veri setini üretir, PyTorch Dataset & DataLoader
yapısını kurar, Conv2D + BatchNorm2D + Dropout içeren CNN mimarisini eğitir,
test kümesinde değerlendirir, teşhis raporunu ve Grad-CAM açıklanabilirlik görselini kaydeder.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Proje kök dizinini ekle
proje_kok = Path(__file__).resolve().parent
if str(proje_kok) not in sys.path:
    sys.path.insert(0, str(proje_kok))

import numpy as np
import torch

from src.model_mimari import PyTorchVisionCNN
from src.veri_hazirlayici import VeriYoneticisi, SentetikGorselDataset
from src.egitici import PyTorchEgitici, EgitimSonucu
from src.gorsellestirici import PyTorchGorsellestirici
from src.grad_cam import GradCAM


def baslik(metin: str) -> None:
    """Konsol çıktıları için formatlı bölüm başlığı basar."""
    cizgi = "=" * 80
    print(f"\n{cizgi}\n>>> {metin}\n{cizgi}")


def main() -> None:
    """Day 21 PyTorch CNN görsel sınıflandırma tam eğitim ve değerlendirme akışını koşturur."""
    baslik("AŞAMA 1: Görsel Veri Setinin Üretilmesi")
    yonetici = VeriYoneticisi(hedef_boyut=(64, 64), random_state=42)
    X, y, sinif_isimleri = yonetici.sentetik_veri_seti_uret(sinif_basina_ornek=50)

    print(f"[+] Toplam Üretilen Görsel Sayısı: {len(X)}")
    print(f"[+] Tensör Şekli: {X.shape} (N={X.shape[0]}, H={X.shape[1]}, W={X.shape[2]}, C={X.shape[3]})")
    print(f"[+] Piksel Değer Aralığı: [{X.min():.1f}, {X.max():.1f}]")
    print(f"[+] Sınıflar: {sinif_isimleri}")

    baslik("AŞAMA 2: Stratified Veri Bölümleme ve PyTorch DataLoader Kurulumu")
    train_loader, val_loader, test_loader, X_test, y_test = yonetici.veri_bol_ve_yukleyicileri_olustur(
        X, y, val_orani=0.15, test_orani=0.15, batch_size=16
    )

    print(f"[+] Eğitim Kümesi (Train DataLoader)       : {len(train_loader.dataset)} örnek ({len(train_loader)} batch)")
    print(f"[+] Doğrulama Kümesi (Validation DataLoader): {len(val_loader.dataset)} örnek ({len(val_loader)} batch)")
    print(f"[+] Test Kümesi (Test DataLoader)           : {len(test_loader.dataset)} örnek ({len(test_loader)} batch)")

    # İlk batch tensör şeklini kontrol et
    ornek_x, ornek_y = next(iter(train_loader))
    print(f"[+] PyTorch Girdi Tensör Şekli (B, C, H, W) : {list(ornek_x.shape)}")

    baslik("AŞAMA 3: PyTorch CNN Model Mimarisinin Kurulması ve Ağırlık İlklendirmesi")
    model = PyTorchVisionCNN(
        in_channels=3,
        num_classes=len(sinif_isimleri),
        dropout_rate=0.3,
        input_size=(64, 64),
    )

    param_bilgi = model.count_parameters()
    print(f"[+] Model Sınıfı: {model.__class__.__name__}")
    print(f"[+] Toplam Parametre Sayısı: {param_bilgi['total']:,}")
    print(f"[+] Eğitilebilir Parametre Sayısı: {param_bilgi['trainable']:,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Hesaplama Aygıtı (Device): {device}")

    baslik("AŞAMA 4: PyTorch Modeli Eğitimi (AdamW + CosineAnnealingLR + EarlyStopping)")
    egitici = PyTorchEgitici(model, device=device)

    t0 = time.perf_counter()
    tarihce, en_iyi_model = egitici.egit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=35,
        learning_rate=0.001,
        weight_decay=1e-4,
        patience=10,
        grad_clip=1.0,
    )
    egitim_suresi = time.perf_counter() - t0

    tamamlanan_epoch = len(tarihce["train_loss"])
    son_train_loss = tarihce["train_loss"][-1]
    son_val_loss = tarihce["val_loss"][-1]
    son_val_acc = tarihce["val_acc"][-1]

    print(f"[+] Eğitim Tamamlandı: {tamamlanan_epoch} Epoch ({egitim_suresi:.2f} saniye)")
    print(f"[+] Son Eğitim Kaybı (Train Loss): {son_train_loss:.4f}")
    print(f"[+] Son Doğrulama Kaybı (Val Loss): {son_val_loss:.4f}")
    print(f"[+] Son Doğrulama Doğruluğu (Val Acc): %{son_val_acc * 100:.2f}")

    baslik("AŞAMA 5: Test Kümesinde Kapsamlı Değerlendirme")
    sonuc: EgitimSonucu = egitici.degerlendir(test_loader, tarihce, egitim_suresi)

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
    rapor_dosyasi = cikti_klasor / "pytorch_cnn_raporu.png"

    PyTorchGorsellestirici.egitim_raporu_ciz(
        sonuc=sonuc,
        sinif_isimleri=sinif_isimleri,
        X_test=X_test,
        hedef_dosya=rapor_dosyasi,
    )
    print(f"[+] PyTorch CNN eğitim raporu başarıyla kaydedildi: {rapor_dosyasi}")

    baslik("AŞAMA 7: Mini Görev (Challenge) - PyTorch Grad-CAM ile Model Açıklanabilirliği (XAI)")
    # Hedef katman olarak blok3'ün Conv katmanını seç
    hedef_katman = model.blok3.conv
    grad_cam = GradCAM(model=model, hedef_katman=hedef_katman)

    # Test kümesinden bir örnek al ve tensöre çevir
    ornek_idx = 0
    ornek_rgb = X_test[ornek_idx]
    ornek_chw = np.transpose(ornek_rgb, (2, 0, 1))
    ornek_norm = (ornek_chw - 0.5) / 0.5
    girdi_tensor = torch.from_numpy(ornek_norm).float().unsqueeze(0).to(device)

    isi_haritasi, aciklanan_sinif = grad_cam.isi_haritasi_uret(girdi_tensor)
    fig_cam = grad_cam.bindirme_ciz(
        orijinal_rgb=ornek_rgb,
        isi_haritasi=isi_haritasi,
        sinif_adi=sinif_isimleri[aciklanan_sinif],
    )
    cam_dosyasi = cikti_klasor / "grad_cam_aciklanabilirlik.png"
    fig_cam.savefig(cam_dosyasi, bbox_inches="tight")
    print(f"[+] Grad-CAM ısı haritası ve görsel bindirme kaydedildi: {cam_dosyasi}")
    print(f"[+] Açıklanan Hedef Katman: model.blok3.conv (128 kanal, 8x8 özellik haritası)")
    print(f"[+] Açıklanan Sınıf: {sinif_isimleri[aciklanan_sinif]}")


if __name__ == "__main__":
    main()
