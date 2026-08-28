"""Day 27 Ana Çalıştırma Akışı: U-Net ile Anlamsal Bölütleme (Semantic Segmentation).

Bu betik; sentetik mikroskobik hücre dokusu veri setini yükler, U-Net mimarisini
Combo Loss (CE + Dice) ile eğitir, piksel doğruluğu, mIoU ve Dice metriklerini hesaplar
ve 6 panelli endüstri standardı teşhis panosunu oluşturur.
"""

import os
import sys
from pathlib import Path

# Proje kök dizinini ekle
proje_kok = Path(__file__).resolve().parent
if str(proje_kok) not in sys.path:
    sys.path.insert(0, str(proje_kok))

import numpy as np
import torch

from src.unet_modeli import UNet
from src.kayip_ve_metrikler import BolutlemeMetrikleri
from src.sentetik_veri_yoneticisi import SentetikBolutlemeDataset, VeriYoneticisi
from src.egitici import BolutlemeEgitici
from src.gorsellestirici import BolutlemeGorsellestirici


def baslik(metin: str) -> None:
    """Konsol çıktıları için formatlı bölüm başlığı basar."""
    cizgi = "=" * 80
    print(f"\n{cizgi}\n>>> {metin}\n{cizgi}")


def main() -> None:
    """Day 27 U-Net eğitim, değerlendirme ve bölütleme görselleştirme akışını yürütür."""
    baslik("AŞAMA 1: Sentetik Mikroskobik Hücre Veri Setinin Hazırlanması")
    train_loader, val_loader, train_ds, val_ds = VeriYoneticisi.dataloader_olustur(
        train_adet=48,
        val_adet=16,
        img_size=128,
        batch_size=8,
    )
    siniflar = SentetikBolutlemeDataset.SINIFLAR

    print(f"[+] Eğitim Örnek Sayısı     : {len(train_ds)}")
    print(f"[+] Doğrulama Örnek Sayısı  : {len(val_ds)}")
    print(f"[+] Görsel Çözünürlüğü      : 128x128x3")
    print(f"[+] Bölütleme Sınıfları     : {siniflar}")

    baslik("AŞAMA 2: U-Net Mimarisinin Oluşturulması")
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(in_channels=3, num_classes=3, kanal_tabani=32, bilinear=False)

    toplam_parametre = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[+] U-Net Modeli Başlatıldı (Cihaz: {cihaz.upper()})")
    print(f"[+] Toplam Eğitilebilir Parametre Sayısı: {toplam_parametre:,}")

    baslik("AŞAMA 3: U-Net Eğitim Döngüsü (Combo Loss = 0.5*CE + 0.5*Dice)")
    egitici = BolutlemeEgitici(model, device=cihaz, lr=1e-3, alpha=0.5)
    tarihce = egitici.tam_egitim(train_loader, val_loader, siniflar, epochs=5)

    baslik("AŞAMA 4: Doğrulama Kümesinde Kapsamlı Bölütleme Metrikleri")
    val_loss, rapor = egitici.dogrula(val_loader, siniflar)

    print(f"[+] Ortalama Piksel Doğruluğu (Pixel Accuracy) : %{rapor['pixel_accuracy'] * 100:.2f}")
    print(f"[+] Ortalama IoU (Mean IoU / Jaccard İndeksi) : %{rapor['miou'] * 100:.2f}")
    print(f"[+] Ortalama Dice Katsayısı (Mean Dice / F1)  : %{rapor['mean_dice'] * 100:.2f}")

    print("\n--- Sınıf Bazında Performans Tablosu ---")
    for s_ad, d in rapor["sinif_raporu"].items():
        print(f"  - {s_ad:<16}: IoU (Jaccard) = %{d['iou']*100:<5.1f} | Dice (F1) = %{d['dice']*100:<5.1f}")

    baslik("AŞAMA 5: Örnek Test Görseli Çıkarımı ve Piksel Hata Haritası")
    # Doğrulama kümesinden bir örnek al
    test_img_tensor, test_mask_tensor = val_ds[0]

    model.eval()
    with torch.no_grad():
        input_batch = test_img_tensor.unsqueeze(0).to(cihaz)
        logits = model(input_batch)
        pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

    gt_mask = test_mask_tensor.numpy()
    orig_img_rgb = (test_img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    hata_orani = np.mean(gt_mask != pred_mask) * 100.0
    print(f"[+] Seçilen Test Görselinde Piksel Uyuşmazlığı (Hata): %{hata_orani:.2f}")

    baslik("AŞAMA 6: 6 Panelli Teşhis Panosunun (Dashboard) Kaydedilmesi")
    cikti_klasor = proje_kok / "ciktilar"
    cikti_klasor.mkdir(parents=True, exist_ok=True)
    dashboard_dosyasi = cikti_klasor / "bolutleme_teshis_paneli.png"

    BolutlemeGorsellestirici.dashboard_ciz(
        orijinal_gorsel=orig_img_rgb,
        gt_maske=gt_mask,
        pred_maske=pred_mask,
        egitim_tarihcesi=tarihce,
        sinif_raporu=rapor["sinif_raporu"],
        sinif_isimleri=siniflar,
        hedef_dosya=dashboard_dosyasi,
    )
    print(f"[+] 6 panelli bölütleme teşhis panosu başarıyla kaydedildi: {dashboard_dosyasi}")


if __name__ == "__main__":
    main()
