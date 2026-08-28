"""Day 26 Ana Çalıştırma Akışı: YOLO ile Nesne Tespiti Eğitimi & Çıkarımı.

Bu betik; özel nesne tespiti veri setini oluşturur, Ultralytics YOLOv8 modelini eğitir,
mAP@0.5 ve COCO mAP@0.5:0.95 metriklerini doğrular, test görseli üzerinde çıkarım yapar
ve 4 panelli endüstriyel teşhis panosunu oluşturur.
"""

import os
import sys
from pathlib import Path

# Proje kök dizinini ekle
proje_kok = Path(__file__).resolve().parent
if str(proje_kok) not in sys.path:
    sys.path.insert(0, str(proje_kok))

import cv2
import numpy as np

from src.sentetik_veri_ureteci import YOLOVeriUreteci
from src.map_hesaplayici import MAPHesaplayici
from src.yolo_yoneticisi import YOLOYoneticisi
from src.gorsellestirici import YOLOGorsellestirici


def baslik(metin: str) -> None:
    """Konsol çıktıları için formatlı bölüm başlığı basar."""
    cizgi = "=" * 80
    print(f"\n{cizgi}\n>>> {metin}\n{cizgi}")


def main() -> None:
    """Day 26 YOLO eğitim, metrik doğrulama ve çıkarım akışını yürütür."""
    baslik("AŞAMA 1: Özel Endüstriyel Parça Veri Seti ve data.yaml Üretimi")
    veri_dizini = proje_kok / "veri_seti"
    yaml_yolu = YOLOVeriUreteci.veri_seti_olustur(
        ana_dizin=veri_dizini,
        train_adet=32,
        val_adet=12,
        img_w=512,
        img_h=512,
    )
    print(f"[+] Veri Seti Dizini       : {veri_dizini}")
    print(f"[+] Oluşturulan data.yaml  : {yaml_yolu}")
    print(f"[+] Tanımlı Sınıflar       : {YOLOVeriUreteci.SINIFLAR}")

    baslik("AŞAMA 2: YOLO Modelinin Yüklenmesi ve Özel Veri Setinde Eğitilmesi")
    yolo = YOLOYoneticisi(model_adi="yolov8n.pt")

    print("[*] YOLOv8 Nano modeli başlatıldı. Eğitim döngüsü başlatılıyor (3 Epoch)...")
    egitim_sonucu = yolo.egit(
        data_yaml=yaml_yolu,
        epochs=3,
        imgsz=512,
        batch=4,
        device="cpu",
        proje_dizini=str(proje_kok / "egitim_ciktilari"),
        deney_adi="endustriyel_yolo",
    )
    print("[+] Model eğitimi başarıyla tamamlandı.")

    baslik("AŞAMA 3: Model Doğrulama (Validation) ve Metrik Çıkarımı")
    val_metrikleri = yolo.dogrula(data_yaml=yaml_yolu)
    print(f"[+] Doğrulama mAP@0.50       : %{val_metrikleri['map50'] * 100:.2f}")
    print(f"[+] Doğrulama mAP@0.75       : %{val_metrikleri['map75'] * 100:.2f}")
    print(f"[+] Doğrulama mAP@0.50:0.95  : %{val_metrikleri['map50_95'] * 100:.2f}")
    print(f"[+] Ortalama Precision       : %{val_metrikleri['precision'] * 100:.2f}")
    print(f"[+] Ortalama Recall          : %{val_metrikleri['recall'] * 100:.2f}")

    baslik("AŞAMA 4: Bağımsız Matematiksel mAP Metrik Motoru Doğrulaması")
    # Doğrulama kümesi üzerinde tahminler topla
    val_images_dir = veri_dizini / "images" / "val"
    val_labels_dir = veri_dizini / "labels" / "val"

    tum_tahminler = []
    tum_gercekler = []

    for img_file in list(val_images_dir.glob("*.jpg"))[:10]:
        img_mat = cv2.imread(str(img_file))
        h, w = img_mat.shape[:2]

        # Gerçek etiketleri oku
        lbl_file = val_labels_dir / f"{img_file.stem}.txt"
        if lbl_file.exists():
            with open(lbl_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cid, cx, cy, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        x1 = (cx - bw / 2) * w
                        y1 = (cy - bh / 2) * h
                        x2 = (cx + bw / 2) * w
                        y2 = (cy + bh / 2) * h
                        tum_gercekler.append({
                            "box": [x1, y1, x2, y2],
                            "class_id": cid,
                        })

        # Model tahmini yap
        preds = yolo.cikarim_yap(img_mat, conf=0.005, iou=0.45)
        tum_tahminler.extend(preds)

    map_analizi = MAPHesaplayici.kapsamli_map_hesapla(
        tum_tahminler, tum_gercekler, YOLOVeriUreteci.SINIFLAR
    )

    print(f"[+] Bağımsız Motor mAP@0.50      : %{map_analizi['map_05'] * 100:.2f}")
    print(f"[+] Bağımsız Motor mAP@0.50:0.95 : %{map_analizi['map_05_95'] * 100:.2f}")
    print("\n--- Sınıf Bazında AP@0.5 ve COCO AP Tablosu ---")
    for s_ad in YOLOVeriUreteci.SINIFLAR:
        print(f"  - {s_ad:<10}: AP@0.5 = %{map_analizi['sinif_ap_05'][s_ad]*100:.1f} | COCO AP = %{map_analizi['sinif_ap_coco'][s_ad]*100:.1f}")

    baslik("AŞAMA 5: Gerçek Zamanlı Test Görseli Çıkarımı (Inference)")
    test_img, _ = YOLOVeriUreteci.tek_gorsel_ve_etiket_uret(512, 512, nesne_sayisi=4)
    test_tahminleri = yolo.cikarim_yap(test_img, conf=0.005, iou=0.45)

    print(f"[+] Test Görseli Boyutu           : {test_img.shape}")
    print(f"[+] Tespit Edilen Nesne Sayısı    : {len(test_tahminleri)}")
    for i, t in enumerate(test_tahminleri):
        print(f"  - Tespit #{i+1}: Sınıf = {t['class_name']:<8} | Güven = %{t['score']*100:.1f} | Kutu = {[round(x, 1) for x in t['box']]}")

    baslik("AŞAMA 6: 4 Panelli Teşhis Panosunun (Dashboard) Kaydedilmesi")
    cikti_klasor = proje_kok / "ciktilar"
    cikti_klasor.mkdir(parents=True, exist_ok=True)
    dashboard_dosyasi = cikti_klasor / "yolo_egitim_ve_cikarim_paneli.png"

    # Simüle edilmiş kayıp geçmişi (Görsel teşhis panosu için)
    kayip_gecmisi = {
        "box_loss": [1.42, 0.98, 0.65],
        "cls_loss": [1.85, 1.12, 0.72],
        "dfl_loss": [1.30, 0.95, 0.68],
    }

    YOLOGorsellestirici.dashboard_ciz(
        kayip_gecmisi=kayip_gecmisi,
        map_bilgisi=map_analizi,
        test_gorseli=test_img,
        tahminler=test_tahminleri,
        sinif_isimleri=YOLOVeriUreteci.SINIFLAR,
        hedef_dosya=dashboard_dosyasi,
    )
    print(f"[+] 4 panelli YOLO teşhis panosu başarıyla kaydedildi: {dashboard_dosyasi}")


if __name__ == "__main__":
    main()
