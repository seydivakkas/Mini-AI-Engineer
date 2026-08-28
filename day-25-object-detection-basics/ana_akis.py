"""Day 25 Ana Çalıştırma Akışı: Nesne Tespiti Temelleri & Bounding Box Regresyonu.

Bu betik; Bounding Box koordinat formatı dönüşümleri, IoU / GIoU / DIoU çakışma analizleri,
Klasik NMS vs Soft-NMS filtrelemeleri, Çok Ölçekli Anchor Box üretimi ve
regresyon hedeflerinin çözümlenmesini uçtan uca simüle eder ve görselleştirir.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

# Proje kök dizinini ekle
proje_kok = Path(__file__).resolve().parent
if str(proje_kok) not in sys.path:
    sys.path.insert(0, str(proje_kok))

import numpy as np

from src.kutu_donusturucu import KutuDonusturucu
from src.iou_hesaplayici import IoUHesaplayici
from src.nms_filtresi import NMSFiltresi
from src.anchor_ureteci import AnchorUreteci
from src.gorsellestirici import TespitGorsellestirici


def baslik(metin: str) -> None:
    """Konsol çıktıları için formatlı bölüm başlığı basar."""
    cizgi = "=" * 80
    print(f"\n{cizgi}\n>>> {metin}\n{cizgi}")


def simule_nesne_ve_adaylar_uret() -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Gerçekçi bir trafik sahnesi simülasyonu üretir (GT kutuları ve çakışan ham tahminler)."""
    np.random.seed(42)
    sinif_isimleri = ["Araba", "Yaya", "Bisiklet"]

    # 3 adet Ground Truth kutusu: [x1, y1, x2, y2]
    gt_boxes = np.array([
        [60.0, 180.0, 220.0, 320.0],   # Araba 1
        [280.0, 120.0, 360.0, 380.0],  # Yaya
        [390.0, 220.0, 480.0, 390.0],  # Bisiklet
    ])
    gt_labels = ["Araba", "Yaya", "Bisiklet"]

    raw_boxes = []
    raw_scores = []
    raw_labels = []

    # Her GT nesnesi etrafında çok sayıda çakışan tespit adayı simüle et
    for i, (gt, lbl) in enumerate(zip(gt_boxes, gt_labels)):
        w = gt[2] - gt[0]
        h = gt[3] - gt[1]

        # Ana yüksek skorlu tahmin
        raw_boxes.append(gt + np.random.uniform(-4, 4, size=4))
        raw_scores.append(np.random.uniform(0.88, 0.96))
        raw_labels.append(lbl)

        # 6-8 adet çakışan yan tahmin (mükerrer proposals)
        for _ in range(7):
            delta = np.random.uniform(-20, 20, size=4)
            perturbed_box = gt + delta
            score = np.random.uniform(0.35, 0.85)
            raw_boxes.append(perturbed_box)
            raw_scores.append(score)
            raw_labels.append(lbl)

    # 4 adet rastgele düşük skorlu arka plan yanlış tespiti (False Positive)
    for _ in range(4):
        rx1 = np.random.uniform(50, 400)
        ry1 = np.random.uniform(50, 400)
        raw_boxes.append([rx1, ry1, rx1 + 80, ry1 + 90])
        raw_scores.append(np.random.uniform(0.15, 0.38))
        raw_labels.append(np.random.choice(sinif_isimleri))

    raw_boxes_arr = np.array(raw_boxes)
    raw_scores_arr = np.array(raw_scores)

    # Etiket indeksleri
    sinif_haritasi = {ad: idx for idx, ad in enumerate(sinif_isimleri)}
    raw_label_indices = np.array([sinif_haritasi[lbl] for lbl in raw_labels])

    return gt_boxes, gt_labels, raw_boxes_arr, raw_scores_arr, raw_label_indices, raw_labels


def main() -> None:
    """Day 25 nesne tespiti, NMS ve anchor box akışını yürütür."""
    baslik("AŞAMA 1: Sahne Simülasyonu ve Bounding Box Format Dönüşümleri")
    gt_boxes, gt_labels, raw_boxes, raw_scores, raw_label_indices, raw_labels = simule_nesne_ve_adaylar_uret()

    print(f"[+] Ground Truth Nesne Sayısı : {len(gt_boxes)}")
    print(f"[+] Ham Dedektör Aday Sayısı   : {len(raw_boxes)}")

    # Format Dönüşüm Doğrulaması
    coco_xywh = KutuDonusturucu.xyxy_to_xywh(gt_boxes)
    yolo_cxcywh = KutuDonusturucu.xyxy_to_cxcywh(gt_boxes)
    geri_donusen_xyxy = KutuDonusturucu.cxcywh_to_xyxy(yolo_cxcywh)

    print("\n--- Örnek Pascal VOC (xyxy) vs COCO (xywh) vs YOLO (cxcywh) ---")
    for i, (xyxy, xywh, cxcywh) in enumerate(zip(gt_boxes, coco_xywh, yolo_cxcywh)):
        print(f"  - Nesne #{i+1} ({gt_labels[i]}):")
        print(f"      VOC  (xyxy)   : {[round(x, 1) for x in xyxy]}")
        print(f"      COCO (xywh)   : {[round(x, 1) for x in xywh]}")
        print(f"      YOLO (cxcywh) : {[round(x, 1) for x in cxcywh]}")

    baslik("AŞAMA 2: Çakışma Metrikleri Analizi: IoU, GIoU ve DIoU")
    # İlk 5 ham aday ile 3 GT arasındaki metrikleri hesapla
    ornek_adaylar = raw_boxes[:5]
    iou_mat = IoUHesaplayici.iou_matrisi(ornek_adaylar, gt_boxes)
    giou_mat = IoUHesaplayici.giou_matrisi(ornek_adaylar, gt_boxes)
    diou_mat = IoUHesaplayici.diou_matrisi(ornek_adaylar, gt_boxes)

    print("[+] Aday Kutu #1 ile GT #1 Arasındaki Çakışma:")
    print(f"      Standart IoU : {iou_mat[0, 0]:.4f}")
    print(f"      GIoU         : {giou_mat[0, 0]:.4f}")
    print(f"      DIoU         : {diou_mat[0, 0]:.4f}")

    baslik("AŞAMA 3: Non-Maximum Suppression (NMS) ve Soft-NMS Karşılaştırması")
    secilen_nms_idx = NMSFiltresi.sinifa_duyarli_nms(
        raw_boxes, raw_scores, raw_label_indices, iou_esigi=0.45, skor_esigi=0.40
    )

    nms_boxes = raw_boxes[secilen_nms_idx]
    nms_scores = raw_scores[secilen_nms_idx]
    nms_labels = [raw_labels[idx] for idx in secilen_nms_idx]

    # Soft NMS
    soft_boxes, soft_scores, soft_idx = NMSFiltresi.soft_nms(
        raw_boxes, raw_scores, iou_esigi=0.45, sigma=0.5, skor_esigi=0.40, method="gaussian"
    )

    print(f"[+] NMS Öncesi Toplam Kutu Sayısı : {len(raw_boxes)}")
    print(f"[+] Klasik NMS Sonrası Kalan Kutu : {len(nms_boxes)} (Elenen: {len(raw_boxes) - len(nms_boxes)})")
    print(f"[+] Soft-NMS Sonrası Kalan Kutu    : {len(soft_boxes)} (Skor sönümleme uygulandı)")

    print("\n--- Klasik NMS Tarafından Doğrulanan Nihai Tespitler ---")
    for i, (b, s, lbl) in enumerate(zip(nms_boxes, nms_scores, nms_labels)):
        print(f"  - Tespit #{i+1}: Sınıf = {lbl:<10} | Güven Skoru = %{s*100:.1f} | Kutu = {[round(x, 1) for x in b]}")

    baslik("AŞAMA 4: Çok Ölçekli Anchor Box Üretimi ve Bounding Box Regresyonu")
    # 8x8 feature map grid, stride=64, 3 ölçek, 3 en-boy oranı = 8 * 8 * 9 = 576 anchors
    anchors = AnchorUreteci.anchor_grid_uret(
        grid_w=8, grid_h=8, stride=64,
        olcekler=(48.0, 96.0, 192.0),
        en_boy_oranlari=(0.5, 1.0, 2.0)
    )
    print(f"[+] Üretilen Toplam Anchor Kutusu Sayısı : {len(anchors)} (8x8 Izgara x 9 Anchor/Hücre)")

    # GT Eşleme (Matching)
    etiketler, eslesen_gt, max_ious = AnchorUreteci.ground_truth_esle(
        anchors, gt_boxes, pos_iou_esigi=0.5, neg_iou_esigi=0.3
    )

    pozitif_sayisi = int(np.sum(etiketler == 1))
    negatif_sayisi = int(np.sum(etiketler == 0))
    notr_sayisi = int(np.sum(etiketler == -1))

    print(f"[+] Pozitif Eşleşen Anchor Sayısı (Ön Plan) : {pozitif_sayisi}")
    print(f"[+] Negatif Eşleşen Anchor Sayısı (Arka Plan): {negatif_sayisi}")
    print(f"[+] Nötr / Göz Ardı Edilen Anchor Sayısı     : {notr_sayisi}")

    # Pozitif Anchorlar için Delta Regresyon Hedeflerini Hesapla
    poz_mask = etiketler == 1
    poz_anchors = anchors[poz_mask]
    eslesen_gt_boxes = gt_boxes[eslesen_gt[poz_mask]]

    deltalar = AnchorUreteci.kutu_regresyon_hedefleri(poz_anchors, eslesen_gt_boxes)
    cozulmus_kutular = AnchorUreteci.regresyondan_kutulari_coz(poz_anchors, deltalar)

    # Regresyon çözümü hata payı (Kayıpsız doğrulanmalı)
    regresyon_hatasi = np.max(np.abs(cozulmus_kutular - eslesen_gt_boxes))
    print(f"[+] Delta Regresyon Çözümleme Maksimum Sapması : {regresyon_hatasi:.2e} piksel (Mükemmel)")

    baslik("AŞAMA 5: 4 Panelli Teşhis Panosunun (Dashboard) Kaydedilmesi")
    cikti_klasor = proje_kok / "ciktilar"
    cikti_klasor.mkdir(parents=True, exist_ok=True)
    dashboard_dosyasi = cikti_klasor / "nesne_tespiti_paneli.png"

    # Örnek anchor alt kümesi (Görsel temizlik için merkez civarı 8 anchor)
    merkez_mask = (anchors[:, 0] > 150) & (anchors[:, 0] < 350) & (anchors[:, 1] > 150) & (anchors[:, 1] < 350)
    anchors_sample = anchors[merkez_mask][:8]

    TespitGorsellestirici.dashboard_ciz(
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        raw_boxes=raw_boxes,
        raw_scores=raw_scores,
        raw_labels=raw_labels,
        nms_boxes=nms_boxes,
        nms_scores=nms_scores,
        nms_labels=nms_labels,
        anchors_sample=anchors_sample,
        iou_mat=iou_mat,
        giou_mat=giou_mat,
        img_size=(512, 512),
        hedef_dosya=dashboard_dosyasi,
    )
    print(f"[+] 4 panelli nesne tespiti panosu kaydedildi: {dashboard_dosyasi}")


if __name__ == "__main__":
    main()
