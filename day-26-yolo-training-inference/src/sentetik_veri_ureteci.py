"""YOLO Formatında Sentetik Veri Seti ve YAML Yapılandırma Üretim Modülü.

Bu modül; nesne tespiti eğitimi için endüstriyel parça nesneleri ("Vida", "Somun", "Rulman") içeren
sentetik görseller üretir, YOLO normalleştirilmiş etiketlerini (labels/*.txt) yazar ve
Ultralytics YOLO eğitimi için standart 'data.yaml' dosyasını oluşturur.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
import yaml


class YOLOVeriUreteci:
    """YOLOv8/YOLO11 uyumlu sentetik veri seti ve etiket üreticisi."""

    SINIFLAR = ["Vida", "Somun", "Rulman"]

    @classmethod
    def tek_gorsel_ve_etiket_uret(
        cls,
        img_w: int = 512,
        img_h: int = 512,
        nesne_sayisi: int = 4,
    ) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
        """Tek bir sentetik görsel ve normalize YOLO [class, cx, cy, w, h] etiketlerini üretir."""
        # Arka plan dokusu (Hafif gürültülü endüstriyel metal tezgah)
        gorsel = np.random.randint(220, 245, (img_h, img_w, 3), dtype=np.uint8)

        # Izgara desen çizgileri
        for x in range(0, img_w, 64):
            cv2.line(gorsel, (x, 0), (x, img_h), (210, 210, 210), 1)
        for y in range(0, img_h, 64):
            cv2.line(gorsel, (0, y), (img_w, y), (210, 210, 210), 1)

        etiketler = []

        for _ in range(nesne_sayisi):
            sinif_id = np.random.randint(0, len(cls.SINIFLAR))

            # Nesne boyutları
            if sinif_id == 0:  # Vida: İnce uzun dikdörtgen
                w_box = np.random.randint(25, 45)
                h_box = np.random.randint(70, 110)
            elif sinif_id == 1:  # Somun: Karemsi altıgen
                w_box = np.random.randint(50, 75)
                h_box = np.random.randint(50, 75)
            else:  # Rulman: Çemberimsi büyük nesne
                cap = np.random.randint(70, 100)
                w_box, h_box = cap, cap

            # Konum seç
            cx_pix = np.random.randint(w_box // 2 + 10, img_w - w_box // 2 - 10)
            cy_pix = np.random.randint(h_box // 2 + 10, img_h - h_box // 2 - 10)

            x1 = cx_pix - w_box // 2
            y1 = cy_pix - h_box // 2
            x2 = cx_pix + w_box // 2
            y2 = cy_pix + h_box // 2

            # Nesneyi çiz
            if sinif_id == 0:  # Vida çizimi
                cv2.rectangle(gorsel, (x1, y1), (x2, y2), (70, 70, 70), -1)
                # Vida başı
                cv2.rectangle(gorsel, (x1 - 4, y1), (x2 + 4, y1 + 15), (40, 40, 40), -1)
                # Vida yivleri
                for yy in range(y1 + 25, y2 - 5, 8):
                    cv2.line(gorsel, (x1, yy), (x2, yy + 3), (180, 180, 180), 2)

            elif sinif_id == 1:  # Somun çizimi (Altıgenimsi)
                cv2.rectangle(gorsel, (x1, y1), (x2, y2), (90, 90, 110), -1)
                cv2.rectangle(gorsel, (x1, y1), (x2, y2), (30, 30, 40), 2)
                # İç delik
                cv2.circle(gorsel, (cx_pix, cy_pix), min(w_box, h_box) // 4, (230, 230, 230), -1)
                cv2.circle(gorsel, (cx_pix, cy_pix), min(w_box, h_box) // 4, (40, 40, 40), 2)

            else:  # Rulman çizimi (İç içe halkalar ve bilyeler)
                r_dis = min(w_box, h_box) // 2
                cv2.circle(gorsel, (cx_pix, cy_pix), r_dis, (80, 80, 80), -1)
                cv2.circle(gorsel, (cx_pix, cy_pix), r_dis - 6, (160, 160, 160), -1)
                cv2.circle(gorsel, (cx_pix, cy_pix), r_dis // 2, (230, 230, 230), -1)
                cv2.circle(gorsel, (cx_pix, cy_pix), r_dis // 2, (50, 50, 50), 2)
                # Bilyeler
                for aci in np.linspace(0, 2 * np.pi, 7, endpoint=False):
                    bx = int(cx_pix + (r_dis * 0.72) * np.cos(aci))
                    by = int(cy_pix + (r_dis * 0.72) * np.sin(aci))
                    cv2.circle(gorsel, (bx, by), 5, (40, 40, 40), -1)

            # YOLO formatı normalize koordinatlar: [class, cx, cy, w, h] (0..1)
            norm_cx = cx_pix / img_w
            norm_cy = cy_pix / img_h
            norm_w = w_box / img_w
            norm_h = h_box / img_h

            etiketler.append((sinif_id, norm_cx, norm_cy, norm_w, norm_h))

        return gorsel, etiketler

    @classmethod
    def veri_seti_olustur(
        cls,
        ana_dizin: Path,
        train_adet: int = 30,
        val_adet: int = 10,
        img_w: int = 512,
        img_h: int = 512,
    ) -> Path:
        """Tüm train/val görsellerini ve etiketlerini oluşturur, 'data.yaml' dosyasını yazar."""
        np.random.seed(42)

        images_train_dir = ana_dizin / "images" / "train"
        images_val_dir = ana_dizin / "images" / "val"
        labels_train_dir = ana_dizin / "labels" / "train"
        labels_val_dir = ana_dizin / "labels" / "val"

        for d in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Train kümesi
        for i in range(train_adet):
            img, annots = cls.tek_gorsel_ve_etiket_uret(img_w, img_h, nesne_sayisi=np.random.randint(2, 5))
            img_path = images_train_dir / f"train_{i:04d}.jpg"
            label_path = labels_train_dir / f"train_{i:04d}.txt"

            cv2.imwrite(str(img_path), img)
            with open(label_path, "w", encoding="utf-8") as f:
                for a in annots:
                    f.write(f"{a[0]} {a[1]:.6f} {a[2]:.6f} {a[3]:.6f} {a[4]:.6f}\n")

        # Val kümesi
        for i in range(val_adet):
            img, annots = cls.tek_gorsel_ve_etiket_uret(img_w, img_h, nesne_sayisi=np.random.randint(2, 5))
            img_path = images_val_dir / f"val_{i:04d}.jpg"
            label_path = labels_val_dir / f"val_{i:04d}.txt"

            cv2.imwrite(str(img_path), img)
            with open(label_path, "w", encoding="utf-8") as f:
                for a in annots:
                    f.write(f"{a[0]} {a[1]:.6f} {a[2]:.6f} {a[3]:.6f} {a[4]:.6f}\n")

        # data.yaml dosyasını oluştur
        yaml_icerik = {
            "path": str(ana_dizin.resolve().as_posix()),
            "train": "images/train",
            "val": "images/val",
            "names": {i: ad for i, ad in enumerate(cls.SINIFLAR)},
        }

        yaml_path = ana_dizin / "data.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_icerik, f, sort_keys=False)

        return yaml_path
