"""Ultralytics YOLOv8 / YOLO11 Model Yönetici ve Çıkarım Modülü.

Bu modül; Ultralytics YOLO mimarisini yükleme, özel veri seti (data.yaml) üzerinde
eğitme (Training), doğrulama (Validation) ve görsel/video kareleri üzerinde
yüksek hızlı çıkarım (Inference / Prediction) işlemlerini yönetir.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import cv2
import numpy as np
from ultralytics import YOLO


class YOLOYoneticisi:
    """YOLO model eğitimi, doğrulaması ve çıkarımını yöneten sınıf."""

    def __init__(self, model_adi: str = "yolov8n.pt") -> None:
        """YOLO modelini başlatır."""
        self.model_adi = model_adi
        self.model = YOLO(model_adi)

    def egit(
        self,
        data_yaml: Union[str, Path],
        epochs: int = 3,
        imgsz: int = 512,
        batch: int = 4,
        device: str = "cpu",
        proje_dizini: str = "egitim_ciktilari",
        deney_adi: str = "yolo_deney",
    ) -> Dict:
        """YOLO modelini özel veri seti üzerinde eğitir."""
        sonuclar = self.model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=proje_dizini,
            name=deney_adi,
            exist_ok=True,
            verbose=False,
            plots=False,
        )
        return sonuclar

    def dogrula(
        self,
        data_yaml: Optional[Union[str, Path]] = None,
        device: str = "cpu",
    ) -> Dict:
        """Eğitilmiş modeli doğrulama kümesinde değerlendirir."""
        val_sonuc = self.model.val(
            data=str(data_yaml) if data_yaml else None,
            device=device,
            verbose=False,
        )
        return {
            "map50": float(val_sonuc.box.map50),
            "map75": float(val_sonuc.box.map75),
            "map50_95": float(val_sonuc.box.map),
            "precision": float(val_sonuc.box.mp),
            "recall": float(val_sonuc.box.mr),
        }

    def cikarim_yap(
        self,
        kaynak: Union[str, Path, np.ndarray],
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = "cpu",
    ) -> List[Dict]:
        """Görsel üzerinde nesne tespiti çıkarımı yapar."""
        tahminler = self.model.predict(
            source=kaynak,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )

        tespitler = []
        if len(tahminler) > 0:
            sonuc = tahminler[0]
            boxes = sonuc.boxes.xyxy.cpu().numpy()
            scores = sonuc.boxes.conf.cpu().numpy()
            classes = sonuc.boxes.cls.cpu().numpy().astype(int)
            names = sonuc.names

            for box, score, cls_id in zip(boxes, scores, classes):
                tespitler.append({
                    "box": box.tolist(),  # [x1, y1, x2, y2]
                    "score": float(score),
                    "class_id": int(cls_id),
                    "class_name": names.get(cls_id, str(cls_id)),
                })

        return tespitler
