"""
Platform Yöneticisi: Uçtan Uca Çoklu Görev Görsel Analiz Platformu Orkestratörü.
"""

from typing import Dict, Any, List
import numpy as np
import torch
import cv2

from .coklu_gorev_modeli import CokluGorevGorselModeli
from .takip_ve_analitik_motoru import CokluGorevTakipAnalitikMotoru


class PlatformYoneticisi:
    """
    Tüm çoklu görev bileşenlerini (Sınıflandırma, Tespit, Bölütleme, Takip, Telemetri)
    tek bir entegre üretim hattında (Production Pipeline) koordine eder.
    """

    SAHNE_SINIFLARI = {0: "Otoyol (Highway)", 1: "Şehir İçi (Urban)", 2: "Otopark (Parking)", 3: "Tünel (Tunnel)"}
    NESNE_SINIFLARI = {0: "Arka Plan", 1: "Araç", 2: "Yaya", 3: "Trafik İşareti"}
    SEG_SINIFLARI = {0: "Gökyüzü", 1: "Yol", 2: "Bina", 3: "Araç", 4: "Yaya"}

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = CokluGorevGorselModeli(
            in_channels=3,
            num_scene_classes=4,
            num_object_classes=4,
            num_seg_classes=5,
            reid_dim=128
        ).to(self.device)
        self.model.eval()

        self.takip_motoru = CokluGorevTakipAnalitikMotoru(max_cosine_dist=0.40, max_age=15)

    def isle_kare(self, kare_rgb: np.ndarray, simule_nesneler: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Tek bir video karesini tüm çoklu görev başlıklarından geçirir ve
        yapılandırılmış analiz telemetrisi üretir.
        """
        h, w, _ = kare_rgb.shape
        img_tensor = torch.tensor(kare_rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0

        with torch.no_grad():
            ciktilar = self.model(img_tensor)

        # 1. Sahne Sınıflandırma
        scene_probs = torch.softmax(ciktilar["scene_logits"], dim=1)[0].cpu().numpy()
        scene_id = int(np.argmax(scene_probs))
        scene_label = self.SAHNE_SINIFLARI.get(scene_id, "Bilinmeyen")
        scene_conf = float(scene_probs[scene_id])

        # 2. Anlamsal Bölütleme Maskesi
        seg_probs = torch.softmax(ciktilar["seg_logits"], dim=1)[0].cpu().numpy()
        seg_mask = np.argmax(seg_probs, axis=0).astype(np.uint8)

        # 3. Nesne Tespiti ve Re-ID Çıkarımı
        tespit_kutulari = []
        tespit_siniflari = []
        tespit_embeddingleri = []

        # Gerçekçi simüle edilmiş nesnelerden kutu ve Re-ID üretimi
        for obj in simule_nesneler:
            box = np.array(obj["box"], dtype=np.float32)
            lbl = obj.get("sinif_id", 1)
            # Re-ID embedding (modelin reid çıktısı ile harmanlanmış)
            crop_h = int(box[3] - box[1])
            crop_w = int(box[2] - box[0])
            if crop_h > 5 and crop_w > 5:
                crop = kare_rgb[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                crop_resized = cv2.resize(crop, (32, 64))
                crop_tensor = torch.tensor(crop_resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
                with torch.no_grad():
                    emb = self.model.reid_head(self.model.layer3(self.model.layer2(self.model.layer1(self.model.stem(crop_tensor))))).cpu().numpy()[0]
                    emb /= (np.linalg.norm(emb) + 1e-6)
            else:
                emb = np.random.randn(128).astype(np.float32)
                emb /= np.linalg.norm(emb)

            tespit_kutulari.append(box)
            tespit_siniflari.append(lbl)
            tespit_embeddingleri.append(emb)

        if tespit_embeddingleri:
            emb_matrisi = np.array(tespit_embeddingleri)
        else:
            emb_matrisi = np.empty((0, 128), dtype=np.float32)

        # 4. Çoklu Nesne Takibi
        aktif_takipciler = self.takip_motoru.guncelle(
            tespit_kutulari=tespit_kutulari,
            tespit_embeddingleri=emb_matrisi,
            tespit_siniflari=tespit_siniflari
        )

        return {
            "sahne_etiketi": scene_label,
            "sahne_guveni": scene_conf,
            "seg_maskesi": seg_mask,
            "tespit_kutulari": tespit_kutulari,
            "tespit_siniflari": tespit_siniflari,
            "aktif_takipciler": aktif_takipciler,
            "omurga_aktivasyonu": ciktilar["backbone_features"][2][0, 0].cpu().numpy()
        }
