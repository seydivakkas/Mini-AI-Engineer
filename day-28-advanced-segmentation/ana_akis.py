"""
Day 28: İleri Düzey Bölütleme & Mask R-CNN / SegFormer Ana Yürütme Betiği.
"""

import os
import numpy as np
import torch
import cv2

from src.sentetik_sahne_ureteci import SentetikSahneUreteci
from src.bolutleme_turleri import PanoptikDonusturucu
from src.mask_rcnn_modulu import MaskRCNNYoneticisi
from src.segformer_mimari import SegFormerModeli
from src.panoptik_ve_mask_metrikleri import PanoptikMetrikHesaplayici, MaskeIoUHesaplayici
from src.gorsellestirici import IleriBolutlemeGorsellestirici


def main():
    print("=" * 80)
    print(">>> ASAMA 1: Sentetik Coklu Nesne Sahnesinin Uretimi (Things + Stuff)")
    print("=" * 80)
    uretec = SentetikSahneUreteci(img_size=256, seed=42)
    sahne = uretec.sahne_uret(num_instances=6)

    print(f"[+] Sahne Cozunurlugu       : {sahne['gorsel_rgb'].shape}")
    print(f"[+] Uretilen Nesne Sayisi   : {len(sahne['ornek_maskeleri'])}")
    print(f"[+] Semantik Sinif Dagilimi : {np.unique(sahne['semantik_harita'])}")
    print(f"[+] Panoptik Segment Sayisi : {len(sahne['panoptik_segmentler'])}")

    print("\n" + "=" * 80)
    print(">>> ASAMA 2: Mask R-CNN Mimarisi & RoIAlign + FCN Maske Basligi")
    print("=" * 80)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Calisma Cihazi          : {device}")

    mask_rcnn = MaskRCNNYoneticisi(girdi_kanali=256, sinif_sayisi=5).to(device)
    mask_rcnn.eval()

    # Sahte omurga ozellik haritasi (B, 256, 256, 256)
    feature_map = torch.randn(1, 256, 256, 256, device=device)
    # RoI listesi: [batch_id, x1, y1, x2, y2]
    rois_list = [[0, b[0], b[1], b[2], b[3]] for b in sahne["ornek_kutulari"]]
    rois_tensor = torch.tensor(rois_list, dtype=torch.float32, device=device)

    with torch.no_grad():
        rcnn_ciktilari = mask_rcnn(feature_map, rois_tensor)

    print(f"[+] RoIAlign Cikti Boyutu   : {rcnn_ciktilari['roi_features'].shape}")
    print(f"[+] Maske Lojitleri         : {rcnn_ciktilari['mask_logits'].shape} (28x28 ikili maskeler)")
    print(f"[+] Siniflandirma Ciktisi   : {rcnn_ciktilari['cls_logits'].shape}")
    print(f"[+] Kutu Regresyon Ciktisi  : {rcnn_ciktilari['box_deltas'].shape}")

    # Coklu Gorev Kaybi (Multi-Task Loss) Hesaplama
    gt_labels = torch.tensor(sahne["ornek_siniflari"], dtype=torch.long, device=device)
    gt_boxes = torch.zeros((len(sahne["ornek_kutulari"]), 4), device=device)
    gt_masks = torch.zeros((len(sahne["ornek_maskeleri"]), 28, 28), device=device)
    for i, m in enumerate(sahne["ornek_maskeleri"]):
        m_resized = cv2.resize(m, (28, 28), interpolation=cv2.INTER_NEAREST)
        gt_masks[i] = torch.tensor(m_resized, device=device)

    kayiplar = MaskRCNNYoneticisi.coklu_gorev_kaybi(
        cls_logits=rcnn_ciktilari["cls_logits"],
        box_deltas=rcnn_ciktilari["box_deltas"],
        mask_logits=rcnn_ciktilari["mask_logits"],
        gt_labels=gt_labels,
        gt_boxes=gt_boxes,
        gt_masks=gt_masks
    )
    print(f"[+] Multi-Task Loss Toplami : {kayiplar['toplam_kayip'].item():.4f} (L_cls: {kayiplar['l_cls']:.3f}, L_box: {kayiplar['l_box']:.3f}, L_mask: {kayiplar['l_mask']:.3f})")

    print("\n" + "=" * 80)
    print(">>> ASAMA 3: SegFormer Vision Transformer Mimarisi")
    print("=" * 80)
    segformer = SegFormerModeli(in_channels=3, num_classes=5, embed_dims=[32, 64, 128, 256]).to(device)
    segformer.eval()
    toplam_param = sum(p.numel() for p in segformer.parameters() if p.requires_grad)
    print(f"[+] SegFormer Modeli Baslatildi (Egitilebilir Parametre: {toplam_param:,})")

    img_tensor = torch.tensor(sahne["gorsel_rgb"], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    with torch.no_grad():
        seg_logits = segformer(img_tensor)
    print(f"[+] SegFormer Cikti Boyutu   : {seg_logits.shape} (B, NumClasses, H, W)")

    print("\n" + "=" * 80)
    print(">>> ASAMA 4: Panoptik Kalite (PQ) ve Maske AP Metrikleri")
    print("=" * 80)
    # Tahmin edilen panoptik harita (gercek ile hafif gurultulu ornek)
    pred_panoptic = sahne["panoptik_harita"].copy()
    # Hafif kenar asindirmasi ile gercekci model cikarimi olustur
    kernel = np.ones((3, 3), np.uint8)
    for seg in sahne["panoptik_segmentler"]:
        if seg["kategori_tipi"] == "thing":
            m = (pred_panoptic == seg["id"]).astype(np.uint8)
            m_eroded = cv2.erode(m, kernel, iterations=1)
            pred_panoptic[m == 1] = 0
            pred_panoptic[m_eroded == 1] = seg["id"]

    pq_sonuclari = PanoptikMetrikHesaplayici.hesapla_pq(
        pred_panoptic=pred_panoptic,
        gt_panoptic=sahne["panoptik_harita"],
        iou_esigi=0.5
    )

    pred_scores = [0.95, 0.91, 0.88, 0.85, 0.82, 0.79]
    ap_sonuclari = PanoptikMetrikHesaplayici.hesapla_instance_ap(
        pred_masks=sahne["ornek_maskeleri"],
        pred_scores=pred_scores,
        pred_labels=sahne["ornek_siniflari"],
        gt_masks=sahne["ornek_maskeleri"],
        gt_labels=sahne["ornek_siniflari"]
    )

    metrikler = {
        "genel_pq": pq_sonuclari["genel_pq"],
        "genel_sq": pq_sonuclari["genel_sq"],
        "genel_rq": pq_sonuclari["genel_rq"],
        "AP_50": ap_sonuclari["AP_50"],
        "AP_75": ap_sonuclari["AP_75"]
    }

    print(f"[+] Panoptik Kalite (PQ)    : %{metrikler['genel_pq']*100:.2f}")
    print(f"[+] Bolutleme Kalitesi (SQ) : %{metrikler['genel_sq']*100:.2f}")
    print(f"[+] Tanima Kalitesi (RQ)    : %{metrikler['genel_rq']*100:.2f}")
    print(f"[+] Ornek Maske AP@50       : %{metrikler['AP_50']*100:.2f}")
    print(f"[+] Ornek Maske AP@75       : %{metrikler['AP_75']*100:.2f}")

    print("\n" + "=" * 80)
    print(">>> ASAMA 5: 6 Panelli Teshis Panosunun (Dashboard) Uretilmesi")
    print("=" * 80)
    # RoIAlign ve Maske ornek gorseli
    roi_ornek_feat = rcnn_ciktilari["roi_features"][0, 0].cpu().numpy()
    roi_ornek_feat = cv2.resize(roi_ornek_feat, (28, 28))
    mask_pred_ornek = torch.sigmoid(rcnn_ciktilari["mask_logits"][0, sahne["ornek_siniflari"][0]]).cpu().numpy()

    cikis_resmi = IleriBolutlemeGorsellestirici.teshis_panosu_ciz(
        sahne_verisi=sahne,
        roi_align_ornek=roi_ornek_feat,
        mask_pred_ornek=mask_pred_ornek,
        metrikler=metrikler,
        hedef_path="day-28-advanced-segmentation/ciktilar/ileri_bolutleme_teshis_paneli.png"
    )
    print(f"[+] 6 Panelli Teshis Panosu Basariyla Kaydedildi: {os.path.abspath(cikis_resmi)}")
    print("=" * 80)
    print("Day 28: Ileri Duzey Bolutleme & Mask R-CNN / SegFormer Basariyla Tamamlandi!")
    print("=" * 80)


if __name__ == "__main__":
    main()
