"""
Day 29: Çoklu Nesne Takibi & Kalman Filtresi / DeepSORT Ana Yürütme Betiği.
"""

import os
import numpy as np

from src.video_sahne_simulasyonu import VideoSahneSimulasyonu
from src.reid_cikarici import ReIDEmbeddingCikarici
from src.takipci_yoneticisi import DeepSORTTakipci
from src.mot_metrik_motoru import MOTMetrikMotoru
from src.gorsellestirici import CokluNesneTakipGorsellestirici


def main():
    print("=" * 80)
    print(">>> ASAMA 1: Yapay Video Dizisi ve Kesişen Yörüngelerin Simülasyonu")
    print("=" * 80)
    simulasyon = VideoSahneSimulasyonu(genislik=512, yukseklik=384, toplam_kare=40, seed=42)
    video_kareleri = simulasyon.uret_video_dizisi()
    print(f"[+] Toplam Video Karesi      : {len(video_kareleri)}")
    print(f"[+] Sahne Çözünürlüğü       : 512x384")
    print(f"[+] Simüle Edilen Hedefler  : 4 Hareketli Nesne (Kesişme ve Kapanma Olayları)")

    print("\n" + "=" * 80)
    print(">>> ASAMA 2: Re-ID Embedding Çıkarıcı & DeepSORT Takipçisinin Başlatılması")
    print("=" * 80)
    reid_extractor = ReIDEmbeddingCikarici(feature_dim=128, device="cpu")
    tracker = DeepSORTTakipci(max_cosine_distance=0.40, nn_budget=100, max_age=15, n_init=2)
    print("[+] 128 Boyutlu L2 Normalize Re-ID Modülü Hazır")
    print("[+] 8 Boyutlu Kalman Filtresi (Sabit Hızlı Hareket Modeli) Aktif")
    print("[+] Macar Algoritması Eşleme (Kosinüs + Mahalanobis Kapılama) Devrede")

    print("\n" + "=" * 80)
    print(">>> ASAMA 3: Kare Kare Çoklu Nesne Takip Döngüsü (Online Tracking)")
    print("=" * 80)

    tum_kareler_gt = []
    tum_kareler_tahmin = []

    ornek_kare_img = None
    ornek_aktif_takipciler = []
    kalman_u_history = []
    kalman_vu_history = []

    ornek_maliyet_matrisi = np.zeros((4, 4))
    ornek_reid_matrisi = np.zeros((4, 4))

    for f_idx, kare in enumerate(video_kareleri):
        tespitler = kare["tespitler"]
        kirpintilar = kare["kirpintilar"]
        gt_hedefler = kare["gt_hedefler"]

        # 1. Re-ID Embeddinglerini Çıkar
        embeddings = reid_extractor.cikar(kirpintilar)

        # 2. DeepSORT Takip Adımı
        aktif_takipciler = tracker.adim(tespitler, embeddings)

        # Tahminleri Kaydet
        kare_tahminleri = []
        for t in aktif_takipciler:
            kare_tahminleri.append({
                "id": t.track_id,
                "box": t.guncel_kutu().tolist()
            })

        tum_kareler_gt.append(gt_hedefler)
        tum_kareler_tahmin.append(kare_tahminleri)

        # Örnek 1. Nesnenin Kalman Durum Geçmişini Kaydet
        for t in tracker.takipciler:
            if t.track_id == 1:
                kalman_u_history.append(float(t.mean[0]))
                kalman_vu_history.append(float(t.mean[4]))

        # Orta karede görselleştirme için anlık verileri yakala
        if f_idx == 20:
            ornek_kare_img = kare["gorsel_rgb"].copy()
            ornek_aktif_takipciler = list(tracker.takipciler)
            if len(tracker.takipciler) > 0 and len(tespitler) > 0:
                t_feats = np.array([t.features[-1] for t in tracker.takipciler])
                ornek_reid_matrisi = reid_extractor.kosinus_mesafesi(t_feats, embeddings)
                ornek_maliyet_matrisi = ornek_reid_matrisi.copy()

        if (f_idx + 1) % 10 == 0:
            print(f"[*] Kare [{f_idx+1:02d}/40] İşlendi | Aktif Onaylı Takipçi Sayısı: {len(aktif_takipciler)}")

    print("\n" + "=" * 80)
    print(">>> ASAMA 4: CLEAR MOT & IDF1 Metriklerinin Değerlendirilmesi")
    print("=" * 80)
    metrikler = MOTMetrikMotoru.degerlendir_video(
        kareler_gt=tum_kareler_gt,
        kareler_tahmin=tum_kareler_tahmin,
        iou_esigi=0.5
    )

    print(f"[+] MOTA (Tracking Accuracy) : %{metrikler['MOTA']*100:.2f}")
    print(f"[+] IDF1 (ID F1-Score)       : %{metrikler['IDF1']*100:.2f}")
    print(f"[+] Kimlik Değişimi (IDSW)   : {metrikler['IDSW']} adet")
    print(f"[+] Yanlış Pozitif (FP)      : {metrikler['FP']} adet")
    print(f"[+] Kaçırılan Hedef (FN)     : {metrikler['FN']} adet")
    print(f"[+] Hassasiyet (Precision)   : %{metrikler['Hassasiyet']*100:.2f}")
    print(f"[+] Anma (Recall)            : %{metrikler['Anma']*100:.2f}")
    print(f"[+] Çoğunlukla İzlenen (MT)  : {metrikler['MT']}/{metrikler['Toplam_Hedef']}")

    print("\n" + "=" * 80)
    print(">>> ASAMA 5: 6 Panelli Teşhis Panosunun (Dashboard) Üretilmesi")
    print("=" * 80)
    kalman_ornek = {
        "x_history": np.array(kalman_u_history),
        "vx_history": np.array(kalman_vu_history)
    }

    if ornek_kare_img is None:
        ornek_kare_img = video_kareleri[-1]["gorsel_rgb"]
        ornek_aktif_takipciler = tracker.takipciler

    cikis_resmi = CokluNesneTakipGorsellestirici.teshis_panosu_ciz(
        ornek_kare=ornek_kare_img,
        aktif_takipciler=ornek_aktif_takipciler,
        maliyet_matrisi=ornek_maliyet_matrisi,
        reid_mesafe_matrisi=ornek_reid_matrisi,
        kalman_durum_ornek=kalman_ornek,
        metrikler=metrikler,
        hedef_path="day-29-multi-object-tracking/ciktilar/coklu_nesne_takip_paneli.png"
    )
    print(f"[+] 6 Panelli MOT Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikis_resmi)}")
    print("=" * 80)
    print("Day 29: Çoklu Nesne Takibi & DeepSORT Başarıyla Tamamlandı!")
    print("=" * 80)


if __name__ == "__main__":
    main()
