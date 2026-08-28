"""
Day 29: Çoklu Nesne Takibi, Kalman Filtresi, DeepSORT ve MOT Metrikleri Birim Testleri.
"""

import os
import numpy as np
import pytest

from src.kalman_filtresi import KalmanKutuFiltresi
from src.reid_cikarici import ReIDEmbeddingCikarici
from src.takipci_yoneticisi import Takipci, TakipDurumu, DeepSORTTakipci
from src.mot_metrik_motoru import MOTMetrikMotoru
from src.video_sahne_simulasyonu import VideoSahneSimulasyonu
from src.gorsellestirici import CokluNesneTakipGorsellestirici


def test_kalman_kutu_filtresi_tahmin_ve_guncelle():
    """Kalman filtresi ilklendirme, tahmin ve güncelleme adımları testi."""
    kalman = KalmanKutuFiltresi()
    kutu = np.array([10.0, 20.0, 50.0, 100.0])  # w=40, h=80, u=30, v=60, gamma=0.5

    x, P = kalman.ilklendir(kutu)
    assert x.shape == (8,)
    assert P.shape == (8, 8)
    assert pytest.approx(x[0], 0.01) == 30.0
    assert pytest.approx(x[1], 0.01) == 60.0
    assert pytest.approx(x[3], 0.01) == 80.0

    # Tahmin adımı
    x_pred, P_pred = kalman.tahmin(x, P)
    assert x_pred.shape == (8,)

    # Güncelleme adımı
    yeni_olcum = np.array([12.0, 22.0, 52.0, 102.0])
    x_up, P_up = kalman.guncelle(x_pred, P_pred, yeni_olcum)
    assert x_up.shape == (8,)


def test_kalman_mahalanobis_kapilama():
    """Mahalanobis kapılama mesafesi testi."""
    kalman = KalmanKutuFiltresi()
    kutu = np.array([10.0, 10.0, 50.0, 90.0])
    x, P = kalman.ilklendir(kutu)

    olcumler = np.array([
        [10.0, 10.0, 50.0, 90.0],   # Çok yakın (Düşük mesafe)
        [200.0, 200.0, 240.0, 280.0] # Çok uzak (Yüksek mesafe)
    ])

    mesafeler = kalman.mahalanobis_mesafesi(x, P, olcumler)
    assert mesafeler.shape == (2,)
    assert mesafeler[0] < mesafeler[1]
    assert mesafeler[0] < KalmanKutuFiltresi.MAHALANOBIS_ESIK_095


def test_reid_embedding_cikarici():
    """128D Re-ID öznitelik çıkarımı ve L2 norm testi."""
    reid = ReIDEmbeddingCikarici(feature_dim=128, device="cpu")
    crop1 = np.full((64, 32, 3), 200, dtype=np.uint8)
    crop2 = np.full((64, 32, 3), 50, dtype=np.uint8)

    embs = reid.cikar([crop1, crop2])
    assert embs.shape == (2, 128)

    # L2 Norm testi (Birim küre)
    normlar = np.linalg.norm(embs, axis=1)
    for n in normlar:
        assert pytest.approx(n, 0.001) == 1.0

    # Kosinüs mesafesi testi
    d_cos = ReIDEmbeddingCikarici.kosinus_mesafesi(embs[:1], embs[1:])
    assert d_cos.shape == (1, 1)
    assert d_cos[0, 0] >= 0.0


def test_takipci_durum_gecisi():
    """Takipçi durumunun TENTATIVE -> CONFIRMED -> DELETED geçişleri testi."""
    kalman = KalmanKutuFiltresi()
    kutu = np.array([10.0, 10.0, 40.0, 80.0])
    emb = np.random.randn(128).astype(np.float32)
    emb /= np.linalg.norm(emb)

    t = Takipci(kutu, emb, kalman, n_init=2, max_age=3)
    assert t.state == TakipDurumu.TENTATIVE

    # 1. Güncelleme ile n_init=2 sağlanır
    t.tahmin_et()
    t.guncelle(kutu, emb)
    assert t.state == TakipDurumu.CONFIRMED

    # Güncellenmeden max_age aşılırsa silinme
    for _ in range(4):
        t.tahmin_et()
    if t.time_since_update > t.max_age:
        t.silindi_isaretle()
    assert t.state == TakipDurumu.DELETED


def test_deepsort_takip_dongusu():
    """DeepSORT çoklu kare takip döngüsü ve kimlik koruma testi."""
    tracker = DeepSORTTakipci(max_cosine_distance=0.5, n_init=2, max_age=5)
    reid = ReIDEmbeddingCikarici(feature_dim=128, device="cpu")

    # 3 kare boyunca hareket eden 1 nesne
    crops = [np.full((64, 32, 3), 150, dtype=np.uint8)]
    embs = reid.cikar(crops)

    # 1. Kare (Tentative başlar)
    t1 = tracker.adim([np.array([10.0, 10.0, 40.0, 80.0])], embs)
    assert len(tracker.takipciler) == 1

    # 2. Kare (Confirmed olur)
    t2 = tracker.adim([np.array([12.0, 11.0, 42.0, 81.0])], embs)
    assert len(t2) == 1
    assert t2[0].track_id == 1

    # 3. Kare (Kimlik 1 korunur)
    t3 = tracker.adim([np.array([14.0, 12.0, 44.0, 82.0])], embs)
    assert len(t3) == 1
    assert t3[0].track_id == 1


def test_mot_metrik_motoru():
    """CLEAR MOT (MOTA, IDF1, IDSW) metrik hesaplama testi."""
    gt_frames = [
        [{"id": 1, "box": [10.0, 10.0, 40.0, 80.0]}],
        [{"id": 1, "box": [12.0, 11.0, 42.0, 81.0]}],
    ]
    pred_frames = [
        [{"id": 101, "box": [10.0, 10.0, 40.0, 80.0]}],
        [{"id": 101, "box": [12.0, 11.0, 42.0, 81.0]}],
    ]

    metrikler = MOTMetrikMotoru.degerlendir_video(gt_frames, pred_frames, iou_esigi=0.5)
    assert pytest.approx(metrikler["MOTA"], 0.001) == 1.0
    assert pytest.approx(metrikler["IDF1"], 0.001) == 1.0
    assert metrikler["IDSW"] == 0
    assert metrikler["FP"] == 0
    assert metrikler["FN"] == 0


def test_video_simulasyonu_ve_gorsellestirici(tmp_path):
    """Video sahne üretimi ve 6 panelli teşhis panosu çizim testi."""
    sim = VideoSahneSimulasyonu(genislik=256, yukseklik=192, toplam_kare=5, seed=42)
    video = sim.uret_video_dizisi()
    assert len(video) == 5

    kalman = KalmanKutuFiltresi()
    t = Takipci(np.array([10, 10, 40, 60]), np.random.randn(128), kalman)
    t.state = TakipDurumu.CONFIRMED

    maliyet = np.array([[0.1, 0.9], [0.8, 0.2]])
    reid_mat = np.array([[0.1, 0.9], [0.8, 0.2]])
    kalman_ornek = {"x_history": np.array([10, 15, 20]), "vx_history": np.array([5, 5, 5])}
    metrikler = {"MOTA": 0.95, "IDF1": 0.96, "Hassasiyet": 0.98, "Anma": 0.97, "MT": 4, "Toplam_Hedef": 4}

    out_file = str(tmp_path / "test_mot_paneli.png")
    cizim_path = CokluNesneTakipGorsellestirici.teshis_panosu_ciz(
        ornek_kare=video[0]["gorsel_rgb"],
        aktif_takipciler=[t],
        maliyet_matrisi=maliyet,
        reid_mesafe_matrisi=reid_mat,
        kalman_durum_ornek=kalman_ornek,
        metrikler=metrikler,
        hedef_path=out_file
    )
    assert os.path.exists(cizim_path)
