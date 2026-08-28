"""
Day 72: Temsil Uzayı Geometrisi ve Boyut İndirgeme Birim Testleri
----------------------------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
import pytest
import numpy as np
import torch

from src.model_ozellik_cikarici import GorselTemsilAgi, TemsilVeriUreteci
from src.boyut_indirgeme import BoyutIndirgemeMotoru
from src.geometri_analizoru import TemsilGeometrisiAnalizoru
from src.gorsellestirici import TemsilGeometrisiGorsellestirici


@pytest.fixture
def ornek_temsiller():
    X, y, meta = TemsilVeriUreteci.uret_kontrollu_temsiller(
        ornek_sayisi=150, boyut=32, sinif_sayisi=3, tohum=42
    )
    return X, y, meta


@pytest.fixture
def cokmus_temsiller():
    X_cokmus, y_cokmus = TemsilVeriUreteci.uret_boyutsal_cokmus_temsiller(
        ornek_sayisi=150, boyut=32, efektif_boyut=2, tohum=42
    )
    return X_cokmus, y_cokmus


def test_gorsel_temsil_agi_forward():
    model = GorselTemsilAgi(giris_kanali=3, temsil_boyutu=32, sinif_sayisi=3)
    x = torch.randn(4, 3, 32, 32)
    temsil, logits = model(x, normalize=True)
    
    assert temsil.shape == (4, 32)
    assert logits.shape == (4, 3)
    
    # L2 normalize edilmişlik testi (norm ~ 1.0)
    normlar = torch.norm(temsil, p=2, dim=1)
    assert torch.allclose(normlar, torch.ones(4), atol=1e-4)


def test_temsil_veri_ureteci(ornek_temsiller, cokmus_temsiller):
    X, y, meta = ornek_temsiller
    X_cok, y_cok = cokmus_temsiller
    
    assert X.shape == (150, 32)
    assert len(np.unique(y)) == 3
    assert X_cok.shape == (150, 32)


def test_pca_indirgeme(ornek_temsiller):
    X, _, _ = ornek_temsiller
    motor = BoyutIndirgemeMotoru(rastgele_tohum=42)
    
    X_pca, varyans = motor.uygula_pca(X, bilesen_sayisi=2)
    assert X_pca.shape == (150, 2)
    assert len(varyans) == 2
    assert 0.0 < np.sum(varyans) <= 1.0


def test_tsne_indirgeme(ornek_temsiller):
    X, _, _ = ornek_temsiller
    motor = BoyutIndirgemeMotoru(rastgele_tohum=42)
    
    X_tsne, kl_kayip = motor.uygula_tsne(X, bilesen_sayisi=2, perplexity=10.0, iterasyon_sayisi=300)
    assert X_tsne.shape == (150, 2)
    assert kl_kayip > 0.0


def test_umap_indirgeme(ornek_temsiller):
    X, _, _ = ornek_temsiller
    motor = BoyutIndirgemeMotoru(rastgele_tohum=42)
    
    X_umap = motor.uygula_umap(X, bilesen_sayisi=2, komsu_sayisi=10)
    assert X_umap.shape == (150, 2)


def test_izotropi_hesaplama(ornek_temsiller, cokmus_temsiller):
    X, _, _ = ornek_temsiller
    X_cok, _ = cokmus_temsiller
    analizor = TemsilGeometrisiAnalizoru()
    
    izotropi_norm = analizor.hesapla_izotropi(X)
    izotropi_cok = analizor.hesapla_izotropi(X_cok)
    
    assert 0.0 < izotropi_norm["izotropi_skoru"] <= 1.0
    assert 0.0 < izotropi_cok["izotropi_skoru"] <= 1.0
    # Sağlıklı temsilin izotropisi çökmüş olandan belirgin şekilde yüksek olmalıdır
    assert izotropi_norm["izotropi_skoru"] > izotropi_cok["izotropi_skoru"]
    assert izotropi_norm["efektif_boyut"] > izotropi_cok["efektif_boyut"]


def test_kosinus_geometrisi_ve_ayrisma(ornek_temsiller):
    X, y, _ = ornek_temsiller
    analizor = TemsilGeometrisiAnalizoru()
    
    sonuc = analizor.hesapla_kosinus_geometrisi(X, y)
    assert "sinif_ici_ortalama_kosinus" in sonuc
    assert "siniflar_arasi_ortalama_kosinus" in sonuc
    assert "ayrisma_marjini" in sonuc
    
    # Kümeli temsillerde sınıf içi benzerlik sınıflar arasından yüksek olmalıdır
    assert sonuc["sinif_ici_ortalama_kosinus"] > sonuc["siniflar_arasi_ortalama_kosinus"]
    assert sonuc["ayrisma_marjini"] > 0.0


def test_boyutsal_cokus_teshisi_ve_gorsellestirici(tmp_path, ornek_temsiller, cokmus_temsiller):
    X, y, _ = ornek_temsiller
    X_cok, _ = cokmus_temsiller
    analizor = TemsilGeometrisiAnalizoru()
    
    teshis_norm = analizor.teshis_boyutsal_cokus(X, esik_varyans=0.90, ilk_k=2)
    teshis_cok = analizor.teshis_boyutsal_cokus(X_cok, esik_varyans=0.90, ilk_k=2)
    
    assert teshis_norm["cokus_tespit_edildi"] is False
    assert teshis_cok["cokus_tespit_edildi"] is True
    
    # Görselleştirici Testi
    motor = BoyutIndirgemeMotoru(rastgele_tohum=42)
    X_pca, varyans = motor.uygula_pca(X, bilesen_sayisi=2)
    X_tsne, tsne_kl = motor.uygula_tsne(X, bilesen_sayisi=2, perplexity=10.0, iterasyon_sayisi=250)
    X_umap = motor.uygula_umap(X, bilesen_sayisi=2, komsu_sayisi=10)
    
    izotropi_norm = analizor.hesapla_izotropi(X)
    izotropi_cok = analizor.hesapla_izotropi(X_cok)
    kosinus_sonuc = analizor.hesapla_kosinus_geometrisi(X, y)
    
    test_cikti = str(tmp_path / "test_pano.png")
    gorsellestirici = TemsilGeometrisiGorsellestirici()
    kaydedilen = gorsellestirici.olustur_teshis_paneli(
        X_pca=X_pca,
        pca_varyans=varyans,
        X_tsne=X_tsne,
        tsne_kl=tsne_kl,
        X_umap=X_umap,
        y=y,
        izotropi_normal=izotropi_norm,
        izotropi_cokmus=izotropi_cok,
        kosinus_metrikleri=kosinus_sonuc,
        kayit_yolu=test_cikti
    )
    assert os.path.exists(kaydedilen)
    assert os.path.getsize(kaydedilen) > 10000
