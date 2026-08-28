"""
Day 95: MiniViT v1 Sürüm Adayı ve Regresyon Test Paketi (PyTest).
8 adet kapsamlı birim ve entegrasyon testi içerir.
"""

import os
import shutil
import tempfile
import pytest
import torch
import numpy as np

from src.konfigurasyon import MiniViTConfig
from src.model import MiniViTForImageClassification
from src.surum_yoneticisi import ReleaseManifestYoneticisi, SurumAdayiPaketleyici
from src.regresyon_motoru import RegresyonDenetleyicisi, KaliteKapisi
from src.gorsellestirici import RCGorsellestirici


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def minivit_model():
    torch.manual_seed(42)
    config = MiniViTConfig(
        goruntu_boyutu=32,
        yama_boyutu=4,
        kanal_sayisi=3,
        gizli_boyut=64,
        katman_sayisi=2,
        dikkat_baslik_sayisi=2,
        ileri_besleme_boyutu=128,
        sinif_sayisi=10,
    )
    return MiniViTForImageClassification(config)


def test_minivit_rc_model_olusturma_ve_ileri_gecis(minivit_model):
    """MiniViT modelinin başlatılması ve ileri geçiş tensör boyutlarının doğrulanması."""
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = minivit_model(x)

    assert out.logits.shape == (2, 10)
    assert not torch.isnan(out.logits).any()


def test_sha256_hesaplayici_ve_dogrulama(temp_dir):
    """Dosya düzeyinde SHA-256 karma özetinin deterministik üretilmesi."""
    dosya_yolu = os.path.join(temp_dir, "test.bin")
    with open(dosya_yolu, "wb") as f:
        f.write(b"MiniViT-v1.0-RC1-Checksum-Data")

    sha = ReleaseManifestYoneticisi.dosya_sha256_hesapla(dosya_yolu)
    assert isinstance(sha, str)
    assert len(sha) == 64


def test_manifesto_olusturma_ve_imzalama(minivit_model, temp_dir):
    """Sürüm paketinin ve RELEASE_MANIFEST.json dosyasının doğru üretilmesi."""
    paketleyici = SurumAdayiPaketleyici()
    manifesto = paketleyici.paketi_hazirla(
        model=minivit_model,
        hedef_dizin=temp_dir,
        surum_etiketi="v1.0.0-rc1",
        repo_adi="test/minivit-rc1",
    )

    assert os.path.exists(os.path.join(temp_dir, "RELEASE_MANIFEST.json"))
    assert os.path.exists(os.path.join(temp_dir, "model.safetensors"))
    assert os.path.exists(os.path.join(temp_dir, "config.json"))
    assert manifesto["surum_bilgisi"]["release_tag"] == "v1.0.0-rc1"
    assert "manifesto_imzasi_sha256" in manifesto


def test_manifesto_bozulma_ve_tamir_denetimi(minivit_model, temp_dir):
    """Paketteki bir dosya değiştirildiğinde manifesto bütünlük denetiminin hata vermesi."""
    paketleyici = SurumAdayiPaketleyici()
    paketleyici.paketi_hazirla(
        model=minivit_model,
        hedef_dizin=temp_dir,
        surum_etiketi="v1.0.0-rc1",
    )

    yonetici = ReleaseManifestYoneticisi()
    dogrulama_ilk = yonetici.manifesto_dogrula(temp_dir)
    assert dogrulama_ilk["gecerli"] is True

    # Dosyalardan birini manipüle et (Bozulma simülasyonu)
    config_yolu = os.path.join(temp_dir, "config.json")
    with open(config_yolu, "a", encoding="utf-8") as f:
        f.write(" ")

    dogrulama_sonra = yonetici.manifesto_dogrula(temp_dir)
    assert dogrulama_sonra["gecerli"] is False
    assert len(dogrulama_sonra["bozuk_dosyalar"]) > 0


def test_altin_veri_seti_regresyon_testi(minivit_model):
    """Altın veri seti üzerinde sayısal tolerans karşılaştırması."""
    denetleyici = RegresyonDenetleyicisi()
    minivit_model.eval()

    torch.manual_seed(42)
    altin_x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        altin_logits = minivit_model(altin_x).logits

    # 1. Birebir aynı girdi ile test (Tolerans 1e-4)
    uyumlu, maks_fark, _ = denetleyici.altin_veri_seti_testi(
        model=minivit_model,
        altin_girdiler=altin_x,
        altin_hedef_logits=altin_logits,
        tolerans=1e-4,
    )
    assert uyumlu is True
    assert maks_fark < 1e-4

    # 2. Bozuk hedef ile test
    bozuk_logits = altin_logits + 1.0
    uyumlu_bozuk, _, _ = denetleyici.altin_veri_seti_testi(
        model=minivit_model,
        altin_girdiler=altin_x,
        altin_hedef_logits=bozuk_logits,
        tolerans=1e-4,
    )
    assert uyumlu_bozuk is False


def test_metrik_ve_sla_gecikme_testi(minivit_model):
    """Metrik hesaplaması ve çıkarım gecikmesi SLA bütçesi kontrolü."""
    denetleyici = RegresyonDenetleyicisi()

    # SLA Testi
    sla_uyumlu, p50, p95, gecikmeler = denetleyici.gecikme_sla_testi(
        model=minivit_model,
        girdi_sekli=(1, 3, 32, 32),
        iterasyon=20,
        max_p50_ms=20.0,
        max_p95_ms=50.0,
    )
    assert sla_uyumlu is True
    assert p50 > 0.0
    assert p95 >= p50
    assert len(gecikmeler) == 20


def test_bellek_kararlilik_ve_leak_denetimi(minivit_model):
    """Ardışık çıkarımlarda bellek artışının sınırlandırıldığının doğrulanması."""
    denetleyici = RegresyonDenetleyicisi()
    kararli, artis_yuzde, mem_list = denetleyici.bellek_kararlilik_testi(
        model=minivit_model,
        iterasyon=30,
        tolerans_artis_yuzde=10.0,
    )
    assert kararli is True
    assert len(mem_list) > 0


def test_kalite_kapisi_tam_is_akisi_ve_gorsellestirme(minivit_model, temp_dir):
    """Kalite kapısı tam denetiminin ve 6-panelli teşhis panosunun oluşturulması."""
    paketleyici = SurumAdayiPaketleyici()
    manifesto = paketleyici.paketi_hazirla(
        model=minivit_model,
        hedef_dizin=temp_dir,
        surum_etiketi="v1.0.0-rc1",
    )

    minivit_model.eval()
    altin_x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        altin_logits = minivit_model(altin_x).logits

    altin_veri = {"girdiler": altin_x, "logits": altin_logits}

    kk = KaliteKapisi()
    sonuc = kk.tam_denetim_yap(
        model=minivit_model,
        paket_dizini=temp_dir,
        altin_veri=altin_veri,
        max_p50_ms=30.0,
        max_p95_ms=60.0,
    )

    assert sonuc.onaylandi_mi is True
    assert "GO" in sonuc.nihai_karar

    # Görselleştirici Testi
    pano_yolu = os.path.join(temp_dir, "test_pano.png")
    gorsellestirici = RCGorsellestirici(dpi=100)
    gorsellestirici.pano_olustur(sonuc=sonuc, manifesto=manifesto, kayit_yolu=pano_yolu)

    assert os.path.exists(pano_yolu)
    assert os.path.getsize(pano_yolu) > 1000
