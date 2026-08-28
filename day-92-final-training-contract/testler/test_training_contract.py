"""
Day 92: Eğitim Öncesi Veri Sözleşmesi ve Hazır Bulunuşluk Birim Testleri
-----------------------------------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
import tempfile
import numpy as np
import pytest
import torch

from src.sozlesme_kurallari import VeriSozlesmesi, IhlalSeviyesi
from src.sizinti_dedektoru import VeriSizintiDedektoru, SizintiRaporu
from src.veri_denetleyici import VeriDenetleyici, DenetimSonucu
from src.hazir_bulunusluk_kapisi import HazirBulunuslukKapisi, KapiDurumu, KapiKarari
from src.gorsellestirici import VeriSozlesmesiGorsellestirici


def test_sozlesme_varsayilan_degerleri():
    """VeriSozlesmesi sınıfının varsayılan kural parametrelerini ve özelliklerini test eder."""
    sozlesme = VeriSozlesmesi(beklenen_kanal=3, beklenen_yukseklik=32, beklenen_genislik=32)
    assert sozlesme.beklenen_sekil == (3, 32, 32)
    assert sozlesme.min_ornek_sayisi == 50
    assert sozlesme.nan_inf_yasak is True


def test_denetleyici_temiz_veri_gecerli():
    """Kurallara tam uyan temiz bir veri setinin 0 bloke hatayla geçtiğini doğrular."""
    denetleyici = VeriDenetleyici(VeriSozlesmesi(min_ornek_sayisi=20))
    temiz_x = torch.randn(50, 3, 32, 32).clamp(-3.0, 3.0)
    etiketler = torch.randint(0, 10, (50,))

    sonuc: DenetimSonucu = denetleyici.denetle(temiz_x, etiketler)

    assert sonuc.toplam_ornek == 50
    assert sonuc.gecerli_ornek_sayisi == 50
    assert sonuc.nan_inf_sayisi == 0
    assert sonuc.bloklayan_hata_var_mi is False


def test_denetleyici_sekil_ve_boyut_hatasi():
    """Hatalı boyuttaki tensörlerin BLOKE_EDICI ihlal olarak yakalandığını test eder."""
    denetleyici = VeriDenetleyici(VeriSozlesmesi(min_ornek_sayisi=10))

    # 3D tensör (N eksik)
    hatali_3d = torch.randn(3, 32, 32)
    sonuc1 = denetleyici.denetle(hatali_3d)
    assert any(i.kural_adi == "TENSOR_BOYUTU_HATASI" for i in sonuc1.ihlal_listesi)

    # Yanlış çözünürlük (3, 64, 64)
    hatali_res = torch.randn(20, 3, 64, 64)
    sonuc2 = denetleyici.denetle(hatali_res)
    assert any(i.kural_adi == "GIRDI_SEKIL_UYUMSUZLUGU" for i in sonuc2.ihlal_listesi)
    assert sonuc2.bloklayan_hata_var_mi is True


def test_denetleyici_nan_inf_yakalama():
    """Tensör içindeki NaN ve Inf değerlerinin anında bloke edici hata ürettiğini test eder."""
    denetleyici = VeriDenetleyici(VeriSozlesmesi(min_ornek_sayisi=10))
    x = torch.randn(30, 3, 32, 32)
    x[5, 0, 10, 10] = float("nan")
    x[12, 1, 5, 5] = float("inf")

    sonuc = denetleyici.denetle(x)
    assert sonuc.nan_inf_sayisi == 2
    assert sonuc.bloklayan_hata_var_mi is True
    assert any(i.kural_adi == "NAN_INF_TESPITI" for i in sonuc.ihlal_listesi)


def test_denetleyici_sinif_dengesizligi():
    """Aşırı sınıf dengesizliğinde uyarı üretildiğini doğrular."""
    denetleyici = VeriDenetleyici(VeriSozlesmesi(min_ornek_sayisi=10, maks_sinif_dengesizlik_orani=5.0))
    x = torch.randn(100, 3, 32, 32)

    # 0. sınıftan 95 adet, 1. sınıftan 5 adet (19x dengesizlik)
    etiketler = torch.tensor([0] * 95 + [1] * 5)
    sonuc = denetleyici.denetle(x, etiketler)

    assert any(i.kural_adi == "ASIRI_SINIF_DENGESIZLIGI" for i in sonuc.ihlal_listesi)


def test_sizinti_dedektoru_temiz_ve_sizintili():
    """Kriptografik hash ile Train-Val sızıntısının hatasız tespit edildiğini test eder."""
    dedektor = VeriSizintiDedektoru()

    train_x = torch.randn(40, 3, 32, 32)
    val_temiz = torch.randn(20, 3, 32, 32)

    # 1. Temiz durumda sızıntı yok
    rapor_temiz = dedektor.sizinti_tara(train_x, val_temiz)
    assert rapor_temiz.sizinti_var_mi is False
    assert rapor_temiz.kesisen_ornek_sayisi == 0

    # 2. Train'den 5 örneği Val setine kopyalayarak sızıntı oluştur
    val_sizintili = torch.cat([val_temiz[:15], train_x[:5]], dim=0)
    rapor_sizintili = dedektor.sizinti_tara(train_x, val_sizintili)

    assert rapor_sizintili.sizinti_var_mi is True
    assert rapor_sizintili.kesisen_ornek_sayisi == 5
    assert pytest.approx(rapor_sizintili.sizinti_orani_val, abs=1e-3) == 5 / 20


def test_hazir_bulunusluk_kapisi_onay_ve_bloke():
    """Karar kapısının temiz veride ONAYLANDI, sızıntılı veya bozuk veride BLOKE_EDILDI verdiğini test eder."""
    kapi = HazirBulunuslukKapisi(sızıntıda_bloke_et=True)
    denetleyici = VeriDenetleyici(VeriSozlesmesi(min_ornek_sayisi=10))

    # Senaryo 1: Kusursuz veri (her sınıfta 5 örnek, tam dengeli)
    temiz_x = torch.randn(25, 3, 32, 32)
    etiketler_dengeli = torch.tensor([0, 1, 2, 3, 4] * 5)
    sonuc_temiz = denetleyici.denetle(temiz_x, etiketler_dengeli)
    karar_temiz = kapi.degerlendir(sonuc_temiz)
    assert karar_temiz.durum == KapiDurumu.ONAYLANDI
    assert karar_temiz.egitim_baslatilabilir_mi is True

    # Senaryo 2: NaN içeren bozuk veri
    bozuk_x = torch.randn(25, 3, 32, 32)
    bozuk_x[0, 0, 0, 0] = float("nan")
    sonuc_bozuk = denetleyici.denetle(bozuk_x)
    karar_bozuk = kapi.degerlendir(sonuc_bozuk)
    assert karar_bozuk.durum == KapiDurumu.BLOKE_EDILDI
    assert karar_bozuk.egitim_baslatilabilir_mi is False


def test_gorsellestirici_pano_uretme():
    """6 panelli teşhis panosunun dosyaya hatasız çizildiğini test eder."""
    gorsellestirici = VeriSozlesmesiGorsellestirici(cizim_boyutu=(12, 8), dpi=100)

    with tempfile.TemporaryDirectory() as gecici_dizin:
        cikti_dosyasi = os.path.join(gecici_dizin, "test_veri_sozlesmesi.png")

        denetleyici = VeriDenetleyici(VeriSozlesmesi(min_ornek_sayisi=10))
        x = torch.randn(25, 3, 32, 32)
        sonuc = denetleyici.denetle(x, torch.randint(0, 5, (25,)))

        kapi = HazirBulunuslukKapisi()
        karar = kapi.degerlendir(sonuc)

        gorsellestirici.olustur_sozlesme_paneli(
            denetim_sonucu=sonuc,
            sizinti_raporu=None,
            kapi_karari=karar,
            ornek_tensörler=x.numpy(),
            kayit_yolu=cikti_dosyasi,
        )

        assert os.path.exists(cikti_dosyasi)
        assert os.path.getsize(cikti_dosyasi) > 1000
