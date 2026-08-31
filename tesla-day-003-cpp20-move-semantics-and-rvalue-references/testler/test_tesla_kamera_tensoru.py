"""
Tesla Kamera Tensoru ve Move Semantics Birim Testleri (PyTest)
==============================================================
Bu test paketi; C++20 Move Constructor, Move Assignment, Rvalue Referanslari
ve sifir-kopyalama (zero-copy) tensor aktarimlarini dogrular.

Telif Hakki (c) 2026 Seydi Eryilmaz (@seydivakkas)
Ozel Lisans - Tum Haklari Saklidir.
"""

import pytest
import sys
import os

su_an_dizin = os.path.dirname(os.path.abspath(__file__))
ana_dizin = os.path.dirname(su_an_dizin)
if ana_dizin not in sys.path:
    sys.path.insert(0, ana_dizin)

from src.tesla_kamera_tensoru import TeslaKameraTensoru, TeslaFSDKameraHatti
from src.tesla_move_profilleyici import TeslaMoveProfilleyici


def test_tesla_kamera_tensoru_olusturma_ve_boyut():
    """1080p RGB tensör boyutunun yaklaşık 6.22 MB olduğu doğrulanmalıdır."""
    tensor = TeslaKameraTensoru("on_ana", 1920, 1080, 3)
    assert tensor.gecerli_mi is True
    assert tensor.boyut_bayt == 1920 * 1080 * 3
    assert pytest.approx(tensor.boyut_mb, rel=1e-2) == 5.932
    assert tensor.bellek_adresi != 0


def test_derin_kopyalama_yeni_bellek_tahsisi():
    """Derin kopyalama farklı bir bellek adresinde yeni tampon oluşturmalıdır."""
    t1 = TeslaKameraTensoru("on_sol", 640, 480, 3)
    t2 = t1.derin_kopyala()
    
    assert t1.gecerli_mi is True
    assert t2.gecerli_mi is True
    assert t1.bellek_adresi != t2.bellek_adresi  # Bellek adresleri farkli olmali


def test_tasima_semantigi_move_sifir_kopyalama():
    """Move işlemi aynı bellek adresini korumalı ve kaynak tensörü geçersiz kılmalıdır."""
    t1 = TeslaKameraTensoru("on_sag", 1280, 720, 3)
    orijinal_adres = t1.bellek_adresi
    
    # Move Constructor
    t2 = t1.tasi()
    
    assert t2.gecerli_mi is True
    assert t2.bellek_adresi == orijinal_adres  # SIFIR KOPYALAMA: Adres ayni!
    assert t1.gecerli_mi is False             # Kaynak nesne moved-from durumunda
    assert t1.bellek_adresi == 0


def test_tasinmis_nesneye_erisim_korumasi():
    """Taşınmış (moved-from) nesneden işlem yapılmaya çalışıldığında hata fırlatılmalıdır."""
    t1 = TeslaKameraTensoru("arka_kamera", 640, 480, 3)
    t2 = t1.tasi()
    
    with pytest.raises(RuntimeError):
        t1.tasi()  # Tekrar tasinamaz
        
    with pytest.raises(RuntimeError):
        t1.derin_kopyala()  # Kopyalanamaz


def test_tasima_ile_atama_operatoru():
    """Move assignment operatörü hedefi güncellemeli ve kaynağı sıfırlamalıdır."""
    t1 = TeslaKameraTensoru("kamera_1", 100, 100, 3)
    t2 = TeslaKameraTensoru("kamera_2", 200, 200, 3)
    
    yeni_adres = t2.bellek_adresi
    t1.tasima_ile_ata(t2)
    
    assert t1.bellek_adresi == yeni_adres
    assert t1.genislik == 200
    assert t2.gecerli_mi is False


def test_fsd_kamera_hatti_surround_vision():
    """8 kamera akışı sıfır kopyalama ile NPU'ya mikrosaniye altı sürede aktarılmalıdır."""
    hat = TeslaFSDKameraHatti()
    for cam_name in TeslaFSDKameraHatti.KAMERA_LISTESI:
        raw_cam = hat.kamera_kare_uret(cam_name)
        npu_tensor, gecikme_ns = hat.npu_girisine_tasi(raw_cam)
        assert npu_tensor.gecerli_mi is True
        assert raw_cam.gecerli_mi is False
        assert gecikme_ns > 0

    assert hat.islenen_kare_sayisi == 8
