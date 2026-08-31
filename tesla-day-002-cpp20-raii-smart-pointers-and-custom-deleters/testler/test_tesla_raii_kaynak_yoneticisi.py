"""
Tesla RAII ve Akilli Isaretci Birim Testleri (PyTest)
=====================================================
Bu test paketi; RAII kaynak yonetimi, custom deleter cagrilarini,
cifte kapama korumasini ve tasima (move) semantigini test eder.

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

from src.tesla_raii_kaynak_yoneticisi import (
    TeslaDonanimKaynagi,
    DonanimKaynakTipi,
    TeslaCANSoketRAII,
    OzelSiliciAkilliIsaretci,
    TeslaKaynakIzlemeMerkezi
)
from src.tesla_raii_profilleyici import TeslaRAIIProfilleyici


def test_tesla_can_soket_raii_otomatik_kapanma():
    """RAII soketi kapsamdan cikildiginda otomatik olarak kapatilmalidir."""
    kaynak_referansi = None
    with TeslaCANSoketRAII(arayuz_adi="can0") as soket:
        kaynak_referansi = soket.kaynak
        assert kaynak_referansi.acik_mi is True
        soket.telemetri_yaz(0x101, b"FSD_PING")
    
    assert kaynak_referansi is not None
    assert kaynak_referansi.acik_mi is False


def test_tesla_can_soket_cifte_kapatma_korumasi():
    """Ayni soketi birden fazla kez kapatmak (Idempotent) hata vermemelidir."""
    soket = TeslaCANSoketRAII(arayuz_adi="can1")
    soket.kapat()
    assert soket.kaynak.acik_mi is False
    
    # 2. kapatma guvenli olmali
    soket.kapat()
    assert soket.kaynak.acik_mi is False


def test_ozel_silici_akilli_isaretci_custom_deleter():
    """Custom Deleter fonksiyonu yok etme aninda basariyla tetiklenmelidir."""
    silindi_mi = False
    
    def ozel_temizleyici(k: TeslaDonanimKaynagi):
        nonlocal silindi_mi
        silindi_mi = True
        k.donanim_kapat()

    kaynak = TeslaDonanimKaynagi("GPU_TEX_01", DonanimKaynakTipi.GPU_TAMPON)
    with OzelSiliciAkilliIsaretci(kaynak, ozel_temizleyici) as ptr:
        assert ptr.al().kaynak_id == "GPU_TEX_01"
        assert silindi_mi is False

    assert silindi_mi is True
    assert kaynak.acik_mi is False


def test_akilli_isaretci_tasima_semantigi_move():
    """Sahiplik tasindiginda eski isaretci null olmali, yeni isaretci kaynagi yonetmelidir."""
    kaynak = TeslaDonanimKaynagi("DMA_CH_0", DonanimKaynakTipi.DMA_KANAL)
    ptr1 = OzelSiliciAkilliIsaretci(kaynak)
    
    ptr2 = ptr1.tasi()
    
    with pytest.raises(RuntimeError):
        ptr1.al()  # Tasinmis isaretciye erisilemez
        
    assert ptr2.al().kaynak_id == "DMA_CH_0"
    ptr2.serbest_birak_ve_yok_et()
    assert kaynak.acik_mi is False


def test_raii_istisna_guvenligi_ve_sifir_sizinti():
    """Kapsam icinde istisna olussa dahi RAII kaynagi sızıntısız kapatmalidir."""
    merkez = TeslaKaynakIzlemeMerkezi()
    
    try:
        with TeslaCANSoketRAII(arayuz_adi="can_err") as soket:
            merkez.kaydet(soket.kaynak)
            raise ValueError("Kritik Donanim Hatasi!")
    except ValueError:
        pass
    
    assert merkez.aktif_acik_kaynak_sayisi() == 0
    assert merkez.sizinti_orani() == 0.0
