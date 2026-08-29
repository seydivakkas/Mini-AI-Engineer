"""
FAZ 7 GÜN 125: Sandboxed Code Execution & Data Analysis Agent Testleri.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.guvenlik_denetleyicisi import AstGuvenlikDenetleyicisi
from src.izole_calistirici import IzoleKodCalistirici
from src.veri_analiz_ajani import VeriAnalizAjani
from src.gorsellestirici import SandboxGorsellestirici


def test_ast_guvenlik_temiz_kod():
    """AstGuvenlikDenetleyicisi sınıfının zararsız matematiksel kodları onayladığını test eder."""
    temiz_kod = (
        "x = np.array([1, 2, 3, 4, 5])\n"
        "ortalama = np.mean(x)\n"
        "print(f'Ortalama: {ortalama}')\n"
    )
    guvenli_mi, ihlaller, skor = AstGuvenlikDenetleyicisi.denetle(temiz_kod)

    assert guvenli_mi is True
    assert len(ihlaller) == 0
    assert skor == 100.0


def test_ast_guvenlik_yasakli_import_os_subprocess():
    """AstGuvenlikDenetleyicisi sınıfının yasaklı sistem modüllerini yakaladığını test eder."""
    zararli_kod = (
        "import os\n"
        "import subprocess\n"
        "os.system('ls -la')\n"
    )
    guvenli_mi, ihlaller, skor = AstGuvenlikDenetleyicisi.denetle(zararli_kod)

    assert guvenli_mi is False
    assert any("importu tespit edildi: 'os'" in i for i in ihlaller)
    assert any("importu tespit edildi: 'subprocess'" in i for i in ihlaller)
    assert skor < 50.0


def test_ast_guvenlik_yasakli_fonksiyonlar_eval_open():
    """AstGuvenlikDenetleyicisi sınıfının open(), eval() gibi tehlikeli yerleşik fonksiyonları engellediğini test eder."""
    zararli_kod = (
        "f = open('gizli_dosya.txt', 'w')\n"
        "eval('__import__(\"os\").system(\"whoami\")')\n"
    )
    guvenli_mi, ihlaller, skor = AstGuvenlikDenetleyicisi.denetle(zararli_kod)

    assert guvenli_mi is False
    assert any("open()" in i for i in ihlaller)
    assert any("eval()" in i for i in ihlaller)


def test_ast_guvenlik_kacis_vektorleri():
    """AstGuvenlikDenetleyicisi sınıfının __subclasses__ gibi sandbox kaçış vektörlerini yakaladığını test eder."""
    kacis_kodu = "alt_siniflar = ().__class__.__bases__[0].__subclasses__()\n"
    guvenli_mi, ihlaller, skor = AstGuvenlikDenetleyicisi.denetle(kacis_kodu)

    assert guvenli_mi is False
    assert any("__subclasses__" in i or "__bases__" in i for i in ihlaller)


def test_izole_calistirici_basarili_matematik():
    """IzoleKodCalistirici sınıfının güvenli kodları başarıyla çalıştırıp stdout'u yakaladığını test eder."""
    calistirici = IzoleKodCalistirici()
    kod = (
        "a = 10\n"
        "b = 20\n"
        "print(f'Toplam: {a + b}')\n"
    )
    sonuc = calistirici.calistir(kod)

    assert sonuc.basarili is True
    assert "Toplam: 30" in sonuc.stdout
    assert sonuc.stderr == ""


def test_izole_calistirici_saldiri_engelleme():
    """IzoleKodCalistirici sınıfının güvenlik ihlalinde çalıştırmayı durdurduğunu test eder."""
    calistirici = IzoleKodCalistirici()
    zararli_kod = "import sys\nsys.exit(1)\n"
    sonuc = calistirici.calistir(zararli_kod)

    assert sonuc.basarili is False
    assert "GÜVENLİK İHLALİ" in sonuc.stderr
    assert len(sonuc.guvenlik_ihlalleri) > 0


def test_veri_analiz_ajani_calistirma():
    """VeriAnalizAjani sınıfının veri analiz kodu üretip izole ortamda çalıştırdığını test eder."""
    ajan = VeriAnalizAjani()
    with tempfile.TemporaryDirectory() as tmp_dir:
        rapor = ajan.analizi_calistir(
            veri_seti_tanimi="6 aylık satış ve kar verisi",
            analiz_hedefi="Gelir ve kar marjını analiz et",
            grafik_dizini=tmp_dir,
        )

        assert rapor["basarili"] is True
        assert "FİNANSAL VERİ ANALİZİ RAPORU" in rapor["stdout"]
        assert rapor["grafik_sayisi"] >= 1
        assert len(rapor["grafik_dosyalari"]) >= 1


def test_gorsellestirici_pano():
    """SandboxGorsellestirici sınıfının 6 panelli PNG teşhis dosyasını ürettiğini test eder."""
    ajan = VeriAnalizAjani()
    with tempfile.TemporaryDirectory() as tmp_dir:
        rapor = ajan.analizi_calistir(
            veri_seti_tanimi="Satış verisi",
            analiz_hedefi="Trend analizi",
            grafik_dizini=tmp_dir,
        )
        karsilastirma = ajan.benchmark_karsilastir()

        gorsellestirici = SandboxGorsellestirici(dpi=100)
        kayit = os.path.join(tmp_dir, "test_sandbox_pano.png")
        gorsellestirici.pano_olustur(rapor, karsilastirma, kayit_yolu=kayit)

        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
