"""
FAZ 7 GÜN 124: JSON Schema ve Tip Güvenli Tool Calling Testleri.
Tüm testler %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest

from src.arac_semasi import ParametreOzelligi, AracSemasi
from src.json_ayristirici import GuvenliJsonAyristirici
from src.arac_yonlendirici import AracYonlendirici
from src.gramer_kisitlayici import GramerKisitlayici
from src.gorsellestirici import ToolCallingGorsellestirici


def test_parametre_ozelligi_ve_openai_sema():
    """AracSemasi sınıfının OpenAI standardında geçerli JSON Schema ürettiğini test eder."""
    sema = AracSemasi(
        ad="HavaDurumu",
        aciklama="Şehrin anlık hava durumunu getirir.",
        parametreler={
            "sehir": ParametreOzelligi(tip="string", aciklama="Şehir adı"),
            "birim": ParametreOzelligi(tip="string", aciklama="Sıcaklık birimi", enum=["C", "F"]),
        },
        zorunlu_alanlar=["sehir"],
    )
    openai_dict = sema.to_openai_schema()

    assert openai_dict["type"] == "function"
    assert openai_dict["function"]["name"] == "HavaDurumu"
    assert "sehir" in openai_dict["function"]["parameters"]["properties"]
    assert openai_dict["function"]["parameters"]["required"] == ["sehir"]


def test_sema_dogrulama_gecerli_argumanlar():
    """AracSemasi sınıfının geçerli argümanları başarıyla onayladığını test eder."""
    sema = AracSemasi(
        ad="Borsa",
        aciklama="Hisse bilgisi",
        parametreler={
            "sembol": ParametreOzelligi(tip="string", aciklama="Kod", enum=["THYAO", "ASELS"]),
            "adet": ParametreOzelligi(tip="integer", aciklama="Adet", minimum=1, maximum=100),
        },
        zorunlu_alanlar=["sembol", "adet"],
    )
    gecerli_arg = {"sembol": "THYAO", "adet": 10}
    dogru_mu, hatalar = sema.dogrula(gecerli_arg)

    assert dogru_mu is True
    assert len(hatalar) == 0


def test_sema_dogrulama_tip_hatasi_ve_eksik_alan():
    """AracSemasi sınıfının tip uyuşmazlığı ve eksik zorunlu alanları yakaladığını test eder."""
    sema = AracSemasi(
        ad="Borsa",
        aciklama="Hisse bilgisi",
        parametreler={
            "sembol": ParametreOzelligi(tip="string", aciklama="Kod"),
            "adet": ParametreOzelligi(tip="integer", aciklama="Adet"),
        },
        zorunlu_alanlar=["sembol", "adet"],
    )
    # Eksik alan ve tip hatası (adet string olarak verilmiş)
    hatali_arg = {"adet": "on"}
    dogru_mu, hatalar = sema.dogrula(hatali_arg)

    assert dogru_mu is False
    assert any("Zorunlu alan eksik: 'sembol'" in h for h in hatalar)
    assert any("'adet' integer olmalıdır" in h for h in hatalar)


def test_sema_dogrulama_enum_ve_aralik():
    """AracSemasi sınıfının geçersiz enum değerini ve sınır dışı sayıları yakaladığını test eder."""
    sema = AracSemasi(
        ad="Ucus",
        aciklama="Bilet",
        parametreler={
            "sinif": ParametreOzelligi(tip="string", aciklama="Sınıf", enum=["Economy", "Business"]),
            "yolcu": ParametreOzelligi(tip="integer", aciklama="Yolcu", minimum=1, maximum=5),
        },
        zorunlu_alanlar=["sinif", "yolcu"],
    )
    hatali_arg = {"sinif": "FirstClass", "yolcu": 10}
    dogru_mu, hatalar = sema.dogrula(hatali_arg)

    assert dogru_mu is False
    assert any("izin verilen listede yok" in h for h in hatalar)
    assert any("maksimum 5 olmalıdır" in h for h in hatalar)


def test_guvenli_json_ayristirici_markdown_ve_onarim():
    """GuvenliJsonAyristirici sınıfının markdown etiketlerini ve bozuk JSON sözdizimini onardığını test eder."""
    markdown_json = '```json\n{"name": "TestTool", "arguments": {"x": 10, "y": 20,}}\n```'
    basarili, veri, onarildi, hata = GuvenliJsonAyristirici.ayristir(markdown_json)

    assert basarili is True
    assert veri["name"] == "TestTool"
    assert veri["arguments"]["x"] == 10
    assert veri["arguments"]["y"] == 20


def test_arac_yonlendirici_basarili_cagri():
    """AracYonlendirici sınıfının kayıtlı araçları şemaya uygun başarıyla çalıştırdığını test eder."""
    yonlendirici = AracYonlendirici()
    istek = {
        "name": "HisseSenediSorgula",
        "arguments": {"sembol": "THYAO", "para_birimi": "TRY"},
    }
    yanit = yonlendirici.cagir(istek)

    assert yanit["basarili"] is True
    assert yanit["arac_adi"] == "HisseSenediSorgula"
    assert yanit["sonuc"]["sembol"] == "THYAO"
    assert yanit["sonuc"]["fiyat"] == 312.50


def test_arac_yonlendirici_gecersiz_cagri_ve_hata_yonetimi():
    """AracYonlendirici sınıfının şema hatasında SchemaValidationError döndürdüğünü test eder."""
    yonlendirici = AracYonlendirici()
    hatali_istek = {
        "name": "HisseSenediSorgula",
        "arguments": {"sembol": "GEÇERSİZ_HİSSE"},
    }
    yanit = yonlendirici.cagir(hatali_istek)

    assert yanit["basarili"] is False
    assert yanit["hata_tipi"] == "SchemaValidationError"
    assert "izin verilen listede yok" in yanit["hata_mesaji"]


def test_gorsellestirici_pano():
    """ToolCallingGorsellestirici sınıfının 6 panelli PNG teşhis dosyasını ürettiğini test eder."""
    yonlendirici = AracYonlendirici()
    c1 = yonlendirici.cagir({"name": "HisseSenediSorgula", "arguments": {"sembol": "THYAO"}})
    c2 = yonlendirici.cagir({"name": "UcusRezervasyonuYap", "arguments": {"kalkis": "IST", "varis": "LHR", "yolcu_sayisi": 2}})

    kisitlayici = GramerKisitlayici()
    karsilastirma = kisitlayici.benchmark_karsilastir()

    gorsellestirici = ToolCallingGorsellestirici(dpi=100)
    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_tool_calling_pano.png")
        gorsellestirici.pano_olustur([c1, c2], karsilastirma, kayit_yolu=kayit)
        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
