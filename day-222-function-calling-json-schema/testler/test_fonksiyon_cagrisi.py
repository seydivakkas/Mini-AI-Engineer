"""
PyTest Birim Testleri - Day 222: Katı (Strict) JSON Schema ile Fonksiyon Çağrısı.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.fonksiyon_cagrisi_motoru import (
    StrictSchemaBuilder,
    ToolCallValidator,
    StrictFunctionDispatcher,
)
from src.fonksiyon_profilleyici import FonksiyonProfilleyici
from src.gorsellestirici import StrictFonksiyonGorsellestirici


def ornek_fonksiyon(sehir: str, gun: int) -> str:
    return f"{sehir} için {gun} günlük rapor"


def test_strict_schema_builder_properties():
    """1. StrictSchemaBuilder parametre tiplerini doğru haritalamalıdır."""
    sema = StrictSchemaBuilder.sema_uret(ornek_fonksiyon, "Açıklama")
    props = sema["parameters"]["properties"]
    assert props["sehir"]["type"] == "string"
    assert props["gun"]["type"] == "integer"


def test_strict_schema_builder_flags():
    """2. StrictSchemaBuilder strict=True ve additionalProperties=False bayraklarını koymalıdır."""
    sema = StrictSchemaBuilder.sema_uret(ornek_fonksiyon, "Açıklama")
    assert sema["strict"] is True
    assert sema["parameters"]["additionalProperties"] is False
    assert "sehir" in sema["parameters"]["required"]


def test_validator_valid_json():
    """3. ToolCallValidator geçerli JSON çağrısını onaylamalıdır."""
    sema = StrictSchemaBuilder.sema_uret(ornek_fonksiyon, "Açıklama")
    res = ToolCallValidator.dogrula('{"sehir": "İzmir", "gun": 5}', sema)
    assert res["gecerli_mi"] is True
    assert res["ayristirilmis_veri"]["gun"] == 5


def test_validator_syntax_error():
    """4. ToolCallValidator bozuk JSON sözdizimini yakalamalıdır."""
    sema = StrictSchemaBuilder.sema_uret(ornek_fonksiyon, "Açıklama")
    res = ToolCallValidator.dogrula('{"sehir": "İzmir", "gun": 5', sema)
    assert res["gecerli_mi"] is False
    assert res["hata_tipi"] == "JSON_SYNTAX_ERROR"


def test_validator_missing_required():
    """5. ToolCallValidator eksik zorunlu parametreyi tespit etmelidir."""
    sema = StrictSchemaBuilder.sema_uret(ornek_fonksiyon, "Açıklama")
    res = ToolCallValidator.dogrula('{"sehir": "İzmir"}', sema)
    assert res["gecerli_mi"] is False
    assert res["hata_tipi"] == "MISSING_REQUIRED_FIELD"


def test_validator_additional_properties():
    """6. ToolCallValidator katı şemada izinsiz ek alanı reddetmelidir."""
    sema = StrictSchemaBuilder.sema_uret(ornek_fonksiyon, "Açıklama")
    res = ToolCallValidator.dogrula('{"sehir": "İzmir", "gun": 5, "fazla": true}', sema)
    assert res["gecerli_mi"] is False
    assert res["hata_tipi"] == "ADDITIONAL_PROPERTY_FORBIDDEN"


def test_validator_type_mismatch():
    """7. ToolCallValidator tip uyuşmazlığını reddetmelidir (string yerine int vb)."""
    sema = StrictSchemaBuilder.sema_uret(ornek_fonksiyon, "Açıklama")
    res = ToolCallValidator.dogrula('{"sehir": "İzmir", "gun": "bes"}', sema)
    assert res["gecerli_mi"] is False
    assert res["hata_tipi"] == "TYPE_MISMATCH"


def test_dispatcher_and_gorsellestirme(tmp_path):
    """8. StrictFunctionDispatcher yürütmesi ve görselleştirici panosu başarıyla çalışmalıdır."""
    dagitici = StrictFunctionDispatcher()
    dagitici.kaydet(ornek_fonksiyon, "Hava durumu")

    res = dagitici.calistir("ornek_fonksiyon", '{"sehir": "Bursa", "gun": 3}')
    assert res["basarili"] is True
    assert "Bursa için 3 günlük rapor" in res["sonuc"]

    cikti = str(tmp_path / "test_strict_paneli.png")
    profil = FonksiyonProfilleyici.basarim_profili_cikar()
    StrictFonksiyonGorsellestirici.teshis_paneli_olustur(profil, kayit_yolu=cikti)
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
