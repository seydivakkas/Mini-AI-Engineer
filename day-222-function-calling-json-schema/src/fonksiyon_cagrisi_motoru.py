"""
Katı (Strict) JSON Schema ile Fonksiyon Çağrısı ve Tip Doğrulama Motoru (Day 222 - FAZ 12).
Gramer Kısıtlamalı Çıkarım (Constrained Decoding) ve Deterministik Araç Yürütme.
"""

from typing import Dict, Any, List, Optional, Callable, Union
import json
import inspect


class StrictSchemaBuilder:
    """Python Fonksiyonlarından Katı (Strict) JSON Şeması Üretici."""

    TIP_ESLESTIRME = {
        int: "integer",
        float: "number",
        str: "string",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    @classmethod
    def sema_uret(
        cls,
        fonksiyon: Callable[..., Any],
        aciklama: str,
        zorunlu_alanlar: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Fonksiyondan katı (strict) JSON schema üretir."""
        imza = inspect.signature(fonksiyon)
        ozellikler = {}
        tum_parametreler = []

        for isim, param in imza.parameters.items():
            tum_parametreler.append(isim)
            tip_adi = "string"
            if param.annotation in cls.TIP_ESLESTIRME:
                tip_adi = cls.TIP_ESLESTIRME[param.annotation]

            ozellikler[isim] = {
                "type": tip_adi,
                "description": f"{isim} parametresi",
            }

        required = zorunlu_alanlar if zorunlu_alanlar is not None else tum_parametreler

        return {
            "name": fonksiyon.__name__,
            "description": aciklama,
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": ozellikler,
                "required": required,
                "additionalProperties": False,
            },
        }


class ToolCallValidator:
    """Çalışma Zamanı Katı Şema Doğrulayıcısı."""

    @classmethod
    def dogrula(
        cls,
        cagri_json_metni: str,
        hedef_sema: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Gelen JSON metnini şemaya göre katı denetimden geçirir."""
        # 1. JSON Sözdizimi Kontrolü
        try:
            veri = json.loads(cagri_json_metni)
        except Exception as e:
            return {
                "gecerli_mi": False,
                "hata_tipi": "JSON_SYNTAX_ERROR",
                "hata_mesaji": f"Geçersiz JSON sözdizimi: {str(e)}",
                "ayristirilmis_veri": None,
            }

        if not isinstance(veri, dict):
            return {
                "gecerli_mi": False,
                "hata_tipi": "INVALID_ROOT_TYPE",
                "hata_mesaji": "Kök nesne JSON object (dict) olmalıdır.",
                "ayristirilmis_veri": None,
            }

        parametre_semasi = hedef_sema.get("parameters", {})
        ozellikler = parametre_semasi.get("properties", {})
        zorunlular = parametre_semasi.get("required", [])
        ek_ozellik_yasak = not parametre_semasi.get("additionalProperties", True)

        # 2. Zorunlu Alan Kontrolü
        for z in zorunlular:
            if z not in veri:
                return {
                    "gecerli_mi": False,
                    "hata_tipi": "MISSING_REQUIRED_FIELD",
                    "hata_mesaji": f"Zorunlu alan eksik: '{z}'",
                    "ayristirilmis_veri": None,
                }

        # 3. İstenmeyen Fazlalık Alan Kontrolü (Strict: additionalProperties=False)
        if ek_ozellik_yasak:
            for k in veri.keys():
                if k not in ozellikler:
                    return {
                        "gecerli_mi": False,
                        "hata_tipi": "ADDITIONAL_PROPERTY_FORBIDDEN",
                        "hata_mesaji": f"Katı şemada tanımsız ek alan yasaktır: '{k}'",
                        "ayristirilmis_veri": None,
                    }

        # 4. Tip Kontrolleri
        for k, v in veri.items():
            beklenen_tip = ozellikler.get(k, {}).get("type")
            if beklenen_tip == "integer" and not (isinstance(v, int) and not isinstance(v, bool)):
                return {
                    "gecerli_mi": False,
                    "hata_tipi": "TYPE_MISMATCH",
                    "hata_mesaji": f"'{k}' alanı 'integer' bekliyor ancak '{type(v).__name__}' geldi.",
                    "ayristirilmis_veri": None,
                }
            elif beklenen_tip == "number" and not isinstance(v, (int, float)):
                return {
                    "gecerli_mi": False,
                    "hata_tipi": "TYPE_MISMATCH",
                    "hata_mesaji": f"'{k}' alanı 'number' bekliyor.",
                    "ayristirilmis_veri": None,
                }
            elif beklenen_tip == "string" and not isinstance(v, str):
                return {
                    "gecerli_mi": False,
                    "hata_tipi": "TYPE_MISMATCH",
                    "hata_mesaji": f"'{k}' alanı 'string' bekliyor.",
                    "ayristirilmis_veri": None,
                }

        return {
            "gecerli_mi": True,
            "hata_tipi": None,
            "hata_mesaji": None,
            "ayristirilmis_veri": veri,
        }


class StrictFunctionDispatcher:
    """Katı Doğrulanmış Fonksiyon Yürütücüsü."""

    def __init__(self):
        self._fonksiyonlar: Dict[str, Callable[..., Any]] = {}
        self._semalar: Dict[str, Dict[str, Any]] = {}

    def kaydet(self, fonksiyon: Callable[..., Any], aciklama: str) -> None:
        """Fonksiyonu ve katı şemasını kaydeder."""
        isim = fonksiyon.__name__
        sema = StrictSchemaBuilder.sema_uret(fonksiyon, aciklama)
        self._fonksiyonlar[isim] = fonksiyon
        self._semalar[isim] = sema

    def semayi_al(self, isim: str) -> Dict[str, Any]:
        """Kayıtlı şemayı döner."""
        return self._semalar[isim]

    def calistir(self, fonksiyon_adi: str, cagri_json_metni: str) -> Dict[str, Any]:
        """Katı doğrulama yapıp fonksiyonu yürütür."""
        if fonksiyon_adi not in self._fonksiyonlar:
            return {"basarili": False, "hata": f"Kayıtsız fonksiyon: {fonksiyon_adi}"}

        sema = self._semalar[fonksiyon_adi]
        dogrulama = ToolCallValidator.dogrula(cagri_json_metni, sema)

        if not dogrulama["gecerli_mi"]:
            return {
                "basarili": False,
                "hata_tipi": dogrulama["hata_tipi"],
                "hata": dogrulama["hata_mesaji"],
            }

        parametreler = dogrulama["ayristirilmis_veri"]
        try:
            cikti = self._fonksiyonlar[fonksiyon_adi](**parametreler)
            return {
                "basarili": True,
                "sonuc": cikti,
                "kullanilan_parametreler": parametreler,
            }
        except Exception as e:
            return {"basarili": False, "hata": str(e)}
