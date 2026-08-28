"""
Konfigurasyon Yoneticisi ve YAML Ayristirici (Config Manager)
============================================================
YAML dosyalarini yukler, Pydantic semasiyla dogrular ve
komut satiri / calisma zamani gecersiz kilma (Override) parametrelerini uygular.
"""

from typing import Dict, Any, List, Optional
import os
import yaml
from src.konfigurasyon_semasi import KokKonfigurasyon


class KonfigurasyonYoneticisi:
    """
    YAML ve Pydantic v2 tabanli tip guvenli konfigurasyon orkestratoru.
    """

    @classmethod
    def yaml_yukle(
        cls,
        yaml_yolu: str,
        override_listesi: Optional[List[str]] = None
    ) -> KokKonfigurasyon:
        """
        YAML dosyasini yukler, override'lari uygular ve Pydantic nesnesi dondurur.

        Args:
            yaml_yolu: Konfigurasyon YAML dosyasinin mutlak veya goreli yolu
            override_listesi: 'grup.anahtar=deger' formatinda parametre ezme listesi
        """
        if not os.path.exists(yaml_yolu):
            raise FileNotFoundError(f"Konfigurasyon dosyasi bulunamadi: {yaml_yolu}")

        with open(yaml_yolu, "r", encoding="utf-8") as f:
            ham_veri = yaml.safe_load(f) or {}

        if override_listesi:
            ham_veri = cls._overridelari_uygula(ham_veri, override_listesi)

        return KokKonfigurasyon.model_validate(ham_veri)

    @classmethod
    def _overridelari_uygula(
        cls,
        veri_sozlugu: Dict[str, Any],
        override_listesi: List[str]
    ) -> Dict[str, Any]:
        """
        'egitim.tohum=123' gibi noktasal override ifadelerini ic ice sozluklere isler.
        """
        guncel_sozluk = dict(veri_sozlugu)

        for override in override_listesi:
            if "=" not in override:
                raise ValueError(f"Gecersiz override formati (Beklenen: anahtar.yol=deger): {override}")

            anahtar_yolu, ham_deger = override.split("=", 1)
            parcalar = anahtar_yolu.strip().split(".")

            # Deger tipini otomatik donustur (int, float, bool, str)
            deger = cls._tip_donustur(ham_deger.strip())

            # Ic ice sozlukte ilerle
            su_anki = guncel_sozluk
            for parca in parcalar[:-1]:
                if parca not in su_anki or not isinstance(su_anki[parca], dict):
                    su_anki[parca] = {}
                su_anki = su_anki[parca]

            su_anki[parcalar[-1]] = deger

        return guncel_sozluk

    @staticmethod
    def _tip_donustur(deger_str: str) -> Any:
        """String degeri uygun Python tipine (bool, int, float, str) otomatik donusturur."""
        if deger_str.lower() in ("true", "yes", "on"):
            return True
        if deger_str.lower() in ("false", "no", "off"):
            return False
        try:
            return int(deger_str)
        except ValueError:
            pass
        try:
            return float(deger_str)
        except ValueError:
            pass
        # Liste kontrolu: "[3, 32, 32]"
        if deger_str.startswith("[") and deger_str.endswith("]"):
            try:
                return yaml.safe_load(deger_str)
            except Exception:
                pass
        return deger_str

    @classmethod
    def yaml_kaydet(cls, config: KokKonfigurasyon, hedef_yol: str) -> str:
        """Pydantic modelini temiz bir YAML dosyasina yazar."""
        os.makedirs(os.path.dirname(os.path.abspath(hedef_yol)), exist_ok=True)
        veri = config.model_dump()
        with open(hedef_yol, "w", encoding="utf-8") as f:
            yaml.dump(veri, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        return hedef_yol
