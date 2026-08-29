"""
Hata Toleranslı ve Kendini Onaran JSON Ayrıştırıcı Modülü (Day 124 - Faz 7).
LLM çıktılarındaki markdown bloklarını, sözdizim bozukluklarını ve eksik parantezleri onarır.
"""

from typing import Dict, Any, Tuple, Optional
import json
import re


class GuvenliJsonAyristirici:
    """Bozuk veya markdown formatındaki JSON çıktılarını onarıp ayrıştıran motor."""

    @classmethod
    def _temizle_markdown(cls, metin: str) -> str:
        """Markdown kod çitlerini (```json ... ```) temizler."""
        eslesme = re.search(r"```(?:json)?\s*(.*?)\s*```", metin, re.DOTALL | re.IGNORECASE)
        if eslesme:
            return eslesme.group(1).strip()
        return metin.strip()

    @classmethod
    def _onar(cls, json_metni: str) -> str:
        """Sık rastlanan LLM JSON hatalarını (trailing comma, tek tırnak, unclosed bracket) onarır."""
        onari = json_metni

        # 1. Tek tırnakları çift tırnağa çevir (Python dict string formatı)
        # Sadece anahtar ve değer çevrelerindeki tek tırnaklar
        onari = re.sub(r"(?<=[\{\[\,\:])\s*'([^']+)'\s*(?=[\}\]\,\:])", r'"\1"', onari)
        if "'" in onari and '"' not in onari:
            onari = onari.replace("'", '"')

        # 2. Trailing Comma (Sondaki fazlalık virgüller: [1, 2,] veya {"a": 1,})
        onari = re.sub(r",\s*([\}\]])", r"\1", onari)

        # 3. Anahtarlar tırnaksız ise tırnak ekle ({name: "x"} -> {"name": "x"})
        onari = re.sub(r'([{,]\s*)([a-zA-Z0-9_]+)\s*:', r'\1"\2":', onari)

        # 4. Eksik Kapanış Parantezleri
        acik_suslu = onari.count("{") - onari.count("}")
        acik_kose = onari.count("[") - onari.count("]")

        if acik_suslu > 0:
            onari += "}" * acik_suslu
        if acik_kose > 0:
            onari += "]" * acik_kose

        return onari

    @classmethod
    def ayristir(cls, ham_metin: str) -> Tuple[bool, Optional[Dict[str, Any]], bool, Optional[str]]:
        """
        Ham metinden JSON nesnesini çıkarır ve onarır.
        Dönüş: (basarili: bool, veri: dict, onarildi_mi: bool, hata: str)
        """
        temiz = cls._temizle_markdown(ham_metin)

        # 1. Doğrudan Standart Ayrıştırma Denemesi
        try:
            veri = json.loads(temiz)
            return True, veri, False, None
        except Exception:
            pass

        # 2. Otomatik Onarım ile Ayrıştırma
        onarmis_metin = cls._onar(temiz)
        try:
            veri = json.loads(onarmis_metin)
            return True, veri, True, None
        except Exception as e:
            return False, None, False, f"JSON Ayrıştırma Hatası: {str(e)} (Ham: {temiz[:50]}...)"
