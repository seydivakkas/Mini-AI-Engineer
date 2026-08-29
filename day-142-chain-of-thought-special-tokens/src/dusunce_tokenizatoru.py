"""
Düşünce Tokenizatörü Modülü (Day 142 - Faz 8).
<think> ve </think> özel akıl yürütme tokenlerini yöneten, metinleri ayrıştıran ve maskeleyen motor.
"""

from typing import Dict, Any, List, Tuple
import re
import torch


class DusunceTokenizatoru:
    """Özel düşünce tokenleri (<think>, </think>, <step>, </step>) ile akıl yürütme tokenizasyonu yapar."""

    THINK_START = "<think>"
    THINK_END = "</think>"
    STEP_START = "<step>"
    STEP_END = "</step>"

    OZEL_TOKENLER = {
        THINK_START: 32000,
        THINK_END: 32001,
        STEP_START: 32002,
        STEP_END: 32003,
    }

    def __init__(self, sozluk_boyutu: int = 32004):
        self.sozluk_boyutu = sozluk_boyutu
        self.id_to_token = {v: k for k, v in self.OZEL_TOKENLER.items()}

    def ayristir(self, tam_metin: str) -> Dict[str, Any]:
        """
        Metni <think> düşünce bloğu ve nihai yanıt bloğu olarak ikiye ayırır.
        """
        think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        match = think_pattern.search(tam_metin)

        if match:
            dusunce_kismi = match.group(1).strip()
            # think bloğunun sonrasındaki metin nihai yanıttır
            nihai_yanit = tam_metin[match.end():].strip()
        else:
            dusunce_kismi = ""
            nihai_yanit = tam_metin.strip()

        # Step bloklarını çıkar
        adimlar = re.findall(r"<step>(.*?)</step>", dusunce_kismi, re.DOTALL)
        if not adimlar and dusunce_kismi:
            adimlar = [a.strip() for a in dusunce_kismi.split("\n") if a.strip()]

        return {
            "dusunce_metni": dusunce_kismi,
            "adimlar": [a.strip() for a in adimlar],
            "nihai_yanit": nihai_yanit,
            "dusunce_mevcut_mu": bool(dusunce_kismi),
        }

    def birlestir(self, dusunce_metni: str, nihai_yanit: str) -> str:
        """Düşünce metni ve nihai yanıtı <think> etiketleriyle standart formata sarar."""
        return f"{self.THINK_START}\n{dusunce_metni.strip()}\n{self.THINK_END}\n{nihai_yanit.strip()}"

    def token_dagilimi_hesapla(self, prompt: str, dusunce_metni: str, nihai_yanit: str) -> Dict[str, int]:
        """Prompt, Düşünce (<think>) ve Nihai Yanıt token sayılarını hesaplar."""
        prompt_tokens = len(prompt.split())
        think_tokens = len(dusunce_metni.split()) + 2  # <think> ve </think> tokenleri dahil
        output_tokens = len(nihai_yanit.split())
        toplam = prompt_tokens + think_tokens + output_tokens

        return {
            "prompt_token_sayisi": prompt_tokens,
            "dusunce_token_sayisi": think_tokens,
            "nihai_yanit_token_sayisi": output_tokens,
            "toplam_token_sayisi": toplam,
        }
