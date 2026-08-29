"""
ReAct Çıktı Ayrıştırıcı Modülü (Day 121 - Faz 7).
Dil modelinin ürettiği Thought, Action ve Final Answer bloklarını düzenli ifadeler (Regex) ve kural tabanlı ayrıştırır.
"""

from typing import Dict, Any, Optional, Tuple
import re


class ReActAyristirici:
    """ReAct formatındaki (Thought -> Action[Input] | Final Answer) metinleri ayrıştırır."""

    THOUGHT_REGEX = re.compile(r"Thought:\s*(.*?)(?=\nAction:|\nFinal Answer:|$)", re.DOTALL | re.IGNORECASE)
    ACTION_REGEX = re.compile(r"Action:\s*([A-Za-z0-9_]+)\[(.*?)\]", re.DOTALL | re.IGNORECASE)
    ACTION_ALT_REGEX = re.compile(r"Action:\s*([A-Za-z0-9_]+)\((.*?)\)", re.DOTALL | re.IGNORECASE)
    ACTION_COLON_REGEX = re.compile(r"Action:\s*([A-Za-z0-9_]+):\s*(.*)", re.IGNORECASE)
    FINAL_ANSWER_REGEX = re.compile(r"Final Answer:\s*(.*)", re.DOTALL | re.IGNORECASE)

    @classmethod
    def ayristir(cls, metin: str) -> Dict[str, Any]:
        """
        Model çıktısını analiz eder ve yapılandırılmış sözlük döndürür.
        """
        sonuc: Dict[str, Any] = {
            "tip": "belirsiz",
            "dusunce": "",
            "arac_adi": None,
            "arac_girdisi": None,
            "nihai_yanit": None,
            "ham_metin": metin,
        }

        # 1. Düşünce (Thought) Çıkarımı
        thought_match = cls.THOUGHT_REGEX.search(metin)
        if thought_match:
            sonuc["dusunce"] = thought_match.group(1).strip()

        # 2. Nihai Yanıt (Final Answer) Kontrolü
        final_match = cls.FINAL_ANSWER_REGEX.search(metin)
        if final_match:
            sonuc["tip"] = "final_answer"
            sonuc["nihai_yanit"] = final_match.group(1).strip()
            return sonuc

        # 3. Eylem (Action) Çıkarımı: Action[Input] veya Action(Input) veya Action: Input
        action_match = cls.ACTION_REGEX.search(metin)
        if not action_match:
            action_match = cls.ACTION_ALT_REGEX.search(metin)

        if action_match:
            sonuc["tip"] = "action"
            sonuc["arac_adi"] = action_match.group(1).strip()
            sonuc["arac_girdisi"] = action_match.group(2).strip()
            return sonuc

        colon_match = cls.ACTION_COLON_REGEX.search(metin)
        if colon_match:
            sonuc["tip"] = "action"
            sonuc["arac_adi"] = colon_match.group(1).strip()
            sonuc["arac_girdisi"] = colon_match.group(2).strip()
            return sonuc

        # Format dışı durum
        if sonuc["dusunce"]:
            sonuc["tip"] = "thought_only"
        else:
            sonuc["tip"] = "gecersiz_format"

        return sonuc
