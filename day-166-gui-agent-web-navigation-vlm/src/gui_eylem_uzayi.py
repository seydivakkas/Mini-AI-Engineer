"""
GUI Eylem Uzayı (GUI Action Space) Modülü (Day 166 - FAZ 9).
click(x, y), type(text), scroll(direction), press_key(key) ve terminate eylemlerini yönetir.
"""

from typing import Dict, Any, Optional
import re


class GUIEylemUzayi:
    """GUI Ajanı Eylem Sözlüğü ve Ayrıştırıcı."""

    @classmethod
    def eylem_ayristir(cls, eylem_metni: str) -> Dict[str, Any]:
        """
        Örnekler:
          - 'click(345, 500)'
          - 'type("Python 3.14 VLM")'
          - 'press_key("Enter")'
          - 'scroll("down")'
          - 'terminate("SUCCESS")'
        """
        eylem_metni = eylem_metni.strip()

        # 1. Click Eylemi
        click_match = re.match(r"click\(\s*(\d+)\s*,\s*(\d+)\s*\)", eylem_metni)
        if click_match:
            y, x = map(int, click_match.groups())
            return {"tur": "click", "y": y, "x": x, "gecerli_mi": True}

        # 2. Type Eylemi
        type_match = re.match(r"type\(\s*[\"\'](.*)[\"\']\s*\)", eylem_metni)
        if type_match:
            metin = type_match.group(1)
            return {"tur": "type", "metin": metin, "gecerli_mi": True}

        # 3. Press Key Eylemi
        key_match = re.match(r"press_key\(\s*[\"\'](.*)[\"\']\s*\)", eylem_metni)
        if key_match:
            tus = key_match.group(1)
            return {"tur": "press_key", "tus": tus, "gecerli_mi": True}

        # 4. Scroll Eylemi
        scroll_match = re.match(r"scroll\(\s*[\"\'](.*)[\"\']\s*\)", eylem_metni)
        if scroll_match:
            yon = scroll_match.group(1)
            return {"tur": "scroll", "yon": yon, "gecerli_mi": True}

        # 5. Terminate Eylemi
        term_match = re.match(r"terminate\(\s*[\"\'](.*)[\"\']\s*\)", eylem_metni)
        if term_match:
            durum = term_match.group(1)
            return {"tur": "terminate", "durum": durum, "gecerli_mi": True}

        return {"tur": "bilinmeyen", "ham_metin": eylem_metni, "gecerli_mi": False}
