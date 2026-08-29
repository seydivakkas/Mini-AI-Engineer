"""
Ses Komutlu Robot Ajanı (Voice-Conditioned Robotic Agent) Motoru (Day 256).
Whisper ASR, VLM Semantik ve Mekansal Temellendirme (Grounding) ve VLA Eylem İcrası.
"""

from typing import Dict, Any, List, Tuple
import numpy as np


class WhisperSemanticParser:
    """Whisper ASR ve Doğal Dil Görev Ayrıştırıcı (Task Decomposition)."""

    @classmethod
    def parse_command(cls, raw_text: str) -> Dict[str, Any]:
        """Doğal dil sesli komutunu analiz eder, niyet ve eylem sırasını çıkarır."""
        text_lower = raw_text.lower().strip()

        # Belirsizlik Kontrolü (Ambiguity Detection)
        if "bardağı" in text_lower and "hangi" not in text_lower and "kırmızı" not in text_lower and "mavi" not in text_lower:
            return {
                "orijinal_metin": raw_text,
                "niyet": "BELIRSIZ_TALEP",
                "belirsizlik_var_mi": True,
                "netlestirme_sorusu": "Masada birden fazla bardak var. Kırmızı kupayı mı yoksa mavi bardağı mı kastediyorsunuz?",
                "alt_gorevler": [],
            }

        # Görev Ayrıştırma (Task Decomposition)
        alt_gorevler = []
        hedef_nesne = "bilinmeyen"
        hedef_konum = "bilinmeyen"

        if "kırmızı" in text_lower or "kupa" in text_lower:
            hedef_nesne = "kırmızı kupa"
        elif "mavi" in text_lower or "bardak" in text_lower:
            hedef_nesne = "mavi bardak"

        if "ısıtıcı" in text_lower:
            hedef_konum = "su ısıtıcısı"
        elif "tepsi" in text_lower:
            hedef_konum = "servis tepsisi"
        else:
            hedef_konum = "hedef masa"

        alt_gorevler = [
            {"adim": 1, "eylem": "NESNE_TESPITI", "hedef": hedef_nesne},
            {"adim": 2, "eylem": "YAKLASMA_VE_KAVRAMA", "hedef": hedef_nesne},
            {"adim": 3, "eylem": "KALDIRMA_VE_TASIMA", "hedef": hedef_nesne},
            {"adim": 4, "eylem": "HEDEF_TESPITI", "hedef": hedef_konum},
            {"adim": 5, "eylem": "GUVENLI_BIRAKMA", "hedef": hedef_konum},
        ]

        return {
            "orijinal_metin": raw_text,
            "niyet": "AL_VE_TASI",
            "belirsizlik_var_mi": False,
            "hedef_nesne": hedef_nesne,
            "hedef_konum": hedef_konum,
            "alt_gorevler": alt_gorevler,
        }


class VisualSpatialGrounder:
    """VLM Görsel ve Mekansal Temellendirme (Spatial Grounding) Motoru."""

    # Sahne Nesneleri Bilgi Tabanı (3D Koordinatlar ve Bounding Box)
    SAHNE_NESNELERI = {
        "kırmızı kupa": {"kutu_2d": [120, 180, 210, 290], "3d_merkez": [0.45, 0.15, 0.82], "guven_skoru": 0.98},
        "mavi bardak": {"kutu_2d": [340, 200, 420, 310], "3d_merkez": [0.55, -0.22, 0.82], "guven_skoru": 0.96},
        "su ısıtıcısı": {"kutu_2d": [500, 150, 620, 350], "3d_merkez": [0.70, -0.18, 0.88], "guven_skoru": 0.99},
        "servis tepsisi": {"kutu_2d": [200, 380, 400, 450], "3d_merkez": [0.35, 0.00, 0.78], "guven_skoru": 0.95},
    }

    @classmethod
    def ground_object(cls, nesne_adi: str) -> Dict[str, Any]:
        """Nesne ismini sahnedeki 3D fiziksel koordinatlara bağlar."""
        nesne_key = nesne_adi.lower().strip()
        if nesne_key in cls.SAHNE_NESNELERI:
            veri = cls.SAHNE_NESNELERI[nesne_key]
            return {
                "bulundu": True,
                "nesne_adi": nesne_adi,
                "kutu_2d": veri["kutu_2d"],
                "koordinat_3d_m": veri["3d_merkez"],
                "vlm_guven_skoru": veri["guven_skoru"],
            }
        return {
            "bulundu": False,
            "nesne_adi": nesne_adi,
            "kutu_2d": None,
            "koordinat_3d_m": [0.0, 0.0, 0.0],
            "vlm_guven_skoru": 0.0,
        }


class VoiceConditionedVLAAgent:
    """Uçtan Uca Ses Koşullu Vision-Language-Action (VLA) Robot Ajanı."""

    def __init__(self):
        self.durum = "BEKLEMEDE"  # IDLE
        self.maks_hiz_m_s = 0.25

    def process_voice_instruction(self, voice_text: str) -> Dict[str, Any]:
        """Ses metnini alır, semantik ayrıştırır, mekansal temellendirir ve VLA yörüngesi üretir."""
        # 1. Whisper ASR & Semantik Ayrıştırma
        self.durum = "SES_AYRISTIRILIYOR"
        parse_res = WhisperSemanticParser.parse_command(voice_text)

        if parse_res["belirsizlik_var_mi"]:
            self.durum = "NETLESTIRME_BEKLENIYOR"
            return {
                "durum": self.durum,
                "basarili": False,
                "sesli_geri_bildirim": parse_res["netlestirme_sorusu"],
                "parse_sonucu": parse_res,
                "icra_plani": [],
            }

        # 2. VLM Mekansal Temellendirme
        self.durum = "MEKANSAL_TEMELLENDIRME"
        nesne_ground = VisualSpatialGrounder.ground_object(parse_res["hedef_nesne"])
        hedef_ground = VisualSpatialGrounder.ground_object(parse_res["hedef_konum"])

        if not nesne_ground["bulundu"] or not hedef_ground["bulundu"]:
            self.durum = "HATA_NESNE_BULUNAMADI"
            return {
                "durum": self.durum,
                "basarili": False,
                "sesli_geri_bildirim": "Belirtilen nesnelerden biri kamera görüş alanında tespit edilemedi.",
                "nesne_grounding": nesne_ground,
                "hedef_grounding": hedef_ground,
            }

        # 3. VLA Eylem Planı ve Yörünge İcrası
        self.durum = "ICRA_EDILIYOR"
        icra_plani = []
        for gorev in parse_res["alt_gorevler"]:
            adim_no = gorev["adim"]
            eylem = gorev["eylem"]
            if "KAVRAMA" in eylem or "NESNE" in eylem:
                hedef_xyz = nesne_ground["koordinat_3d_m"]
            else:
                hedef_xyz = hedef_ground["koordinat_3d_m"]

            icra_plani.append({
                "adim": adim_no,
                "eylem": eylem,
                "hedef_koordinat": hedef_xyz,
                "guvenlik_hiz_limiti_m_s": self.maks_hiz_m_s,
                "durum": "TAMAMLANDI",
            })

        self.durum = "GOREV_TAMAMLANDI"
        sesli_cevap = f"{parse_res['hedef_nesne']} başarıyla {parse_res['hedef_konum']} konumuna taşındı."

        return {
            "durum": self.durum,
            "basarili": True,
            "sesli_geri_bildirim": sesli_cevap,
            "parse_sonucu": parse_res,
            "nesne_grounding": nesne_ground,
            "hedef_grounding": hedef_ground,
            "icra_plani": icra_plani,
        }
