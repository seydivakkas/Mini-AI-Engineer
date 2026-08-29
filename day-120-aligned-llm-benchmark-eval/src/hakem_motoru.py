"""
LLM-as-a-Judge Hakemlik ve Yanlılık Giderme Modülü (Day 120 - Faz 6 Capstone).
MT-Bench tekli puanlama, AlpacaEval çiftli karşılaştırma ve Pozisyon Yanlılığı (Position Bias) Swap Testi.
"""

from typing import Dict, Any, List, Tuple
import random


class LLMHakemMotoru:
    """Modellerin yanıtlarını objektif rubrikler ve Swap denetimi ile puanlayan hakem motoru."""

    KATEGORILER = [
        "Kodlama (Coding)",
        "Matematik (Math)",
        "Akıl Yürütme (Reasoning)",
        "Güvenlik & Guardrails",
        "Yaratıcılık & Yazım",
        "Rol Yapma (Roleplay)",
        "Bilgi Çıkarımı (Extraction)",
        "Metin Özetleme (Summarization)",
    ]

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def tekli_puanla(self, soru: str, yanit: str, kategori: str = "Genel") -> Dict[str, Any]:
        """MT-Bench tarzı tekli yanıt puanlama (1-10 puan rubriği)."""
        uzunluk = len(yanit.split())
        icerik_skor = 0.0

        # Rubrik Değerlendirmesi
        if "```" in yanit or "def " in yanit or "O(1)" in yanit:
            icerik_skor += 3.5
        if "<think>" in yanit or "Adım 1" in yanit or "İspat" in yanit:
            icerik_skor += 3.0
        if "Sonuç" in yanit or "Özet" in yanit:
            icerik_skor += 1.5

        taban_puan = min(10.0, 2.0 + icerik_skor + min(2.0, uzunluk / 25.0))
        return {
            "puan": float(taban_puan),
            "kategori": kategori,
            "aciklama": f"Yanıt {kategori} kategorisinde rubrik kriterlerine göre {taban_puan:.1f}/10 puan aldı.",
        }

    def ciftli_karsilastir_tek_yon(
        self,
        soru: str,
        yanit_a: str,
        yanit_b: str,
    ) -> int:
        """
        Model A vs Model B karşılaştırması:
        Dönüş: +1 (A Kazandı), -1 (B Kazandı), 0 (Beraberlik)
        """
        puan_a = self.tekli_puanla(soru, yanit_a)["puan"]
        puan_b = self.tekli_puanla(soru, yanit_b)["puan"]

        fark = puan_a - puan_b
        if abs(fark) < 0.3:
            return 0  # Beraberlik
        return 1 if fark > 0 else -1

    def swap_testli_karsilastir(
        self,
        soru: str,
        yanit_a: str,
        yanit_b: str,
    ) -> Tuple[int, bool]:
        """
        Pozisyon Yanlılığı (Position Bias) tespiti için [A, B] ve [B, A] sıralamasını test eder.
        Dönüş: (nihai_sonuc: int, pozisyon_yanliligi_var_mi: bool)
        """
        # 1. Yön: A solda, B sağda
        karar_1 = self.ciftli_karsilastir_tek_yon(soru, yanit_a, yanit_b)

        # 2. Yön: B solda, A sağda (Swap Test)
        # B'ye göre karar: 1 ise B kazandı, -1 ise A kazandı
        karar_2_b_ye_gore = self.ciftli_karsilastir_tek_yon(soru, yanit_b, yanit_a)
        karar_2 = -karar_2_b_ye_gore  # A'nın perspektifine çevir

        # Eğer iki yönde çelişkili karar verildiyse (Pozisyon Yanlılığı)
        if karar_1 != karar_2:
            return 0, True  # Yanlılık tespit edildi -> Güvenli Beraberlik

        return karar_1, False
