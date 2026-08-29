"""
Length-Bias Cezalandırma ve Over-Thinking Önleme Motoru (Day 214 - FAZ 11).
Lineer, Menteşe (Hinge) ve Uzunluk-Normalize Edilmiş DPO/GRPO Düzenlileştirmesi.
"""

from typing import Dict, Any, List, Optional, Tuple
import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F


class LengthPenaltyObjective:
    """Uzunluk Cezalandırma ve Düzenlileştirme Amaç Fonksiyonları."""

    @classmethod
    def lineer_odul(
        cls,
        ham_odul: float,
        uzunluk: int,
        alpha: float = 0.0005,
    ) -> float:
        """R_lin = R_ham - α * Uzunluk"""
        return float(ham_odul - alpha * uzunluk)

    @classmethod
    def hinge_odul(
        cls,
        ham_odul: float,
        uzunluk: int,
        hedef_uzunluk: int = 500,
        beta: float = 0.002,
    ) -> float:
        """R_hinge = R_ham - β * max(0, Uzunluk - Hedef_Uzunluk)"""
        fazlalik = max(0, uzunluk - hedef_uzunluk)
        return float(ham_odul - beta * fazlalik)

    @classmethod
    def uzunluk_normalize_dpo_kaybi(
        cls,
        logp_pi_w: torch.Tensor,
        logp_ref_w: torch.Tensor,
        logp_pi_l: torch.Tensor,
        logp_ref_l: torch.Tensor,
        len_w: int,
        len_l: int,
        beta: float = 0.1,
    ) -> torch.Tensor:
        """
        Uzunluk-Normalize DPO Kaybı:
        L = -log σ( (β/|y_w|) log(π/π_ref)(y_w) - (β/|y_l|) log(π/π_ref)(y_l) )
        """
        oran_w = (logp_pi_w - logp_ref_w) / max(1, len_w)
        oran_l = (logp_pi_l - logp_ref_l) / max(1, len_l)
        logit = beta * (oran_w - oran_l)
        return -F.logsigmoid(logit)


class OverthinkingDetector:
    """Boş Düşünce Şişmesi (Over-Thinking) ve Döngüsel Gevezelik Tespiti."""

    TEKRAR_KALIPLARI = [
        r"dur bir dakika",
        r"tekrar kontrol edeyim",
        r"baştan hesaplayalım",
        r"emin olmak için bir daha",
        r"let me rethink",
        r"wait let me check",
    ]

    @classmethod
    def analiz_et(cls, metin: str) -> Dict[str, Any]:
        """Metindeki gereksiz düşünce tekrarlarını ve döngülerini ölçer."""
        kucuk_metin = metin.lower()
        tekrar_sayisi = 0
        for kalip in cls.TEKRAR_KALIPLARI:
            bulunanlar = re.findall(kalip, kucuk_metin)
            tekrar_sayisi += len(bulunanlar)

        # Cümle benzerlikleri ve döngü oranı
        cumleler = [c.strip() for c in metin.split(".") if len(c.strip()) > 5]
        benzersiz_cumleler = set(cumleler)
        tekrar_orani = 1.0 - (len(benzersiz_cumleler) / max(1, len(cumleler)))

        # Gevezelik skoru (0.0 temiz, 1.0 aşırı over-thinking)
        gevezelik_skoru = min(1.0, (tekrar_sayisi * 0.25) + (tekrar_orani * 0.50))
        overthinking_var_mi = gevezelik_skoru > 0.40

        return {
            "tekrar_sayisi": tekrar_sayisi,
            "tekrar_orani": float(tekrar_orani),
            "gevezelik_skoru": float(gevezelik_skoru),
            "overthinking_var_mi": overthinking_var_mi,
        }


class AdaptiveLengthController:
    """Problem Karmaşıklığına Göre Dinamik Token Bütçesi Belirleyici."""

    @classmethod
    def hedef_butce_belirle(cls, soru: str) -> int:
        """Sorunun türüne ve zorluğuna göre ideal düşünce uzunluğu belirler."""
        kelime_sayisi = len(soru.split())
        if kelime_sayisi < 10 and any(op in soru for op in ["+", "-", "*", "/"]):
            return 250  # Basit Aritmetik
        elif "ispatlayın" in soru.lower() or "teorem" in soru.lower():
            return 1200  # Karmaşık İspat
        else:
            return 500  # Standart Akıl Yürütme


class LengthRegularizedTrainer:
    """Uzunluk Düzenlileştirmeli Politika Eğiticisi."""

    @classmethod
    def degerlendir(
        cls,
        soru: str,
        yanit: str,
        dogru_mu: bool = True,
    ) -> Dict[str, Any]:
        """Yanıtı doğruluk, uzunluk ve verimlilik açısından puanlar."""
        uzunluk = len(yanit.split())
        hedef = AdaptiveLengthController.hedef_butce_belirle(soru)
        overthinking = OverthinkingDetector.analiz_et(yanit)

        ham_r = 1.0 if dogru_mu else 0.0
        hinge_r = LengthPenaltyObjective.hinge_odul(ham_r, uzunluk, hedef, beta=0.002)

        # Token Verimliliği Metriği (Doğruluk / Harcanan Token)
        verimlilik = (ham_r * 100.0) / max(1, uzunluk)

        return {
            "uzunluk": uzunluk,
            "hedef_butce": hedef,
            "ham_odul": ham_r,
            "duzenlenmis_odul": hinge_r,
            "overthinking": overthinking,
            "verimlilik_skoru": float(verimlilik),
        }
