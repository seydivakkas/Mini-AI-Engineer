"""
MathVista Görsel Matematiksel Akıl Yürütme Değerlendirici Modülü (Day 180 - FAZ 9 FİNALİ).
Lu et al. (2023) MathVista: Evaluating Mathematical Reasoning in Visual Contexts.
Geometri, fonksiyon grafikleri, istatistiksel tablolar ve çok adımlı görsel problem çözme testi.
"""

from typing import Dict, Any, List
import numpy as np


class MathVistaDegerlendirici:
    """MathVista Görsel Matematik ve Mantıksal Çıkarım Değerlendirme Motoru."""

    MATEMATIK_ALANLARI = [
        "geometry_reasoning",
        "function_plots_and_calculus",
        "statistical_data_charts",
        "scientific_diagrams",
        "tables_and_arithmetic",
        "visual_puzzle_and_logic"
    ]

    def __init__(self):
        pass

    @classmethod
    def matematiksel_cevap_karsilastir(
        cls,
        tahmin: str,
        referans: str,
        soru_tipi: str = "sayisal",
        tolerans: float = 1e-2,
    ) -> bool:
        """
        Model tahminini referans matematiksel cevaba göre doğrular.
        Sayısal değerlerde toleransPayı, metinsel/seçenekli sorularda tam eşleşme arar.
        """
        tahmin_clean = str(tahmin).strip().lower()
        referans_clean = str(referans).strip().lower()

        if soru_tipi == "sayisal":
            try:
                # String içindeki sayısal değeri ayrıştırma
                val_tahmin = float("".join(c for c in tahmin_clean if c.isdigit() or c in ".-+"))
                val_ref = float(referans_clean)
                return abs(val_tahmin - val_ref) <= tolerans
            except Exception:
                return tahmin_clean == referans_clean
        else:
            return tahmin_clean == referans_clean

    def toplu_mathvista_degerlendir(
        self,
        alan_sonuclari: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """
        MathVista alt disiplinleri için doğruluk oranlarını hesaplar.
        """
        detaylar = {}
        toplam_soru = 0
        toplam_dogru = 0

        for alan in self.MATEMATIK_ALANLARI:
            sorular = alan_sonuclari.get(alan, [])
            N = len(sorular)
            if N == 0:
                continue

            dogru_sayisi = sum(
                1 for s in sorular
                if self.matematiksel_cevap_karsilastir(
                    s.get("tahmin", ""),
                    s.get("referans", ""),
                    s.get("soru_tipi", "sayisal")
                )
            )

            acc = (dogru_sayisi / N) * 100.0
            detaylar[alan] = {
                "toplam_soru": N,
                "dogru_sayisi": dogru_sayisi,
                "dogruluk_yuzdesi": round(acc, 2),
            }

            toplam_soru += N
            toplam_dogru += dogru_sayisi

        genel_acc = (toplam_dogru / max(toplam_soru, 1)) * 100.0

        return {
            "genel_mathvista_skoru": round(genel_acc, 2),
            "toplam_soru": toplam_soru,
            "toplam_dogru": toplam_dogru,
            "alan_bazli_performans": detaylar,
        }

    @classmethod
    def ornek_model_mathvista_raporu(cls, model_adi: str = "Mini-Omni-v1") -> Dict[str, Any]:
        """Örnek model için MathVista değerlendirme simülasyonu."""
        if "GPT-4o" in model_adi:
            base_acc = 0.68
        elif "Claude" in model_adi or "Gemini" in model_adi:
            base_acc = 0.65
        elif "LLaVA" in model_adi or "Qwen" in model_adi:
            base_acc = 0.54
        else:
            base_acc = 0.49

        np.random.seed(hash(model_adi + "_mathvista") % 2**32)
        alan_sonuclari = {}

        for alan in cls.MATEMATIK_ALANLARI:
            sorular = []
            for i in range(20):
                is_correct = np.random.rand() < base_acc
                ref_val = float(i * 3 + 7)
                pred_val = str(ref_val if is_correct else ref_val + 5.0)
                sorular.append({
                    "tahmin": pred_val,
                    "referans": str(ref_val),
                    "soru_tipi": "sayisal"
                })
            alan_sonuclari[alan] = sorular

        evaluator = cls()
        return evaluator.toplu_mathvista_degerlendir(alan_sonuclari)
