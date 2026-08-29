"""
Ajan Öz-Yansıtma ve Öz-Değerlendirme Motoru (Day 237 - FAZ 12).
Self-Refine & Reflexion: Eleştirmen Denetçi (Critic) ve Yinelemeli İyileştirme Döngüsü.
"""

from typing import Dict, Any, List, Optional, Tuple


class EvaluationScore:
    """Rubrik Bazlı Değerlendirme ve Eleştiri Raporu."""

    def __init__(
        self,
        dogruluk: float,
        guvenlik: float,
        tamlik: float,
        elestiri_metni: str,
        esik_puani: float = 90.0,
    ):
        self.dogruluk = dogruluk
        self.guvenlik = guvenlik
        self.tamlik = tamlik
        self.toplam_skor = dogruluk + guvenlik + tamlik
        self.elestiri_metni = elestiri_metni
        self.onaylandi_mi = self.toplam_skor >= esik_puani


class ReflectionCritic:
    """Çıktıyı Rubrik Kriterlerine Göre Eleştiren ve Skorlayan Denetçi Ajan."""

    @classmethod
    def degerlendir(cls, kod_taslagi: str, esik_puani: float = 90.0) -> EvaluationScore:
        """Kodu inceler, rubrik puanlarını ve yapıcı eleştiriyi üretir."""
        dogruluk = 40.0
        guvenlik = 40.0
        tamlik = 20.0
        elestiriler = []

        # Güvenlik Kontrolü (Bcrypt / Argon2 / Hashleme)
        if "bcrypt" not in kod_taslagi and "hashlib" not in kod_taslagi:
            guvenlik -= 30.0
            elestiriler.append("Şifreler düz metin olarak karşılaştırılıyor! Bcrypt/Argon2 kullanılmalı.")

        # Tamlık ve Tip İpuçları Kontrolü
        if "bool" not in kod_taslagi and "str" not in kod_taslagi:
            tamlik -= 10.0
            elestiriler.append("Tip belirteçleri (Type hints: str, bool) ve docstring eksik.")

        # Hata Yakalama Kontrolü
        if "try" not in kod_taslagi and "except" not in kod_taslagi and len(elestiriler) > 0:
            dogruluk -= 10.0
            elestiriler.append("Girdi tipleri için exception handling eklenmeli.")

        if not elestiriler:
            elestiri_metni = "Kod rubrik standartlarına tam uyumlu. Güvenlik, doğruluk ve tamlık mükemmel."
        else:
            elestiri_metni = " ".join(elestiriler)

        return EvaluationScore(
            dogruluk=dogruluk,
            guvenlik=guvenlik,
            tamlik=tamlik,
            elestiri_metni=elestiri_metni,
            esik_puani=esik_puani,
        )


class SelfRefiningAgent:
    """Öz-Yansıtma ve İyileştirme Döngüsünü Yöneten Ajan."""

    def __init__(self, esik_puani: float = 90.0, maks_iterasyon: int = 3):
        self.esik_puani = esik_puani
        self.maks_iterasyon = maks_iterasyon
        self.iterasyon_gecmisi: List[Dict[str, Any]] = []

    def iyilestir_ve_tamamla(self, taslaklar: List[str]) -> Dict[str, Any]:
        """Taslakları sırayla eleştirip eşik skorunu geçene kadar iyileştirir."""
        self.iterasyon_gecmisi.clear()

        for iter_no, taslak in enumerate(taslaklar, 1):
            skor_raporu = ReflectionCritic.degerlendir(taslak, self.esik_puani)
            kayit = {
                "iterasyon": iter_no,
                "taslak": taslak,
                "skor": skor_raporu.toplam_skor,
                "dogruluk": skor_raporu.dogruluk,
                "guvenlik": skor_raporu.guvenlik,
                "tamlik": skor_raporu.tamlik,
                "elestiri": skor_raporu.elestiri_metni,
                "onaylandi": skor_raporu.onaylandi_mi,
            }
            self.iterasyon_gecmisi.append(kayit)

            if skor_raporu.onaylandi_mi or iter_no >= self.maks_iterasyon:
                return {
                    "nihai_taslak": taslak,
                    "toplam_iterasyon": iter_no,
                    "nihai_skor": skor_raporu.toplam_skor,
                    "onaylandi": skor_raporu.onaylandi_mi,
                    "gecmis": self.iterasyon_gecmisi,
                }

        return {
            "nihai_taslak": taslaklar[-1],
            "toplam_iterasyon": len(taslaklar),
            "nihai_skor": self.iterasyon_gecmisi[-1]["skor"],
            "onaylandi": False,
            "gecmis": self.iterasyon_gecmisi,
        }
