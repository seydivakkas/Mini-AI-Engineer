"""
GAIA (General AI Assistants) Ajan Benchmark Motoru (Day 239 - FAZ 12).
Seviye 1, 2 ve 3 Çok Adımlı Gerçek Dünya Ajan Değerlendirme Paketi (Mialon et al., 2023).
"""

from typing import Dict, Any, List, Optional
import re


class GAIATask:
    """GAIA Benchmark Görev Modeli."""

    def __init__(
        self,
        task_id: str,
        level: int,
        soru: str,
        beklenen_cevap: str,
        gerekli_araclar: List[str],
    ):
        self.task_id = task_id
        self.level = level
        self.soru = soru
        self.beklenen_cevap = beklenen_cevap
        self.gerekli_araclar = gerekli_araclar
        self.tahmin_edilen_cevap: Optional[str] = None
        self.dogru_mu = False


class GAIAEvaluator:
    """Kesin Eşleşme ve Sayısal Tolerans Hakem Modülü."""

    @classmethod
    def normalize_et(cls, metin: str) -> str:
        metin = metin.strip().lower()
        metin = re.sub(r"[,$%]", "", metin)
        metin = re.sub(r"\b(usd|tl|eur|dollar|dollars|euro|tl)\b", "", metin).strip()
        return metin

    @classmethod
    def dogrula(cls, tahmin: str, beklenen: str, tolerans: float = 0.01) -> bool:
        """Sayısal tolerans veya kesin metin eşleşmesi ile cevabı değerlendirir."""
        norm_tahmin = cls.normalize_et(tahmin)
        norm_beklenen = cls.normalize_et(beklenen)

        if norm_tahmin == norm_beklenen:
            return True

        # Sayısal Değer Kontrolü
        try:
            val_t = float(norm_tahmin)
            val_b = float(norm_beklenen)
            fark = abs(val_t - val_b)
            referans = abs(val_b) if abs(val_b) > 1e-6 else 1.0
            return (fark / referans) <= tolerans
        except ValueError:
            return False


class GAIAAgentHarness:
    """GAIA Görev Havuzunu Koşturan ve Kapsamlı Skor Üreten Test Motoru."""

    def __init__(self):
        self.gorevler: List[GAIATask] = []

    def gorev_ekle(self, gorev: GAIATask):
        self.gorevler.append(gorev)

    def ornek_gaia_havuzu_olustur(self):
        """Seviye 1, 2 ve 3 standart GAIA örnek görevleri yükler."""
        # Seviye 1: Basit Arama & Araç
        self.gorev_ekle(
            GAIATask(
                "gaia-101",
                1,
                "2023 yılında Nobel Fizik Ödülü'nü kaç bilim insanı paylaştı?",
                "3",
                ["WebSearch"],
            )
        )
        self.gorev_ekle(
            GAIATask(
                "gaia-102",
                1,
                "PDF raporunun 4. sayfasındaki Q3 toplam cirosu kaç USD?",
                "4500000",
                ["PDFParser"],
            )
        )
        # Seviye 2: Çok Adımlı Arama + Matematik
        self.gorev_ekle(
            GAIATask(
                "gaia-201",
                2,
                "Türkiye ve Almanya nüfus farkının %15'i kaçtır (2024)?",
                "150000",
                ["WebSearch", "PythonCalculator"],
            )
        )
        # Seviye 3: Karmaşık Çok Modlu Ajan Zinciri
        self.gorev_ekle(
            GAIATask(
                "gaia-301",
                3,
                "Excel tablosundaki satış verisini oku, döviz kuru ile çarp, kdv ekle ve sonucu döndür.",
                "128450.50",
                ["ExcelReader", "CurrencyAPI", "PythonCalculator"],
            )
        )

    def degerlendir(self, tahminler: Dict[str, str]) -> Dict[str, Any]:
        """Tüm görevleri hakemle değerlendirip seviye bazlı karne çıkarır."""
        level_stats = {1: {"dogru": 0, "toplam": 0}, 2: {"dogru": 0, "toplam": 0}, 3: {"dogru": 0, "toplam": 0}}

        for g in self.gorevler:
            tahmin = tahminler.get(g.task_id, "")
            g.tahmin_edilen_cevap = tahmin
            g.dogru_mu = GAIAEvaluator.dogrula(tahmin, g.beklenen_cevap)

            lvl = g.level
            level_stats[lvl]["toplam"] += 1
            if g.dogru_mu:
                level_stats[lvl]["dogru"] += 1

        skor_l1 = (level_stats[1]["dogru"] / max(1, level_stats[1]["toplam"])) * 100
        skor_l2 = (level_stats[2]["dogru"] / max(1, level_stats[2]["toplam"])) * 100
        skor_l3 = (level_stats[3]["dogru"] / max(1, level_stats[3]["toplam"])) * 100

        toplam_dogru = sum(s["dogru"] for s in level_stats.values())
        toplam_gorev = sum(s["toplam"] for s in level_stats.values())
        genel_skor = (toplam_dogru / max(1, toplam_gorev)) * 100

        return {
            "seviye_1_basari": skor_l1,
            "seviye_2_basari": skor_l2,
            "seviye_3_basari": skor_l3,
            "genel_gaia_skoru": genel_skor,
            "detaylar": self.gorevler,
        }
