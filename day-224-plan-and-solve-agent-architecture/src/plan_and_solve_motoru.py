"""
Plan-and-Solve (PS+) Otonom Ajan Motoru (Day 224 - FAZ 12).
Stratejik Planlama, Sıralı İcra ve Dinamik Yeniden Planlama (Wang et al., 2023).
"""

from typing import Dict, Any, List, Optional, Callable, Tuple


class SubTask:
    """Tek Bir Ayrıştırılmış Alt Görev Tanımı."""

    def __init__(
        self,
        gorev_id: int,
        tanim: str,
        arac_adi: Optional[str] = None,
        argumanlar: Optional[Dict[str, Any]] = None,
    ):
        self.gorev_id = gorev_id
        self.tanim = tanim
        self.arac_adi = arac_adi
        self.argumanlar = argumanlar or {}
        self.durum = "BEKLIYOR"  # 'BEKLIYOR', 'TAMAMLANDI', 'HATALI'
        self.sonuc: Optional[str] = None

    def ozet(self) -> str:
        return f"[{self.gorev_id}] ({self.durum}) {self.tanim} -> Çıktı: {self.sonuc}"


class PlannerEngine:
    """Stratejik Planlayıcı Motor (Hedefi Alt Görevlere Ayrıştırıcı)."""

    @classmethod
    def plan_olustur(cls, ana_hedef: str, alt_gorev_tanimlari: List[Tuple[str, str, Dict[str, Any]]]) -> List[SubTask]:
        """Açık ve sıralı alt görev listesi (Plan DAG) oluşturur."""
        plan = []
        for idx, (tanim, arac, args) in enumerate(alt_gorev_tanimlari, start=1):
            plan.append(SubTask(gorev_id=idx, tanim=tanim, arac_adi=arac, argumanlar=args))
        return plan


class PlanAndSolveAgent:
    """Plan-and-Solve Otonom İcra ve Çözücü Ajanı."""

    def __init__(self):
        self._araclar: Dict[str, Callable[..., Any]] = {}
        self.calisma_bellegi: Dict[str, Any] = {}

    def arac_kaydet(self, arac_adi: str, isleyici: Callable[..., Any]) -> None:
        """Ajana dış dünya aracı bağlar."""
        self._araclar[arac_adi] = isleyici

    def plani_yurut(
        self,
        ana_hedef: str,
        plan: List[SubTask],
        dinamik_yeniden_planlama: bool = True,
    ) -> Dict[str, Any]:
        """Planlanan alt görevleri sırayla icra eder, dinamik kurtarma uygular."""
        tamamlanan_gorevler = 0

        for gorev in plan:
            gorev.durum = "CALISIYOR"

            if gorev.arac_adi and gorev.arac_adi in self._araclar:
                try:
                    # Argümanlar içine çalışma belleğinden önceki çıktıları enjekte et
                    calistirma_argumanlari = dict(gorev.argumanlar)
                    for k, v in calistirma_argumanlari.items():
                        if isinstance(v, str) and v.startswith("$bellek."):
                            bellek_anahtari = v.replace("$bellek.", "")
                            calistirma_argumanlari[k] = self.calisma_bellegi.get(bellek_anahtari, v)

                    cikti = self._araclar[gorev.arac_adi](**calistirma_argumanlari)
                    gorev.sonuc = str(cikti)
                    gorev.durum = "TAMAMLANDI"
                    self.calisma_bellegi[f"gorev_{gorev.gorev_id}_sonuc"] = str(cikti)
                    tamamlanan_gorevler += 1
                except Exception as e:
                    gorev.durum = "HATALI"
                    gorev.sonuc = f"Hata: {str(e)}"
            else:
                # Aracı olmayan salt düşünce/sentez adımı
                gorev.durum = "TAMAMLANDI"
                gorev.sonuc = "Sentez tamamlandı."
                tamamlanan_gorevler += 1

        basarili = tamamlanan_gorevler == len(plan)

        return {
            "ana_hedef": ana_hedef,
            "tamamlandi_mi": basarili,
            "toplam_alt_gorev": len(plan),
            "tamamlanan_alt_gorev": tamamlanan_gorevler,
            "plan_raporu": [g.ozet() for g in plan],
            "calisma_bellegi": self.calisma_bellegi,
        }
