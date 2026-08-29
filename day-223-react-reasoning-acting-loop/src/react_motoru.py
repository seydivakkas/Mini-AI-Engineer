"""
ReAct (Reasoning + Acting) Otonom Ajan Motoru (Day 223 - FAZ 12).
Düşünce, Eylem ve Gözlem (Thought-Action-Observation) Döngüsü (Yao et al., 2022).
"""

from typing import Dict, Any, List, Optional, Callable, Tuple
import re


class ReActStep:
    """Tek bir ReAct Karar ve İcra Adımı."""

    def __init__(self, adim_no: int, dusunce: str, eylem: Optional[str] = None, gozlem: Optional[str] = None):
        self.adim_no = adim_no
        self.dusunce = dusunce
        self.eylem = eylem
        self.gozlem = gozlem

    def metin_formati(self) -> str:
        """Adımı ReAct prompt formatına dönüştürür."""
        metin = f"Düşünce {self.adim_no}: {self.dusunce}\n"
        if self.eylem:
            metin += f"Eylem {self.adim_no}: {self.eylem}\n"
        if self.gozlem:
            metin += f"Gözlem {self.adim_no}: {self.gozlem}\n"
        return metin


class ReActMemoryTrace:
    """ReAct Ajanının Çalışma Belleği ve Tarihçe İzleri."""

    def __init__(self):
        self.adimlar: List[ReActStep] = []

    def adim_ekle(self, adim: ReActStep) -> None:
        self.adimlar.append(adim)

    def tam_baglam_metni(self) -> str:
        return "".join([a.metin_formati() for a in self.adimlar])

    def son_adim(self) -> Optional[ReActStep]:
        return self.adimlar[-1] if self.adimlar else None


class ReActAgent:
    """Otonom ReAct (Düşünce + Eylem) Karar Motoru."""

    def __init__(self):
        self._araclar: Dict[str, Callable[[str], str]] = {}

    def arac_kaydet(self, arac_adi: str, isleyici: Callable[[str], str]) -> None:
        """Ajana yeni bir harici araç bağlar."""
        self._araclar[arac_adi] = isleyici

    def eylem_ayristir(self, eylem_metni: str) -> Tuple[str, str]:
        """'Arama[Ankara Nüfusu]' veya 'Finish[Sonuç]' ifadesini (arac, arg) çiftine ayırır."""
        eslesme = re.match(r"^(\w+)\[(.*)\]$", eylem_metni.strip())
        if eslesme:
            return eslesme.group(1), eslesme.group(2)
        return "Bilinmeyen", eylem_metni.strip()

    def adim_yurut(self, arac_adi: str, arguman: str) -> str:
        """Harici aracı çalıştırıp gözlem metni üretir."""
        if arac_adi == "Finish":
            return f"GÖREV_TAMAMLANDI: {arguman}"
        if arac_adi not in self._araclar:
            return f"HATA: '{arac_adi}' adında bir araç bulunamadı."
        try:
            return str(self._araclar[arac_adi](arguman))
        except Exception as e:
            return f"ARAÇ_HATASI: {str(e)}"

    def otonom_coz(
        self,
        hedef_soru: str,
        simule_edilen_plan: List[Tuple[str, str]],
        max_adim: int = 5,
    ) -> Dict[str, Any]:
        """
        Belirlenen hedef soru için ReAct döngüsünü yürütür.
        simule_edilen_plan: [(Düşünce, Eylem)] listesi.
        """
        hafiza = ReActMemoryTrace()
        nihai_cevap = None

        for idx, (dusunce, eylem_ifadesi) in enumerate(simule_edilen_plan, start=1):
            if idx > max_adim:
                break

            arac_adi, arguman = self.eylem_ayristir(eylem_ifadesi)
            gozlem = self.adim_yurut(arac_adi, arguman)

            adim = ReActStep(adim_no=idx, dusunce=dusunce, eylem=eylem_ifadesi, gozlem=gozlem)
            hafiza.adim_ekle(adim)

            if arac_adi == "Finish":
                nihai_cevap = arguman
                break

        return {
            "hedef_soru": hedef_soru,
            "tamamlandi_mi": nihai_cevap is not None,
            "nihai_cevap": nihai_cevap,
            "toplam_adim": len(hafiza.adimlar),
            "hafiza_izi": hafiza.tam_baglam_metni(),
        }
