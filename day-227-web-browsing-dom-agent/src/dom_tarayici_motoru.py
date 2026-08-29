"""
Web Tarayıcı ve DOM Ağacı Ajan Motoru (Day 227 - FAZ 12).
HTML DOM Budama, Set-of-Marks [ID] Etkileşim Ağacı ve Otonom Gezinme.
"""

from typing import Dict, Any, List, Optional, Tuple
import re


class DOMElement:
    """Nitelikli ve Budanmış DOM Ağacı Düğümü."""

    def __init__(
        self,
        eleman_id: int,
        etiket: str,
        metin: str,
        nitelikler: Optional[Dict[str, str]] = None,
        etkilesimli_mi: bool = False,
    ):
        self.eleman_id = eleman_id
        self.etiket = etiket.lower()
        self.metin = metin.strip()
        self.nitelikler = nitelikler or {}
        self.etkilesimli_mi = etkilesimli_mi

    def format_metni(self) -> str:
        """Ajan için Set-of-Marks formatında erişilebilirlik çıktısı üretir."""
        if self.etkilesimli_mi:
            nitelik_str = " ".join([f'{k}="{v}"' for k, v in self.nitelikler.items()])
            return f"[{self.eleman_id}] <{self.etiket} {nitelik_str}> {self.metin}"
        return f"<{self.etiket}> {self.metin}"


class DOMTreePruner:
    """Ham HTML Gürültüsünü Temizleyen ve Erişilebilirlik Ağacını Kuran Motor."""

    @classmethod
    def html_temizle_ve_buda(cls, ham_html: str) -> List[DOMElement]:
        """Gereksiz script, style ve yorumları temizleyip etkileşimli elemanları numaralandırır."""
        # 1. Script ve Style bloklarını temizle
        temiz = re.sub(r"<script.*?</script>", "", ham_html, flags=re.DOTALL | re.IGNORECASE)
        temiz = re.sub(r"<style.*?</style>", "", temiz, flags=re.DOTALL | re.IGNORECASE)
        temiz = re.sub(r"<!--.*?-->", "", temiz, flags=re.DOTALL)

        elemanlar: List[DOMElement] = []
        sayac = 1

        # 2. Etiketleri ayrıştır (input, button, a, h1, h2, h3, div, p, span)
        # Önce self-closing veya tekil etiketleri (input, img), sonra ikili etiketleri yakala
        tag_token_kalibi = re.compile(r"<(\w+)([^>]*)>(?:(.*?)</\1>)?|<(\w+)([^>]*)/>", flags=re.DOTALL)

        # Temiz metinden etiketleri tara
        for tag in ["input", "button", "a", "h1", "h2", "h3", "p", "span", "div"]:
            pattern = re.compile(rf"<({tag})([^>]*)>(.*?)</\1>|<({tag})([^>]*)>", flags=re.DOTALL | re.IGNORECASE)
            for match in pattern.finditer(temiz):
                t_name = (match.group(1) or match.group(4)).lower()
                nitelikler_ham = match.group(2) or match.group(5) or ""
                icerik = match.group(3) or ""
                icerik_metni = re.sub(r"<.*?>", " ", icerik).strip()

                nitelikler = {}
                for n_match in re.finditer(r'(\w+)=["\'](.*?)["\']', nitelikler_ham):
                    nitelikler[n_match.group(1)] = n_match.group(2)

                etkilesimli = t_name in ["button", "input", "a", "select"] or "click" in nitelikler_ham.lower()

                if icerik_metni or t_name == "input":
                    # Mükerrer div/span eklemelerini önle
                    if t_name in ["div", "span"] and not etkilesimli and len(icerik_metni) > 60:
                        continue
                    eleman = DOMElement(
                        eleman_id=sayac if etkilesimli else 0,
                        etiket=t_name,
                        metin=icerik_metni,
                        nitelikler=nitelikler,
                        etkilesimli_mi=etkilesimli,
                    )
                    if etkilesimli:
                        sayac += 1
                    elemanlar.append(eleman)

        return elemanlar

    @classmethod
    def agaci_metne_donustur(cls, elemanlar: List[DOMElement]) -> str:
        """Tüm budanmış DOM elemanlarını tek bir okunabilir prompt metnine derler."""
        return "\n".join([e.format_metni() for e in elemanlar])


class WebBrowsingAgent:
    """Otonom Web Tarayıcı ve DOM Etkileşim Ajanı."""

    def __init__(self, baslangic_html: str):
        self.mevcut_html = baslangic_html
        self.eleman_havuzu: Dict[int, DOMElement] = {}
        self.erisebilirlik_agaci = self._agaci_guncelle()
        self.tarayici_hafizasi: List[str] = []

    def _agaci_guncelle(self) -> str:
        elemanlar = DOMTreePruner.html_temizle_ve_buda(self.mevcut_html)
        self.eleman_havuzu = {e.eleman_id: e for e in elemanlar if e.etkilesimli_mi}
        return DOMTreePruner.agaci_metne_donustur(elemanlar)

    def eylem_ayristir(self, eylem_metni: str) -> Tuple[str, List[str]]:
        """'Type[1, "Laptop"]' veya 'Click[2]' ifadesini ayrıştırır."""
        eslesme = re.match(r"^(\w+)\[(.*)\]$", eylem_metni.strip())
        if eslesme:
            komut = eslesme.group(1)
            argumanlar = [a.strip().strip("'\"") for a in eslesme.group(2).split(",")]
            return komut, argumanlar
        return "Bilinmeyen", [eylem_metni.strip()]

    def eylem_icra_et(self, eylem_ifadesi: str) -> str:
        """Tarayıcı eylemini icra edip gözlem metni döner."""
        komut, args = self.eylem_ayristir(eylem_ifadesi)

        if komut == "Click":
            eleman_id = int(args[0])
            if eleman_id in self.eleman_havuzu:
                el = self.eleman_havuzu[eleman_id]
                return f"GÖZLEM: [{eleman_id}] <{el.etiket}> '{el.metin}' tıklandı. Sayfa güncellendi."
            return f"HATA: [{eleman_id}] numaralı etkileşimli eleman bulunamadı."

        elif komut == "Type":
            eleman_id = int(args[0])
            yazilan_metin = args[1]
            if eleman_id in self.eleman_havuzu:
                el = self.eleman_havuzu[eleman_id]
                el.nitelikler["value"] = yazilan_metin
                return f"GÖZLEM: [{eleman_id}] <{el.etiket}> alanına '{yazilan_metin}' yazıldı."
            return f"HATA: [{eleman_id}] giriş kutusu bulunamadı."

        elif komut == "Extract":
            eleman_id = int(args[0])
            if eleman_id in self.eleman_havuzu:
                return f"GÖZLEM: Kazınan Veri -> {self.eleman_havuzu[eleman_id].metin}"
            return f"HATA: [{eleman_id}] verisi okunamadı."

        elif komut == "Finish":
            return f"GÖREV_TAMAMLANDI: {args[0]}"

        return f"HATA: Tanımsız tarayıcı eylemi: {komut}"

    def otonom_gezin(self, hedef_gorev: str, eylem_adimlari: List[str]) -> Dict[str, Any]:
        """Hedef görev için belirlenen eylem dizilimini icra eder."""
        gezinme_raporu = []
        nihai_yanit = None

        for eylem in eylem_adimlari:
            gozlem = self.eylem_icra_et(eylem)
            gezinme_raporu.append(f"Eylem: {eylem} -> {gozlem}")
            if "GÖREV_TAMAMLANDI:" in gozlem:
                nihai_yanit = gozlem.replace("GÖREV_TAMAMLANDI:", "").strip()
                break

        return {
            "hedef_gorev": hedef_gorev,
            "tamamlandi_mi": nihai_yanit is not None,
            "nihai_yanit": nihai_yanit,
            "adim_sayisi": len(gezinme_raporu),
            "gezinme_raporu": gezinme_raporu,
        }
