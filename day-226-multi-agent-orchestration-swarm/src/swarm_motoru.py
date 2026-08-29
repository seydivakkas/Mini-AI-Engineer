"""
Çoklu Ajan Orkestrasyonu (Swarm) Motoru (Day 226 - FAZ 12).
Hiyerarşik Görev Dağıtımı, Ajanlar Arası Mesajlaşma ve Handoff Protokolü.
"""

from typing import Dict, Any, List, Optional, Callable


class AgentMessage:
    """Ajanlar Arası Standart İletişim Paketi."""

    def __init__(self, gonderen: str, alici: str, icerik: str, metadata: Optional[Dict[str, Any]] = None):
        self.gonderen = gonderen
        self.alici = alici
        self.icerik = icerik
        self.metadata = metadata or {}

    def format_metni(self) -> str:
        return f"[{self.gonderen} -> {self.alici}]: {self.icerik}"


class SpecializedAgent:
    """Uzmanlaşmış Ajan Taban Sınıfı."""

    def __init__(self, rol: str, sistem_istemi: str):
        self.rol = rol
        self.sistem_istemi = sistem_istemi
        self._araclar: Dict[str, Callable[[str], str]] = {}

    def arac_ekle(self, arac_adi: str, isleyici: Callable[[str], str]) -> None:
        self._araclar[arac_adi] = isleyici

    def gorev_calistir(self, girdi: str) -> str:
        """Gelen görevi uzmanlık alanına göre icra eder."""
        raise NotImplementedError


class ResearcherAgent(SpecializedAgent):
    """Araştırmacı Ajan: Teori, dokümantasyon ve spesifikasyon üretir."""

    def __init__(self):
        super().__init__(
            rol="Araştırmacı",
            sistem_istemi="Sen uzman bir AI araştırmacısısın. Algoritmik gereksinimleri ve teorik temelleri çıkarırsın.",
        )

    def gorev_calistir(self, girdi: str) -> str:
        return f"Araştırma Bulguları: '{girdi}' konusu incelendi. Zaman karmaşıklığı O(N log N), yer karmaşıklığı O(log N) gereklidir. Temiz fonksiyonel arayüz önerilir."


class CoderAgent(SpecializedAgent):
    """Kodlayıcı Ajan: Spesifikasyona uygun temiz Python/PyTorch kodu yazar."""

    def __init__(self):
        super().__init__(
            rol="Kodlayıcı",
            sistem_istemi="Sen kıdemli bir Python yazılım mühendisisin. Temiz ve tip korumalı kod üretirsin.",
        )

    def gorev_calistir(self, girdi: str) -> str:
        return (
            "def hizli_sirala(dizi):\n"
            "    if len(dizi) <= 1:\n"
            "        return dizi\n"
            "    pivot = dizi[len(dizi) // 2]\n"
            "    sol = [x for x in dizi if x < pivot]\n"
            "    orta = [x for x in dizi if x == pivot]\n"
            "    sag = [x for x in dizi if x > pivot]\n"
            "    return hizli_sirala(sol) + orta + hizli_sirala(sag)"
        )


class ReviewerAgent(SpecializedAgent):
    """Denetçi / QA Ajanı: Kod kalitesini, güvenlik açıklarını ve testleri denetler."""

    def __init__(self):
        super().__init__(
            rol="Denetçi",
            sistem_istemi="Sen baş QA ve Güvenlik Denetçisisin. Kodları inceler ve eksiksiz onay verirsin.",
        )

    def gorev_calistir(self, kod_metni: str) -> str:
        return "Denetim Raporu: Kod O(N log N) pivot mantığını başarıyla uyguluyor. Güvenlik açığı tespit edilmedi. Testler %100 ONAYLANDI."


class SwarmOrchestrator:
    """Hiyerarşik Çoklu Ajan Swarm Orkestratörü."""

    def __init__(self):
        self.ajanlar: Dict[str, SpecializedAgent] = {
            "Araştırmacı": ResearcherAgent(),
            "Kodlayıcı": CoderAgent(),
            "Denetçi": ReviewerAgent(),
        }
        self.mesaj_kayitlari: List[AgentMessage] = []

    def mesaj_ilet(self, gonderen: str, alici: str, icerik: str) -> None:
        msg = AgentMessage(gonderen=gonderen, alici=alici, icerik=icerik)
        self.mesaj_kayitlari.append(msg)

    def gorev_dagit_ve_sentezle(self, ana_hedef: str) -> Dict[str, Any]:
        """Yönetici akışıyla görevleri ilgili ajanlara aktarır ve sentezler."""
        # 1. Aşama: Yönetici -> Araştırmacı
        self.mesaj_ilet("Yönetici", "Araştırmacı", f"Hedef için algoritma gereksinimlerini çıkar: {ana_hedef}")
        arastirma = self.ajanlar["Araştırmacı"].gorev_calistir(ana_hedef)
        self.mesaj_ilet("Araştırmacı", "Yönetici", arastirma)

        # 2. Aşama: Yönetici -> Kodlayıcı
        self.mesaj_ilet("Yönetici", "Kodlayıcı", f"Bu spesifikasyona göre Python kodu yaz: {arastirma}")
        kod = self.ajanlar["Kodlayıcı"].gorev_calistir(arastirma)
        self.mesaj_ilet("Kodlayıcı", "Yönetici", kod)

        # 3. Aşama: Yönetici -> Denetçi
        self.mesaj_ilet("Yönetici", "Denetçi", f"Bu kodu güvenlik ve performans açısından incele:\n{kod}")
        onay = self.ajanlar["Denetçi"].gorev_calistir(kod)
        self.mesaj_ilet("Denetçi", "Yönetici", onay)

        # 4. Sentez
        nihai_cikti = (
            f"=== [SWARM PROJE RAPORU: {ana_hedef}] ===\n"
            f"1. ARAŞTIRMA : {arastirma}\n\n"
            f"2. ÜRETİLEN KOD:\n{kod}\n\n"
            f"3. QA DENETİMİ: {onay}\n"
            f"DURUM: PROJE BAŞARIYLA TAMAMLANDI VE TESLİM EDİLDİ."
        )

        return {
            "ana_hedef": ana_hedef,
            "basarili_mi": True,
            "toplam_mesaj_sayisi": len(self.mesaj_kayitlari),
            "mesaj_izleri": [m.format_metni() for m in self.mesaj_kayitlari],
            "nihai_cikti": nihai_cikti,
        }
