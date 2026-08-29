"""
Graf Tabanlı Ajan İş Akışı Motoru (Day 231 - FAZ 12).
LangGraph / StateGraph Mimarisi, Durum Geçişleri ve Döngüsel Kontrol.
"""

from typing import Dict, Any, List, Optional, Callable, Union, Tuple
import copy

START = "__START__"
END = "__END__"


class AgentState(dict):
    """Tüm Düğümler Arasında Paylaşılan Durum (Shared State)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "mesajlar" not in self:
            self["mesajlar"] = []
        if "adim_sayisi" not in self:
            self["adim_sayisi"] = 0
        if "test_gecti_mi" not in self:
            self["test_gecti_mi"] = False

    def log_ekle(self, mesaj: str):
        self["mesajlar"].append(mesaj)


class CompiledStateGraph:
    """Derlenmiş ve Çalıştırılabilir Durum Grafı."""

    def __init__(
        self,
        dugumler: Dict[str, Callable[[AgentState], AgentState]],
        kenarlar: Dict[str, str],
        kosullu_kenarlar: Dict[str, Tuple[Callable[[AgentState], str], Dict[str, str]]],
        giris_dugumu: str,
        max_tekrarlama: int = 10,
    ):
        self.dugumler = dugumler
        self.kenarlar = kenarlar
        self.kosullu_kenarlar = kosullu_kenarlar
        self.giris_dugumu = giris_dugumu
        self.max_tekrarlama = max_tekrarlama

    def calistir(self, baslangic_durumu: AgentState) -> AgentState:
        """Grafı giriş noktasından başlatır ve END düğümüne kadar döngüsel koşturur."""
        durum = copy.deepcopy(baslangic_durumu)
        mevcut_dugum = self.giris_dugumu
        tekrarlama_sayaci = 0

        while mevcut_dugum != END:
            tekrarlama_sayaci += 1
            if tekrarlama_sayaci > self.max_tekrarlama:
                durum.log_ekle(f"⚠️ [GÜVENLİK SINIRI] Maksimum tekrarlama ({self.max_tekrarlama}) aşıldı.")
                break

            durum["adim_sayisi"] = tekrarlama_sayaci
            durum.log_ekle(f"➡️ [DÜĞÜM ÇALIŞIYOR]: {mevcut_dugum}")

            # 1. Düğüm Fonksiyonunu Çalıştır
            dugum_fonk = self.dugumler[mevcut_dugum]
            sonuc = dugum_fonk(durum)
            if isinstance(sonuc, AgentState):
                durum = sonuc
            elif isinstance(sonuc, dict):
                durum = AgentState(**sonuc)
            else:
                durum = AgentState()

            # 2. Sonraki Düğümü Belirle (Koşullu veya Doğrudan Kenar)
            if mevcut_dugum in self.kosullu_kenarlar:
                yonlendirici_fonk, hedef_haritasi = self.kosullu_kenarlar[mevcut_dugum]
                rota = yonlendirici_fonk(durum)
                mevcut_dugum = hedef_haritasi.get(rota, END)
                durum.log_ekle(f"  ↪️ Koşullu Yönlendirme: '{rota}' -> '{mevcut_dugum}'")
            elif mevcut_dugum in self.kenarlar:
                mevcut_dugum = self.kenarlar[mevcut_dugum]
            else:
                mevcut_dugum = END

        durum.log_ekle("🏁 [GRAF TAMAMLANDI]: END düğümüne ulaşıldı.")
        return durum


class StateGraph:
    """Durum Grafı Tasarım ve Derleme Motoru."""

    def __init__(self):
        self.dugumler: Dict[str, Callable[[AgentState], AgentState]] = {}
        self.kenarlar: Dict[str, str] = {}
        self.kosullu_kenarlar: Dict[str, Tuple[Callable[[AgentState], str], Dict[str, str]]] = {}
        self.giris_dugumu: Optional[str] = None

    def add_node(self, ad: str, fonk: Callable[[AgentState], AgentState]):
        self.dugumler[ad] = fonk

    def set_entry_point(self, ad: str):
        self.giris_dugumu = ad

    def add_edge(self, baslangic: str, hedef: str):
        self.kenarlar[baslangic] = hedef

    def add_conditional_edges(
        self,
        baslangic: str,
        yonlendirici: Callable[[AgentState], str],
        hedef_haritasi: Dict[str, str],
    ):
        self.kosullu_kenarlar[baslangic] = (yonlendirici, hedef_haritasi)

    def compile(self, max_tekrarlama: int = 10) -> CompiledStateGraph:
        if not self.giris_dugumu or self.giris_dugumu not in self.dugumler:
            raise ValueError("Geçerli bir giriş düğümü tanımlanmalıdır.")

        return CompiledStateGraph(
            dugumler=self.dugumler,
            kenarlar=self.kenarlar,
            kosullu_kenarlar=self.kosullu_kenarlar,
            giris_dugumu=self.giris_dugumu,
            max_tekrarlama=max_tekrarlama,
        )
