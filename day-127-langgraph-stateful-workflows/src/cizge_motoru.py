"""
LangGraph Durumsal Çizge (StateGraph) Yürütme Motoru (Day 127 - Faz 7).
Düğümler (Nodes), Kenarlar (Edges), Koşullu Yönlendirme (Conditional Routing) ve Human-in-the-Loop kesinti yönetimi.
"""

from typing import Dict, Any, Callable, List, Optional
import copy

from .cizge_durumu import DurumIndirgeyici
from .kontrol_noktasi_yoneticisi import CheckpointYoneticisi

END = "END"


class DurumsalCizge:
    """LangGraph eşdeğeri durum korumalı ve döngülü çizge motoru."""

    def __init__(self):
        self.dugumler: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
        self.duz_kenarlar: Dict[str, str] = {}
        self.kosullu_kenarlar: Dict[str, Dict[str, Any]] = {}
        self.giris_dugumu: Optional[str] = None
        self.kesinti_dugumleri: set = set()
        self.checkpoint_yoneticisi = CheckpointYoneticisi()

    def add_node(self, isim: str, fonksiyon: Callable[[Dict[str, Any]], Dict[str, Any]]):
        self.dugumler[isim] = fonksiyon

    def add_edge(self, baslangic: str, bitis: str):
        self.duz_kenarlar[baslangic] = bitis

    def add_conditional_edges(
        self,
        baslangic: str,
        yonlendirici_fn: Callable[[Dict[str, Any]], str],
        yol_haritasi: Dict[str, str],
    ):
        self.kosullu_kenarlar[baslangic] = {
            "yonlendirici": yonlendirici_fn,
            "yol_haritasi": yol_haritasi,
        }

    def set_entry_point(self, dugum_adi: str):
        self.giris_dugumu = dugum_adi

    def kesinti_tanimla(self, dugum_adi: str):
        """Bu düğüm çalıştırılmadan önce insan onayı için akışı duraklatır (HITL)."""
        self.kesinti_dugumleri.add(dugum_adi)

    def calistir(
        self,
        baslangic_durumu: Dict[str, Any],
        max_tekrar: int = 10,
        insan_onay_yaniti: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Çizgeyi giriş noktasından END düğümüne kadar döngülü işletir."""
        if not self.giris_dugumu or self.giris_dugumu not in self.dugumler:
            raise ValueError(f"Geçersiz giriş düğümü: {self.giris_dugumu}")

        mevcut_durum = copy.deepcopy(baslangic_durumu)
        su_anki_dugum = self.giris_dugumu
        adim_sayaci = 0

        # Eğer insan onayı verildiyse duruma işle
        if insan_onay_yaniti is not None:
            mevcut_durum["insan_onayladi_mi"] = insan_onay_yaniti

        while su_anki_dugum != END and adim_sayaci < max_tekrar:
            adim_sayaci += 1

            # 1. Human-in-the-Loop Kesinti Kontrolü
            if su_anki_dugum in self.kesinti_dugumleri and mevcut_durum.get("insan_onayladi_mi") is None:
                mevcut_durum["nihai_durum"] = "BEKLIYOR_INSAN_ONAYI"
                mevcut_durum["adim_gecmisi"].append(f"INTERRUPT ({su_anki_dugum})")
                self.checkpoint_yoneticisi.kaydet(adim_sayaci, f"INTERRUPT_{su_anki_dugum}", mevcut_durum)
                return {
                    "tamamlandi": False,
                    "kesinti_noktasi": su_anki_dugum,
                    "durum": mevcut_durum,
                    "adim_sayisi": adim_sayaci,
                }

            # 2. Düğümü Çalıştır
            node_fn = self.dugumler[su_anki_dugum]
            guncelleme = node_fn(mevcut_durum)
            guncelleme.setdefault("adim_gecmisi", [su_anki_dugum])

            # 3. Durumu İndirge ve Birleştir
            mevcut_durum = DurumIndirgeyici.indirge(mevcut_durum, guncelleme)
            mevcut_durum["tekrar_sayisi"] = adim_sayaci

            # 4. Kontrol Noktası Kaydet
            self.checkpoint_yoneticisi.kaydet(adim_sayaci, su_anki_dugum, mevcut_durum)

            # 5. Sıradaki Düğümü Belirle
            if su_anki_dugum in self.kosullu_kenarlar:
                bilgi = self.kosullu_kenarlar[su_anki_dugum]
                rota_anahtari = bilgi["yonlendirici"](mevcut_durum)
                su_anki_dugum = bilgi["yol_haritasi"].get(rota_anahtari, END)
            elif su_anki_dugum in self.duz_kenarlar:
                su_anki_dugum = self.duz_kenarlar[su_anki_dugum]
            else:
                su_anki_dugum = END

        mevcut_durum.setdefault("nihai_durum", "TAMAMLANDI")
        return {
            "tamamlandi": True,
            "kesinti_noktasi": None,
            "durum": mevcut_durum,
            "adim_sayisi": adim_sayaci,
        }
