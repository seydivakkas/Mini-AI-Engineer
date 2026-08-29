"""
Day 231: Graf Tabanlı Ajan İş Akışı (LangGraph / StateGraph) Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.stategraph_motoru import (
    AgentState,
    StateGraph,
    CompiledStateGraph,
    START,
    END,
)
from src.graph_profilleyici import GraphProfilleyici
from src.gorsellestirici import GraphGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 231 (FAZ 12): GRAF TABANLI AJAN İŞ AKIŞI (LANGGRAPH / STATEGRAPH) - DURUM GEÇİŞLERİ VE DÖNGÜSEL KONTROL")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Düğümlerin (Nodes) Tanımlanması
    # -------------------------------------------------------------
    print("\n[1/4] StateGraph Düğümleri (Nodes) ve Görev Mantığı Tanımlanıyor...")

    def kodlayici(state: AgentState) -> AgentState:
        deneme = state.get("deneme_sayisi", 0) + 1
        state["deneme_sayisi"] = deneme
        if deneme == 1:
            state["kod"] = "def fibonacci(n): return n  # Hatalı Taslak"
            state.log_ekle("Kodlayıcı: [1. Deneme] Hatalı taslak kod üretildi.")
        else:
            state["kod"] = "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
            state.log_ekle("Kodlayıcı: [2. Deneme] Özyinelemeli doğru kod üretildi.")
        return state

    def denetci(state: AgentState) -> AgentState:
        kod = state.get("kod", "")
        if "fibonacci(n-1)" in kod:
            state["test_gecti_mi"] = True
            state.log_ekle("Denetçi: Birim testler %100 BAŞARIYLA geçti.")
        else:
            state["test_gecti_mi"] = False
            state.log_ekle("Denetçi: Birim testler BAŞARISIZ oldu.")
        return state

    def kosullu_yonlendirici(state: AgentState) -> str:
        return "tamam" if state.get("test_gecti_mi", False) else "tekrar"

    # -------------------------------------------------------------
    # ADIM 2: Grafın Kurulması ve Derlenmesi
    # -------------------------------------------------------------
    print("\n[2/4] Durum Grafı (StateGraph) İnşa Ediliyor ve Derleniyor...")
    workflow = StateGraph()
    workflow.add_node("kodlayici", kodlayici)
    workflow.add_node("denetci", denetci)

    workflow.set_entry_point("kodlayici")
    workflow.add_edge("kodlayici", "denetci")
    workflow.add_conditional_edges(
        "denetci",
        kosullu_yonlendirici,
        {"tekrar": "kodlayici", "tamam": END},
    )

    app = workflow.compile(max_tekrarlama=5)
    print("  ✓ StateGraph Derlendi: Giriş='kodlayici' -> 'denetci' -> [Koşullu Döngü] -> END")

    # -------------------------------------------------------------
    # ADIM 3: Grafın Çalıştırılması ve Durum Günlüğü
    # -------------------------------------------------------------
    print("\n[3/4] Graf Çalıştırılıyor ve Durum Geçişleri İzleniyor...")
    baslangic = AgentState(gorev="Fibonacci dizisini hesaplayan fonksiyon yaz")
    nihai_durum = app.calistir(baslangic)

    print(f"\n  • Toplam Döngü Adımı: {nihai_durum['adim_sayisi']}")
    print(f"  • Test Geçti mi?     : {nihai_durum['test_gecti_mi']}")

    print("\n--- [Graf Çalışma Günlüğü] ---")
    for log in nihai_durum["mesajlar"]:
        print("  " + log)

    print("\n--- [Nihai Üretilen Kod] ---")
    print(nihai_durum.get("kod"))

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli StateGraph Teşhis Panosu Oluşturuluyor...")
    profil_raporu = GraphProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "stategraph_paneli.png")

    GraphGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ StateGraph Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 231 (FAZ 12): GRAF TABANLI AJAN İŞ AKIŞI BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
