"""
PyTest Birim Testleri - Day 231: Graf Tabanlı Ajan İş Akışı (LangGraph / StateGraph) Paketi.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.stategraph_motoru import (
    AgentState,
    StateGraph,
    CompiledStateGraph,
    START,
    END,
)
from src.graph_profilleyici import GraphProfilleyici
from src.gorsellestirici import GraphGorsellestirici


def test_agent_state_defaults():
    """1. AgentState varsayılan alanları başlatmalı ve log ekleyebilmelidir."""
    st = AgentState(gorev="Test")
    assert st["gorev"] == "Test"
    assert st["mesajlar"] == []
    st.log_ekle("Deneme Log")
    assert len(st["mesajlar"]) == 1


def test_stategraph_node_addition():
    """2. StateGraph düğümleri doğru kaydetmelidir."""
    g = StateGraph()
    g.add_node("node1", lambda s: s)
    assert "node1" in g.dugumler


def test_stategraph_missing_entry_point_error():
    """3. Giriş noktası tanımlanmadığında compile() ValueError fırlatmalıdır."""
    g = StateGraph()
    with pytest.raises(ValueError):
        g.compile()


def test_stategraph_linear_flow():
    """4. StateGraph doğrusal akışı NodeA -> NodeB -> END olarak çalıştırmalıdır."""
    g = StateGraph()
    g.add_node("a", lambda s: {**s, "sayac": s.get("sayac", 0) + 1})
    g.add_node("b", lambda s: {**s, "sayac": s.get("sayac", 0) + 10})
    g.set_entry_point("a")
    g.add_edge("a", "b")
    g.add_edge("b", END)

    app = g.compile()
    res = app.calistir(AgentState())
    assert res["sayac"] == 11


def test_stategraph_conditional_loop_execution():
    """5. StateGraph koşullu kenarlarla başarı sağlanana kadar döngüyü sürdürmelidir."""
    g = StateGraph()

    def adim(state: AgentState) -> AgentState:
        state["tur"] = state.get("tur", 0) + 1
        return state

    g.add_node("adim", adim)
    g.set_entry_point("adim")
    g.add_conditional_edges(
        "adim",
        lambda s: "bitti" if s.get("tur", 0) >= 3 else "tekrar",
        {"tekrar": "adim", "bitti": END},
    )

    app = g.compile()
    res = app.calistir(AgentState())
    assert res["tur"] == 3


def test_stategraph_recursion_limit_guard():
    """6. StateGraph sonsuz döngüde max_tekrarlama sınırını aşmamalıdır."""
    g = StateGraph()
    g.add_node("sonsuz", lambda s: s)
    g.set_entry_point("sonsuz")
    g.add_edge("sonsuz", "sonsuz")

    app = g.compile(max_tekrarlama=4)
    res = app.calistir(AgentState())
    assert res["adim_sayisi"] == 4
    assert any("GÜVENLİK SINIRI" in m for m in res["mesajlar"])


def test_profiler_graph_metrics():
    """7. Profilleyici StateGraph başarısının %90'ın üzerinde olduğunu doğrulamalıdır."""
    prof = GraphProfilleyici.basarim_profili_cikar()
    skor = prof["karsilastirma"]["karmasik_gorev_basarisi"]["StateGraph_LangGraph"]
    assert skor > 90.0


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. GraphGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_graph_paneli.png")
    profil = GraphProfilleyici.basarim_profili_cikar()

    GraphGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
