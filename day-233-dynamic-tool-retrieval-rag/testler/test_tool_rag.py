"""
PyTest Birim Testleri - Day 233: Dinamik Araç Geri Getirme (Tool-RAG) Paketi.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tool_rag_motoru import (
    ToolDefinition,
    ToolRegistry,
    SemanticToolRetriever,
    DynamicToolAgent,
)
from src.rag_profilleyici import RAGProfilleyici
from src.gorsellestirici import RAGGorsellestirici


def test_tool_definition_schema():
    """1. ToolDefinition nesnesi JSON şemasını doğru üretmelidir."""
    tool = ToolDefinition("test_tool", "Açıklama", "Cat", {"x": "int"}, ["test"])
    schema = tool.json_sema()
    assert schema["name"] == "test_tool"
    assert schema["category"] == "Cat"


def test_tool_registry_loading():
    """2. ToolRegistry varsayılan kurumsal araçları başarıyla yüklemelidir."""
    reg = ToolRegistry()
    assert len(reg.araclar) >= 15
    assert "get_stock_price" in reg.araclar
    assert "run_sql_query" in reg.araclar


def test_retriever_finance_query():
    """3. SemanticToolRetriever finans sorgusunda 'get_stock_price' ve 'calculate_rsi' getirmelidir."""
    reg = ToolRegistry()
    ret = SemanticToolRetriever(reg)
    res = ret.retrieve_top_k("Tesla hisse senedi fiyatı ve rsi analizi", k=3)
    isimler = [arac.ad for arac, _ in res]
    assert "get_stock_price" in isimler
    assert "calculate_rsi" in isimler


def test_retriever_devops_query():
    """4. SemanticToolRetriever kubernetes sorgusunda 'restart_k8s_pod' getirmelidir."""
    reg = ToolRegistry()
    ret = SemanticToolRetriever(reg)
    res = ret.retrieve_top_k("Kubernetes pod yeniden başlatma", k=2)
    assert res[0][0].ad == "restart_k8s_pod"


def test_retriever_database_query():
    """5. SemanticToolRetriever sql sorgusunda 'run_sql_query' getirmelidir."""
    reg = ToolRegistry()
    ret = SemanticToolRetriever(reg)
    res = ret.retrieve_top_k("Veritabanında SQL sorgusu çalıştır", k=2)
    assert res[0][0].ad == "run_sql_query"


def test_dynamic_tool_agent_plan():
    """6. DynamicToolAgent dinamik plan oluşturmalı ve token tasarrufu hesaplamalıdır."""
    reg = ToolRegistry()
    ret = SemanticToolRetriever(reg)
    agent = DynamicToolAgent(reg, ret)
    plan = agent.planla_ve_sec("Apple hisse fiyatı", top_k=2)
    assert plan["secilen_birincil_arac"] == "get_stock_price"
    assert plan["tasarruf_yuzdesi"] > 50.0


def test_profiler_rag_metrics():
    """7. Profilleyici Tool-RAG seçim doğruluğunun %90 üstünde olduğunu doğrulamalıdır."""
    prof = RAGProfilleyici.basarim_profili_cikar()
    skor = prof["karsilastirma"]["dogru_arac_secim_orani"]["Tool_RAG_Dinamik"]
    assert skor > 90.0


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. RAGGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_rag_paneli.png")
    profil = RAGProfilleyici.basarim_profili_cikar()

    RAGGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
