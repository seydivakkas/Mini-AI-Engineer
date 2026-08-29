"""
PyTest Birim Testleri - Day 221: Model Context Protocol (MCP) İstemci ve Sunucu Paketi.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.mcp_protokolu import (
    JSONRPCMessage,
    MCPTool,
    MCPServer,
    MCPClient,
)
from src.mcp_profilleyici import MCPProfilleyici
from src.gorsellestirici import MCPGorsellestirici


def test_jsonrpc_message_formatting():
    """1. JSONRPCMessage standart JSON-RPC 2.0 paketlerini doğru üretmelidir."""
    req = JSONRPCMessage.istek_olustur(1, "tools/list", {"filter": "all"})
    assert req["jsonrpc"] == "2.0"
    assert req["method"] == "tools/list"
    assert req["params"]["filter"] == "all"


def test_mcp_tool_execution():
    """2. MCPTool aracı başarıyla çalıştırıp metin içeriği dönmelidir."""
    tool = MCPTool(
        isim="carp",
        aciklama="İki sayıyı çarpar",
        girdi_semasi={"type": "object"},
        isleyici=lambda a, b: a * b,
    )
    res = tool.calistir({"a": 6, "b": 7})
    assert res["isError"] is False
    assert "42" in res["content"][0]["text"]


def test_mcp_tool_error_handling():
    """3. MCPTool hata fırlatıldığında isError=True dönmelidir."""
    tool = MCPTool(
        isim="bol",
        aciklama="Bölme işlemi",
        girdi_semasi={"type": "object"},
        isleyici=lambda a, b: a / b,
    )
    res = tool.calistir({"a": 10, "b": 0})
    assert res["isError"] is True
    assert "division by zero" in res["content"][0]["text"].lower() or "hata" in res["content"][0]["text"].lower()


def test_mcp_server_initialize():
    """4. MCPServer initialize isteğinde protokol sürümünü ve yeteneklerini bildirmelidir."""
    server = MCPServer("TestServer", "1.0")
    yanit = server.istek_isle(JSONRPCMessage.istek_olustur(1, "initialize"))
    assert "result" in yanit
    assert yanit["result"]["protocolVersion"] == "2024-11-05"


def test_mcp_server_tools_list_and_call():
    """5. MCPServer kayıtlı araçları listelemeli ve çalıştırmalıdır."""
    server = MCPServer()
    server.arac_ekle(MCPTool("topla", "Toplama", {}, lambda a, b: a + b))

    # tools/list
    liste_yanit = server.istek_isle(JSONRPCMessage.istek_olustur(2, "tools/list"))
    assert len(liste_yanit["result"]["tools"]) == 1
    assert liste_yanit["result"]["tools"][0]["name"] == "topla"

    # tools/call
    cagri_yanit = server.istek_isle(
        JSONRPCMessage.istek_olustur(3, "tools/call", {"name": "topla", "arguments": {"a": 10, "b": 25}})
    )
    assert cagri_yanit["result"]["content"][0]["text"] == "35"


def test_mcp_server_resources():
    """6. MCPServer kaynakları doğru listelemeli ve okumalıdır."""
    server = MCPServer()
    server.kaynak_ekle("memo://test", "Not", "text/plain", "Gizli Veri 123")

    yanit = server.istek_isle(JSONRPCMessage.istek_olustur(4, "resources/read", {"uri": "memo://test"}))
    assert yanit["result"]["contents"][0]["text"] == "Gizli Veri 123"


def test_mcp_client_workflow():
    """7. MCPClient istemcisi sunucuya bağlanıp araçları uçtan uca yönetebilmelidir."""
    server = MCPServer("TestHost", "1.0")
    server.arac_ekle(MCPTool("ters_cevir", "String çevirir", {}, lambda metin: metin[::-1]))

    client = MCPClient()
    client.sunucuya_baglan(server)
    araclar = client.araclari_listele()
    assert len(araclar) == 1

    sonuc = client.arac_cagir("ters_cevir", {"metin": "yapayzeka"})
    assert sonuc["result"]["content"][0]["text"] == "akezyapay"


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. MCPGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_mcp_paneli.png")
    profil = MCPProfilleyici.basarim_profili_cikar()

    MCPGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
