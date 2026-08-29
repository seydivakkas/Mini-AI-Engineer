"""
Model Context Protocol (MCP) Standart İstemci-Sunucu Protokol Motoru (Day 221 - FAZ 12).
JSON-RPC 2.0 Araç ve Kaynak Entegrasyon Standardı (Anthropic & Antigravity Uyumlu).
"""

from typing import Dict, Any, List, Optional, Callable
import json


class JSONRPCMessage:
    """JSON-RPC 2.0 Mesaj Yapılandırıcısı."""

    @classmethod
    def istek_olustur(cls, mesaj_id: int, metod: str, parametreler: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """JSON-RPC istek paketi oluşturur."""
        paket = {
            "jsonrpc": "2.0",
            "id": mesaj_id,
            "method": metod,
        }
        if parametreler is not None:
            paket["params"] = parametreler
        return paket

    @classmethod
    def yanit_olustur(cls, mesaj_id: int, sonuc: Any) -> Dict[str, Any]:
        """JSON-RPC başarılı yanıt paketi oluşturur."""
        return {
            "jsonrpc": "2.0",
            "id": mesaj_id,
            "result": sonuc,
        }

    @classmethod
    def hata_olustur(cls, mesaj_id: int, kod: int, mesaj: str) -> Dict[str, Any]:
        """JSON-RPC hata paketi oluşturur."""
        return {
            "jsonrpc": "2.0",
            "id": mesaj_id,
            "error": {
                "code": kod,
                "message": mesaj,
            },
        }


class MCPTool:
    """Tekil bir MCP Aracının Şema ve Çalıştırıcı Tanımı."""

    def __init__(
        self,
        isim: str,
        aciklama: str,
        girdi_semasi: Dict[str, Any],
        isleyici: Callable[..., Any],
    ):
        self.isim = isim
        self.aciklama = aciklama
        self.girdi_semasi = girdi_semasi
        self.isleyici = isleyici

    def sema_bilgisi(self) -> Dict[str, Any]:
        """Araç şema meta verisi döner."""
        return {
            "name": self.isim,
            "description": self.aciklama,
            "inputSchema": self.girdi_semasi,
        }

    def calistir(self, parametreler: Dict[str, Any]) -> Dict[str, Any]:
        """Aracı çalıştırır ve çıktıyı paketler."""
        try:
            cikti = self.isleyici(**parametreler)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": str(cikti),
                    }
                ],
                "isError": False,
            }
        except Exception as e:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Araç Hatası: {str(e)}",
                    }
                ],
                "isError": True,
            }


class MCPServer:
    """Model Context Protocol (MCP) Sunucusu."""

    def __init__(self, sunucu_adi: str = "Mini-AI-MCP-Server", versiyon: str = "1.0.0"):
        self.sunucu_adi = sunucu_adi
        self.versiyon = versiyon
        self._araclar: Dict[str, MCPTool] = {}
        self._kaynaklar: Dict[str, Dict[str, Any]] = {}

    def arac_ekle(self, arac: MCPTool) -> None:
        """Sunucuya yeni bir MCP aracı kaydeder."""
        self._araclar[arac.isim] = arac

    def kaynak_ekle(self, uri: str, baslik: str, mime_tipi: str, icerik: str) -> None:
        """Sunucuya okunabilir bir kaynak (resource) kaydeder."""
        self._kaynaklar[uri] = {
            "uri": uri,
            "name": baslik,
            "mimeType": mime_tipi,
            "text": icerik,
        }

    def istek_isle(self, jsonrpc_istek: Dict[str, Any]) -> Dict[str, Any]:
        """Gelen JSON-RPC 2.0 isteğini ayrıştırır ve yürütür."""
        mesaj_id = jsonrpc_istek.get("id", 1)
        metod = jsonrpc_istek.get("method", "")
        parametreler = jsonrpc_istek.get("params", {})

        if metod == "initialize":
            return JSONRPCMessage.yanit_olustur(
                mesaj_id,
                {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {
                        "name": self.sunucu_adi,
                        "version": self.versiyon,
                    },
                    "capabilities": {
                        "tools": {},
                        "resources": {},
                    },
                },
            )

        elif metod == "tools/list":
            arac_listesi = [a.sema_bilgisi() for a in self._araclar.values()]
            return JSONRPCMessage.yanit_olustur(mesaj_id, {"tools": arac_listesi})

        elif metod == "tools/call":
            arac_adi = parametreler.get("name")
            argumanlar = parametreler.get("arguments", {})
            if arac_adi not in self._araclar:
                return JSONRPCMessage.hata_olustur(mesaj_id, -32601, f"Araç bulunamadı: {arac_adi}")
            sonuc = self._araclar[arac_adi].calistir(argumanlar)
            return JSONRPCMessage.yanit_olustur(mesaj_id, sonuc)

        elif metod == "resources/list":
            kaynak_listesi = [
                {"uri": k["uri"], "name": k["name"], "mimeType": k["mimeType"]}
                for k in self._kaynaklar.values()
            ]
            return JSONRPCMessage.yanit_olustur(mesaj_id, {"resources": kaynak_listesi})

        elif metod == "resources/read":
            uri = parametreler.get("uri")
            if uri not in self._kaynaklar:
                return JSONRPCMessage.hata_olustur(mesaj_id, -32602, f"Kaynak bulunamadı: {uri}")
            return JSONRPCMessage.yanit_olustur(mesaj_id, {"contents": [self._kaynaklar[uri]]})

        else:
            return JSONRPCMessage.hata_olustur(mesaj_id, -32601, f"Bilinmeyen metod: {metod}")


class MCPClient:
    """Model Context Protocol (MCP) İstemcisi."""

    def __init__(self):
        self._bagli_sunucu: Optional[MCPServer] = None
        self._istek_sayaci: int = 1

    def sunucuya_baglan(self, sunucu: MCPServer) -> Dict[str, Any]:
        """Hedef MCP sunucusuna bağlanır ve initialize el sıkışmasını yapar."""
        self._bagli_sunucu = sunucu
        istek = JSONRPCMessage.istek_olustur(self._istek_sayaci, "initialize")
        self._istek_sayaci += 1
        return self._bagli_sunucu.istek_isle(istek)

    def araclari_listele(self) -> List[Dict[str, Any]]:
        """Bağlı sunucudaki tüm araçları keşfeder (tools/list)."""
        if not self._bagli_sunucu:
            raise RuntimeError("Önce bir MCP sunucusuna bağlanmalısınız!")
        istek = JSONRPCMessage.istek_olustur(self._istek_sayaci, "tools/list")
        self._istek_sayaci += 1
        yanit = self._bagli_sunucu.istek_isle(istek)
        return yanit.get("result", {}).get("tools", [])

    def arac_cagir(self, arac_adi: str, argumanlar: Dict[str, Any]) -> Dict[str, Any]:
        """Sunucudaki aracı JSON-RPC ile tetikler (tools/call)."""
        if not self._bagli_sunucu:
            raise RuntimeError("Önce bir MCP sunucusuna bağlanmalısınız!")
        istek = JSONRPCMessage.istek_olustur(
            self._istek_sayaci,
            "tools/call",
            {"name": arac_adi, "arguments": argumanlar},
        )
        self._istek_sayaci += 1
        return self._bagli_sunucu.istek_isle(istek)

    def kaynak_oku(self, uri: str) -> Dict[str, Any]:
        """Sunucudaki statik/dinamik kaynağı çeker (resources/read)."""
        if not self._bagli_sunucu:
            raise RuntimeError("Önce bir MCP sunucusuna bağlanmalısınız!")
        istek = JSONRPCMessage.istek_olustur(self._istek_sayaci, "resources/read", {"uri": uri})
        self._istek_sayaci += 1
        return self._bagli_sunucu.istek_isle(istek)
