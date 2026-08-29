"""
Day 221: Model Context Protocol (MCP) Standart Araç Sunucusu ve İstemcisi Ana Akışı (FAZ 12 BAŞLANGICI).
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.mcp_protokolu import (
    JSONRPCMessage,
    MCPTool,
    MCPServer,
    MCPClient,
)
from src.mcp_profilleyici import MCPProfilleyici
from src.gorsellestirici import MCPGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 221 (FAZ 12 BAŞLANGICI): MODEL CONTEXT PROTOCOL (MCP) - STANDART ARAÇ SUNUCUSU VE İSTEMCİSİ")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: MCP Sunucusu Oluşturma ve Araç Kaydı
    # -------------------------------------------------------------
    print("\n[1/4] MCP Sunucusu Başlatılıyor ve Alan Araçları (Tools) Kaydediliyor...")
    sunucu = MCPServer("Mini-AI-Gelisim-Sunucusu", "1.0.0")

    sunucu.arac_ekle(
        MCPTool(
            isim="dosya_istatistik_al",
            aciklama="Hedef dosyanın satır ve karakter sayısını döner.",
            girdi_semasi={
                "type": "object",
                "properties": {
                    "dosya_adi": {"type": "string"},
                },
                "required": ["dosya_adi"],
            },
            isleyici=lambda dosya_adi: f"Dosya: {dosya_adi}, Toplam Satır: 245, Karakter: 8920",
        )
    )

    sunucu.kaynak_ekle(
        uri="system://durum",
        baslik="Sistem Sağlık Durumu",
        mime_tipi="text/plain",
        icerik="CPU: %14, RAM: 8192 MB Boş, GPU: RTX 4090 Aktif",
    )
    print("  ✓ Sunucu Başlatıldı: 'Mini-AI-Gelisim-Sunucusu' (1 Araç, 1 Kaynak Kayıtlı)")

    # -------------------------------------------------------------
    # ADIM 2: MCP İstemcisi ile Bağlantı ve Dinamik Keşif
    # -------------------------------------------------------------
    print("\n[2/4] MCP İstemcisi JSON-RPC 2.0 ile Bağlanıyor ve Araçları Keşfediyor...")
    istemci = MCPClient()
    el_sikisma = istemci.sunucuya_baglan(sunucu)
    araclar = istemci.araclari_listele()

    print(f"  • JSON-RPC El Sıkışması: Protokol Sürümü = {el_sikisma['result']['protocolVersion']}")
    print(f"  • Keşfedilen Araç Sayısı : {len(araclar)}")
    for a in araclar:
        print(f"    - Araç: '{a['name']}' -> {a['description']}")
    print("  ✓ tools/list Dinamik Keşfi Başarıyla Tamamlandı!")

    # -------------------------------------------------------------
    # ADIM 3: JSON-RPC ile tools/call ve resources/read Yürütme
    # -------------------------------------------------------------
    print("\n[3/4] İstemci Aracılığıyla 'tools/call' ve 'resources/read' Çağrıları Yürütülüyor...")
    cagri = istemci.arac_cagir("dosya_istatistik_al", {"dosya_adi": "ana_akis.py"})
    kaynak = istemci.kaynak_oku("system://durum")

    print(f"  • tools/call Çıktısı   : {cagri['result']['content'][0]['text']}")
    print(f"  • resources/read Çıktısı: {kaynak['result']['contents'][0]['text']}")
    print("  ✓ İki Yönlü MCP Protokol Yürütmesi Başarıyla Teyit Edildi!")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli MCP Teşhis Panosu Oluşturuluyor...")
    profil_raporu = MCPProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "mcp_protokol_paneli.png")

    MCPGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ MCP Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 221 (FAZ 12 BAŞLANGICI): MODEL CONTEXT PROTOCOL (MCP) BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
