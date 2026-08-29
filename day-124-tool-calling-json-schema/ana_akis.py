"""
Day 124: JSON Schema Destekli Tip Güvenli Tool Calling & Grammar-Constrained Decoding Ana Akışı.
OpenAI standardında şema tanımları, Pydantic benzeri doğrulama ve hata onarımlı araç icrası.
"""

import os
import sys
import json

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.arac_yonlendirici import AracYonlendirici
from src.gramer_kisitlayici import GramerKisitlayici
from src.gorsellestirici import ToolCallingGorsellestirici


def main():
    print("=" * 105)
    print(">>> Day 124: JSON Schema Supported Type-Safe Tool Calling & Grammar-Constrained Decoding")
    print("=" * 105)

    # -------------------------------------------------------------
    # ADIM 1: Araç Yönlendirici ve OpenAI JSON Şemalarının Üretilmesi
    # -------------------------------------------------------------
    print("\n[1/3] Tip Güvenli Araç Yönlendirici (Tool Dispatcher) ve JSON Şemaları Başlatılıyor...")
    yonlendirici = AracYonlendirici()
    openai_araclar = yonlendirici.to_openai_tools()

    print(f"  * Kayıtlı Araç Sayısı: {len(openai_araclar)}")
    print(f"  * Üretilen OpenAI JSON Schema:\n{json.dumps(openai_araclar[0], indent=2, ensure_ascii=False)}")

    # -------------------------------------------------------------
    # ADIM 2: Farklı Senaryolarda Araç Çağrılarının Yürütülmesi
    # -------------------------------------------------------------
    print("\n[2/3] Örnek Tool Calling İstekleri Doğrulanıyor ve Çalıştırılıyor...")

    senaryolar = [
        # Senaryo 1: Geçerli Hisse Senedi Çağrısı
        {
            "ad": "Senaryo 1: Geçerli BIST Hisse Sorgulama",
            "veri": {"name": "HisseSenediSorgula", "arguments": {"sembol": "THYAO", "para_birimi": "TRY"}},
        },
        # Senaryo 2: Geçerli Uçuş Rezervasyonu
        {
            "ad": "Senaryo 2: Geçerli Uçak Bileti Rezervasyonu",
            "veri": {
                "name": "UcusRezervasyonuYap",
                "arguments": {"kalkis": "IST", "varis": "LHR", "yolcu_sayisi": 2, "business_class": True},
            },
        },
        # Senaryo 3: Bozuk JSON Formatı (Markdown çitleri + Tek Tırnak + Trailing Comma)
        {
            "ad": "Senaryo 3: Bozuk JSON (Markdown & Trailing Comma Otomatik Onarımı)",
            "veri": "```json\n{'name': 'HisseSenediSorgula', 'arguments': {'sembol': 'ASELS', 'para_birimi': 'TRY',}}\n```",
        },
        # Senaryo 4: Şema Validasyon Hatası (İzin verilmeyen geçersiz hisse sembolü)
        {
            "ad": "Senaryo 4: Şema Reddi (Geçersiz Enum Değeri)",
            "veri": {"name": "HisseSenediSorgula", "arguments": {"sembol": "GECERSIZ_HISSE_KODU"}},
        },
    ]

    cagri_raporlari = []

    for s in senaryolar:
        print(f"\n--- {s['ad']} ---")
        yanit = yonlendirici.cagir(s["veri"])
        cagri_raporlari.append(yanit)

        if yanit["basarili"]:
            print(f"  [✓] Durum: BAŞARILI | Araç: {yanit['arac_adi']}")
            print(f"  [✓] Çıktı: {json.dumps(yanit['sonuc'], ensure_ascii=False)}")
        else:
            print(f"  [✗] Durum: REDDEDİLDİ | Hata Tipi: {yanit['hata_tipi']}")
            print(f"  [!] Hata Mesajı: {yanit['hata_mesaji']}")

    # -------------------------------------------------------------
    # ADIM 3: Mimari Kıyaslama ve Teşhis Panosu Çizimi
    # -------------------------------------------------------------
    print("\n[3/3] Regex Parsing vs JSON Mode vs JSON Schema vs Grammar-Constrained Kıyaslaması Çiziliyor...")
    kisitlayici = GramerKisitlayici()
    karsilastirma = kisitlayici.benchmark_karsilastir()

    print("\n" + "=" * 105)
    print(f"{'YÖNTEM':<36} | {'GEÇERLİ JSON (%)':<18} | {'TİP HATASI (%)':<16} | {'ONARIM (%)':<14}")
    print("-" * 105)
    for y, g, th, o in zip(
        karsilastirma["yontemler"],
        karsilastirma["json_gecerlilik_orani"],
        karsilastirma["arguman_tip_hatasi"],
        karsilastirma["kendi_kendini_onarim"],
    ):
        print(f"{y:<36} | %{g:>15.1f} | %{th:>13.1f} | %{o:>11.1f}")
    print("-" * 105)

    gorsellestirici = ToolCallingGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ciktilar",
        "tool_calling_paneli.png",
    )
    gorsellestirici.pano_olustur(
        cagri_raporlari=cagri_raporlari,
        karsilastirma=karsilastirma,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 105)
    print("[OK] Day 124: JSON SCHEMA VE TİP GÜVENLİ TOOL CALLING BAŞARIYLA TAMAMLANDI!")
    print("=" * 105)


if __name__ == "__main__":
    main()
