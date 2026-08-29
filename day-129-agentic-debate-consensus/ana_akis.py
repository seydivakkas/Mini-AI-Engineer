"""
Day 129: Multi-Agent Tartışma (Debate) & Konsensüs Oylama Ana Akışı.
Çelişkili mimari kararlarda çok turlu çapraz sorgulama, hakem moderasyonu ve ağırlıklı güven oylaması.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.tartisma_motoru import MultiAgentTartismaMotoru
from src.gorsellestirici import DebateGorsellestirici


def main():
    print("=" * 105)
    print(">>> Day 129: Multi-Agent Debate & Consensus Voting (Judge-Moderated Decision Framework)")
    print("=" * 105)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    tartisma_konusu = (
        "Yüksek Frekanslı Finansal Ödeme ve Bankacılık Altyapısı Mimarisi Seçimi: "
        "Tam Sıfır Güven (Air-Gapped Zero-Trust) vs Küresel Düşük Gecikmeli Edge Serverless vs Hibrit Dengeli Model"
    )

    print(f"\n[1/3] Tartışma Konusu Hakem Tarafından Açılıyor:\n'{tartisma_konusu}'\n")

    motor = MultiAgentTartismaMotoru(max_tur=3)
    sonuc = motor.tartismayi_yurut(tartisma_konusu)

    # -------------------------------------------------------------
    # ADIM 1: Turları ve Argümanları Yazdır
    # -------------------------------------------------------------
    for tur_bilgi in sonuc["tur_kayitlari"]:
        tur_no = tur_bilgi["tur_no"]
        print(f"\n" + "=" * 90)
        print(f"--- TARTIŞMA TURU {tur_no} (Çapraz Sorgulama & Pozisyon Savunması) ---")
        print("=" * 90)
        for arg in tur_bilgi["argumanlar"]:
            print(f"  [{arg['ajan']}] (Güven: %{arg['guven_skoru']*100:.0f})")
            print(f"    -> Tez: \"{arg['tez']}\"")
            print(f"    -> Tercih: {arg['tercih_edilen_secenek']}")

        hakem_rap = tur_bilgi["hakem_raporu"]
        print(f"\n  ⚖️ [HAKEM MODERATÖR RAPORU - Tur {tur_no}]:")
        print(f"    Puanlar: {hakem_rap['ajan_skorlari']}")
        print(f"    Durum  : {hakem_rap['hakem_yorumu']}")

    # -------------------------------------------------------------
    # ADIM 2: Oylama ve Nihai Karar
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print("--- NİHAİ KONSENSÜS OYLAMASI VE HAKEM HÜKMÜ ---")
    print("=" * 90)
    agirlikli = sonuc["agirlikli_oylama"]
    print(f"  [🗳️] Oylama Metodu     : {agirlikli['yontem']}")
    print(f"  [🏆] Kazanan Karar     : {agirlikli['kazanan_secenek']}")
    print(f"  [📊] Güven Yüzdeleri   : {agirlikli['guven_yuzdeleri']}")
    print("\n" + sonuc["nihai_hukum"]["hukum_metni"])

    # -------------------------------------------------------------
    # ADIM 3: Teşhis Panosu Çizimi
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli Agentic Debate Teşhis Panosu Çiziliyor...")
    karsilastirma = motor.benchmark_karsilastir()

    gorsellestirici = DebateGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "agentic_debate_paneli.png")
    gorsellestirici.pano_olustur(
        debate_raporu=sonuc,
        karsilastirma=karsilastirma,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 105)
    print("[OK] Day 129: AGENTIC DEBATE & CONSENSUS VOTING BAŞARIYLA TAMAMLANDI!")
    print("=" * 105)


if __name__ == "__main__":
    main()
