"""
Day 142: Açık Akıl Yürütme Akışı (<think> ... </think>), Düşünce Tokenizasyonu ve Self-Consistency Ana Akışı.
Çoklu akıl yürütme yolları üzerinde çoğunluk oylaması (Majority Voting) simülasyonu.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.cot_akil_yurutucu import COTAkilYurutucu
from src.self_consistency_birlestirici import SelfConsistencyBirlestirici
from src.gorsellestirici import COTSelfConsistencyGorsellestirici


def main():
    print("=" * 105)
    print(">>> Day 142: Explicit Chain-of-Thought (<think>) & Self-Consistency (FAZ 8)")
    print("=" * 105)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    yurutucu = COTAkilYurutucu(tohum=42)
    soru_id = "sopave_top"
    soru_metni = "Bir sopa ve bir top toplamda $1.10 tutmaktadır. Sopa, toptan $1.00 daha pahalıdır. Top kaç paradır?"

    # -------------------------------------------------------------
    # ADIM 1: Çoklu Akıl Yürütme Yollarını Örnekleme (K=5)
    # -------------------------------------------------------------
    print(f"\n[1/3] K=5 Bağımsız Akıl Yürütme Yolu Örnekleniyor: '{soru_metni}'")
    orneklenen_yollar = yurutucu.ornekle_coklu_yol(soru_id, soru_metni, k=5, sicaklik=0.7)

    for yol in orneklenen_yollar:
        print(f"\n--- YOL #{yol['yol_no']}: Strateji = {yol['strateji']} | Tahmin = {yol['tahmin']} ---")
        print("  <think>")
        for adim in yol["adimlar"]:
            print(f"    {adim}")
        print("  </think>")
        print(f"  Nihai Yanıt: {yol['nihai_yanit']}")

    # -------------------------------------------------------------
    # ADIM 2: Self-Consistency Çoğunluk Oylaması (Majority Vote)
    # -------------------------------------------------------------
    print("\n[2/3] Self-Consistency Çoğunluk Oylaması (Majority Voting) Yapılıyor...")
    konsensus = SelfConsistencyBirlestirici.birlestir(orneklenen_yollar)

    print(f"  • Toplam Örnek Sayısı (K)  : {konsensus['toplam_oy']}")
    print(f"  • Oy Dağılımı              : {konsensus['oy_dagilimi']}")
    print(f"  • Kazanan Tahmin (Konsensüs): {konsensus['kazanan_tahmin']} ({konsensus['nihai_yanit']})")
    print(f"  • Konsensüs Güven Skoru    : %{konsensus['konsensus_skoru'] * 100.0:.1f}")
    print(f"  • Tespit Edilen Sapan Yol  : {konsensus['sapan_yol_sayisi']} adet")

    # -------------------------------------------------------------
    # ADIM 3: Teşhis Panosu Çizimi
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli Teşhis Panosu Üretiliyor...")
    token_dagilimi = orneklenen_yollar[0]["token_bilgisi"]

    gorsellestirici = COTSelfConsistencyGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "chain_of_thought_special_tokens_paneli.png")
    gorsellestirici.pano_olustur(
        konsensus_sonucu=konsensus,
        orneklenen_yollar=orneklenen_yollar,
        token_dagilimi=token_dagilimi,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 105)
    print("✓ Day 142: CHAIN-OF-THOUGHT & SELF-CONSISTENCY BAŞARIYLA TAMAMLANDI!")
    print("=" * 105)


if __name__ == "__main__":
    main()
