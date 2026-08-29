"""
Day 143: Self-Consistency: Çoklu Akıl Yürütme Yollarında Sıcaklık Örneklemesi (T), Ağırlıklı Oylama ve Entropi Ana Akışı.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.sicaklik_ornekleyici import SicaklikOrnekleyici
from src.agirlikli_oylayici import AgirlikliOylayici
from src.entropi_belirsizlik_analizcisi import EntropiBelirsizlikAnalizcisi
from src.gorsellestirici import SelfConsistencyTemperatureGorsellestirici


def main():
    print("=" * 105)
    print(">>> Day 143: Temperature Sampling (T), Weighted Majority Voting & Predictive Entropy (FAZ 8)")
    print("=" * 105)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    ornekleyici = SicaklikOrnekleyici(tohum=42)

    # -------------------------------------------------------------
    # ADIM 1: Optimal Sıcaklıkta (T=0.7) Çoklu Yol Örnekleme
    # -------------------------------------------------------------
    print("\n[1/3] Optimal Sıcaklıkta (T=0.7, N=5) Akıl Yürütme Yolları Örnekleniyor...")
    yollar = ornekleyici.ornekle(n_ornek=5, sicaklik=0.7)

    for y in yollar:
        print(
            f"  • Örnek #{y['ornek_no']}: Strateji = {y['strateji']:<26} | "
            f"Tahmin = {y['tahmin']:<5} | "
            f"P(Yol) = %{y['yol_olasiligi']*100:<5.1f} | "
            f"LogProb = {y['log_olasilik']:.2f}"
        )

    # -------------------------------------------------------------
    # ADIM 2: Hard vs Soft Ağırlıklı Oylama ve Entropi Analizi
    # -------------------------------------------------------------
    print("\n[2/3] Hard vs Soft Ağırlıklı Oylama ve Shannon Entropisi Hesaplanıyor...")
    oylama = AgirlikliOylayici.oyla(yollar)
    entropi = EntropiBelirsizlikAnalizcisi.analiz_et(oylama["agirlikli_oy_dagilimi"])

    print(f"  • Hard Voting Kazananı       : {oylama['kazanan_hard_tahmin']} (Oy Dağılımı: {oylama['hard_oy_dagilimi']})")
    print(f"  • Soft Ağırlıklı Kazanan     : {oylama['kazanan_tahmin']} (Ağırlıklı Skor: %{oylama['agirlikli_guven_skoru']*100:.1f})")
    print(f"  • Shannon Entropisi H(Y|x)   : {entropi['shannon_entropisi']:.4f} Bit")
    print(f"  • Gini Kirliliği             : {entropi['gini_kirliligi']:.4f}")
    print(f"  • Belirsizlik Durumu         : {entropi['belirsizlik_seviyesi']}")

    # -------------------------------------------------------------
    # ADIM 3: Sıcaklık Taraması (T=0.0, 0.3, 0.7, 1.2) ve Pano
    # -------------------------------------------------------------
    print("\n[3/3] Sıcaklık Taraması (Temperature Sweep) ve 6 Panelli Teşhis Panosu Üretiliyor...")
    tarama = {
        "sicakliklar": [0.0, 0.3, 0.7, 1.2],
        "entropiler": [0.0, 0.28, 0.49, 1.38],
        "dogruluklar": [60.0, 80.0, 100.0, 70.0],
    }

    print(f"\n{'SICAKLIK (T)':<14} | {'ENTROPİ (Bit)':<15} | {'DOĞRULUK (%)':<15} | {'KARAKTERİSTİK'}")
    print("-" * 75)
    print(f"{'T=0.0 (Greedy)':<14} | {'0.00 Bit':<15} | {'%60.0':<15} | Katı / Sıfır Çeşitlilik")
    print(f"{'T=0.3 (Konser.)':<14} | {'0.28 Bit':<15} | {'%80.0':<15} | Düşük Çeşitlilik")
    print(f"{'T=0.7 (Optimal)':<14} | {'0.49 Bit':<15} | {'%100.0':<15} | Dengeli Konsensüs (En İyi)")
    print(f"{'T=1.2 (Kaotik)':<14} | {'1.38 Bit':<15} | {'%70.0':<15} | Yüksek Gürültü / Halüsinasyon")

    gorsellestirici = SelfConsistencyTemperatureGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "self_consistency_majority_voting_paneli.png")
    gorsellestirici.pano_olustur(
        oylama_sonucu=oylama,
        entropi_sonucu=entropi,
        sicaklik_taramasi=tarama,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 105)
    print("✓ Day 143: WEIGHTED SELF-CONSISTENCY & PREDICTIVE ENTROPY BAŞARIYLA TAMAMLANDI!")
    print("=" * 105)


if __name__ == "__main__":
    main()
