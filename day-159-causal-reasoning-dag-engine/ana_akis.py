"""
Day 159: Nedensellik Analizi (Causal Inference & Reasoning) Ana Akışı.
Causal DAG, Backdoor Adjustment, Do-Calculus ve Karşıgelişçi Akıl Yürütme.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.nedensel_dag_modeli import NedenselDAGModeli
from src.do_calculus_motoru import DoCalculusMotoru
from src.karsigelisci_akil_yurutucu import KarsigelisciAkilYurutucu
from src.gorsellestirici import NedensellikGorsellestirici


def main():
    print("=" * 105)
    print(">>> Day 159: Causal Inference & Reasoning Engine: Causal DAG, Do-Calculus & Counterfactuals (FAZ 8)")
    print("=" * 105)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # -------------------------------------------------------------
    # 1. NEDENSEL DAG MODELİ VE SEVİYE 1: GÖZLEM (ASSOCIATION)
    # -------------------------------------------------------------
    print("\n[1/4] Causal DAG İnşa Ediliyor & Seviye 1: Gözlemsel Korelasyon Hesaplanıyor...")
    dag = NedenselDAGModeli()
    gozlem = dag.gozlemsel_olasilik_hesapla()

    print(f"  • P(İyileşme | İlaç Aldı)    : P(Y=1 | X=1) = {gozlem['p_y1_given_x1']}")
    print(f"  • P(İyileşme | İlaç Almadı)  : P(Y=1 | X=0) = {gozlem['p_y1_given_x0']}")
    print(f"  >> Gözlemsel Fark (Korelasyon): +%{gozlem['gozlemsel_fark']*100:.1f} (Yaş konfondörü nedeniyle yanıltıcı!)")

    # -------------------------------------------------------------
    # 2. SEVİYE 2: MÜDAHALE (INTERVENTION & DO-CALCULUS)
    # -------------------------------------------------------------
    print("\n[2/4] Seviye 2: Do-Calculus & Backdoor Ayarlaması Uygulanıyor (Intervention)...")
    mudahale = DoCalculusMotoru.mudahale_etkisi_hesapla(dag)

    print(f"  • P(İyileşme | do(İlaç=1))    : P(Y=1 | do(X=1)) = {mudahale['p_y1_do_x1']}")
    print(f"  • P(İyileşme | do(İlaç=0))    : P(Y=1 | do(X=0)) = {mudahale['p_y1_do_x0']}")
    print(f"  >> Gerçek Nedensel Etki (ATE) : +%{mudahale['ortalama_nedensel_etki_ate']*100:.1f} (Net Saf Fayda)")

    # -------------------------------------------------------------
    # 3. SEVİYE 3: KARŞIGELİŞÇİ AKIL YÜRÜTME (COUNTERFACTUALS)
    # -------------------------------------------------------------
    print("\n[3/4] Seviye 3: Karşıgelişçi Sorgu Çözümleniyor (Counterfactual Analysis)...")
    karsigelisci = KarsigelisciAkilYurutucu.karsigelisci_analiz(
        dag, birey_z=0, gerceklesen_x=1, gerceklesen_y=1, karsigelisci_x=0
    )

    print(f"  • Senaryo                  : {karsigelisci['birey_yas_grubu']} bir hasta, {karsigelisci['gerceklesen_tedavi']} ve {karsigelisci['gerceklesen_sonuc']}.")
    print(f"  • Karşıgelişçi Soru        : 'Bu hasta ilacı ALMASAYDI ne olurdu?'")
    print(f"  • Karşıgelişçi İyileşme    : %{karsigelisci['karsigelisci_iyilesme_olasiligi']*100:.1f}")
    print(f"  • Zorunluluk Olasılığı (PN): %{karsigelisci['zorunluluk_olasiligi_pn']*100:.1f} (İyileşmenin doğrudan ilaca bağımlılığı)")

    # -------------------------------------------------------------
    # 4. 6 PANELLİ TEŞHİS PANOSU ÜRETİMİ
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Nedensellik Teşhis Panosu Üretiliyor...")
    gorsellestirici = NedensellikGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "causal_reasoning_dag_paneli.png")
    gorsellestirici.pano_olustur(gozlem, mudahale, karsigelisci, kayit_yolu=cikis_resmi)

    print("\n" + "=" * 105)
    print("✓ Day 159: CAUSAL REASONING & DO-CALCULUS BAŞARIYLA TAMAMLANDI!")
    print("=" * 105)


if __name__ == "__main__":
    main()
