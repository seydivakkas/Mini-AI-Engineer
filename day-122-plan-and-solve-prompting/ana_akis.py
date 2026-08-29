"""
Day 122: Plan-and-Solve (PS / PS+) Prompting & Görev Ayrıştırma DAG Mimarisi Ana Akışı.
Karmaşık finansal ve operasyonel problemlerin alt görevlere bölünmesi ve topolojik yürütülmesi.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.plan_and_solve_motoru import PlanAndSolveMotoru
from src.gorsellestirici import PlanAndSolveGorsellestirici


def main():
    print("=" * 105)
    print(">>> Day 122: Plan-and-Solve (PS / PS+) Prompting & Task Decomposition DAG Architecture")
    print("=" * 105)

    # -------------------------------------------------------------
    # ADIM 1: Problem Tanımı ve Motorun Başlatılması
    # -------------------------------------------------------------
    print("\n[1/3] Plan-and-Solve (PS+) Akıl Yürütme ve DAG Motoru Başlatılıyor...")
    motor = PlanAndSolveMotoru()

    problem = (
        "Bir teknoloji şirketinin A yazılım paketi birim satış fiyatı 150 TL, birim lisans/sunucu maliyeti 60 TL, "
        "aylık sabit genel gideri 40,000 TL'dir. Ayda 1,200 adet lisans satıldığında ve %20 kurumlar vergisi "
        "düşüldüğünde şirketin vergi sonrası net karı kaç TL olur?"
    )
    print(f"\n[-] HEDEF PROBLEM:\n'{problem}'\n")

    # -------------------------------------------------------------
    # ADIM 2: Görev Ayrıştırma (Decomposition) ve Topolojik Çözüm
    # -------------------------------------------------------------
    print("[2/3] Problem Alt Görevlere Ayrıştırılıyor (Task Decomposition) ve Topolojik Sırayla Yürütülüyor...")
    rapor = motor.coz(problem, mod="PS+")

    print("\n" + "=" * 90)
    print("                      📊 TOPOLOJİK ALT GÖREV YÜRÜTME ÇİZELGESİ                       ")
    print("=" * 90)
    print(f"{'GÖREV ID':<26} | {'GÖREV TANIMI':<34} | {'HESAPLANAN SONUÇ':<22}")
    print("-" * 90)
    for adim in rapor["adim_kayitlari"]:
        print(f"{adim['id']:<26} | {adim['tanim'][:32]:<34} | {adim['sonuc']:>16,.2f} TL")
    print("-" * 90)

    print(f"\n[✓] NİHAİ DOĞRULANMIŞ SONUÇ : {rapor['nihai_deger']:,.2f} TL")
    print(f"[✓] TOPLAM ÇÖZÜM SÜRESİ     : {rapor['toplam_sure_sn']*1000:.2f} ms ({rapor['sirali_gorev_sayisi']} Alt Görev)")

    # -------------------------------------------------------------
    # ADIM 3: Mimari Kıyaslama ve Teşhis Panosu Çizimi
    # -------------------------------------------------------------
    print("\n[3/3] Zero-Shot CoT vs ReAct vs PS vs PS+ Kıyaslaması ve Teşhis Panosu Çiziliyor...")
    karsilastirma = motor.benchmark_karsilastir()

    print("\n" + "=" * 95)
    print(f"{'YÖNTEM':<24} | {'DOĞRULUK (%)':<14} | {'HESAPLAMA HATASI (%)':<22} | {'EKSİK ADIM (%)':<16}")
    print("-" * 95)
    for y, d, hh, ea in zip(
        karsilastirma["yontemler"],
        karsilastirma["dogruluk_orani"],
        karsilastirma["hesaplama_hatasi"],
        karsilastirma["eksik_adim_atlama"],
    ):
        print(f"{y:<24} | %{d:>11.1f} | %{hh:>19.1f} | %{ea:>13.1f}")
    print("-" * 95)

    gorsellestirici = PlanAndSolveGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ciktilar",
        "plan_and_solve_paneli.png",
    )
    gorsellestirici.pano_olustur(
        cozum_raporu=rapor,
        karsilastirma=karsilastirma,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 105)
    print("[OK] Day 122: PLAN-AND-SOLVE (PS+) GÖREV AYRIŞTIRMA MİMARİSİ BAŞARIYLA TAMAMLANDI!")
    print("=" * 105)


if __name__ == "__main__":
    main()
