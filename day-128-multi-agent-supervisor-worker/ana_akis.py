"""
Day 128: Multi-Agent Supervisor-Worker Mimarisi Ana Akışı.
Hiyerarşik görev dağıtımı, araştırmacı, yazılımcı ve denetçi işçi ajan koordinasyonu ve nihai sentez raporu.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.supervisor_yonetici import SupervisorAjan
from src.gorsellestirici import MultiAgentGorsellestirici


def main():
    print("=" * 105)
    print(">>> Day 128: Multi-Agent Supervisor-Worker Architecture (Researcher, Coder, Reviewer)")
    print("=" * 105)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    hedef_problem = (
        "Dizi içindeki en büyük toplamlı ardışık alt diziyi (Maximum Subarray Sum) bulan "
        "ve tüm elemanların negatif olma sınır durumunu O(1) ek bellek ile O(N) sürede çözen fonksiyonu geliştirin."
    )

    print(f"\n[1/3] Supervisor Görevi Alıyor: '{hedef_problem}'\n")

    supervisor = SupervisorAjan(max_revizyon=3)
    rapor = supervisor.gorevi_orkestre_et(hedef_problem)

    print("=" * 95)
    print(f"{'ADIM':<6} | {'AJAN':<16} | {'İŞLEM':<32} | {'AÇIKLAMA'}")
    print("-" * 95)
    for adim in rapor["adim_gecmisi"]:
        print(f"{adim['adim_no']:<6} | {adim['ajan']:<16} | {adim['islem']:<32} | {adim['cikti_ozeti'][:38]}...")
    print("=" * 95)

    # -------------------------------------------------------------
    # ADIM 2: Nihai Kodun Gerçek Zamanlı Doğrulanması
    # -------------------------------------------------------------
    print("\n[2/3] Üretilen Nihai Python Kodu Çalıştırılıp Sınır Durumlar Doğrulanıyor...")
    print("-" * 65)
    print(rapor["nihai_kod"])
    print("-" * 65)

    yerel_ortam = {}
    exec(rapor["nihai_kod"], yerel_ortam)
    fn = yerel_ortam["max_alt_dizi"]

    test_1 = fn([-2, 1, -3, 4, -1, 2, 1, -5, 4])
    test_2 = fn([-7, -3, -9, -2, -5])
    test_3 = fn([5, 4, -1, 7, 8])

    print(f"  [✓] Test 1 (Karışık Dizi [-2, 1, -3, 4, ...])   : {test_1} (Beklenen: 6)")
    print(f"  [✓] Test 2 (Tamamen Negatif Sınır [-7, -3, -9]): {test_2} (Beklenen: -2)")
    print(f"  [✓] Test 3 (Pozitif Dizi [5, 4, -1, 7, 8])     : {test_3} (Beklenen: 23)")

    assert test_1 == 6 and test_2 == -2 and test_3 == 23
    print("  [✓] Tüm Sınır Durumu Testleri %100 Başarıyla Geçti!")

    # -------------------------------------------------------------
    # ADIM 3: Teşhis Panosu Çizimi
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli Multi-Agent Teşhis Panosu Çiziliyor...")
    karsilastirma = supervisor.benchmark_karsilastir()

    gorsellestirici = MultiAgentGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "multi_agent_supervisor_paneli.png")
    gorsellestirici.pano_olustur(
        calisma_raporu=rapor,
        karsilastirma=karsilastirma,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 105)
    print("[OK] Day 128: MULTI-AGENT SUPERVISOR-WORKER MİMARİSİ BAŞARIYLA TAMAMLANDI!")
    print("=" * 105)


if __name__ == "__main__":
    main()
