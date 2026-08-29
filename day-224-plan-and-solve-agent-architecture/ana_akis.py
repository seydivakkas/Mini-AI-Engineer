"""
Day 224: Plan-and-Solve (PS+) Ajan Mimarisi ve Dinamik Yeniden Planlama Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.plan_and_solve_motoru import (
    SubTask,
    PlannerEngine,
    PlanAndSolveAgent,
)
from src.plan_profilleyici import PlanProfilleyici
from src.gorsellestirici import PlanAndSolveGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 224 (FAZ 12): PLAN-AND-SOLVE (PS+) AJAN MİMARİSİ - STRATEJİK PLANLAMA VE SIRALI İCRA")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Stratejik Planlayıcı (Planner) ile Görev Ayrıştırması
    # -------------------------------------------------------------
    print("\n[1/4] Planlayıcı (Planner) Karmaşık Hedefi Alt Görevlere Ayrıştırıyor...")
    hedef = "3 Bölgenin Çeyrek Satışlarını SQLite'tan Çek, Ortalamayı Hesapla ve Raporla"

    alt_gorevler = [
        ("Bölge satış verilerini sorgula", "SatisVerisiCek", {"tablo": "bolgeler"}),
        ("Bölge satışlarının ortalamasını hesapla", "OrtalamaHesapla", {"ham_veri": "$bellek.gorev_1_sonuc"}),
        ("Yönetici için Markdown raporu derle", "RaporHazirla", {"ortalama": "$bellek.gorev_2_sonuc"}),
    ]

    plan = PlannerEngine.plan_olustur(hedef, alt_gorevler)
    print(f"  • Ana Hedef      : '{hedef}'")
    print(f"  • Planlanan Görev : {len(plan)} Adet Alt Görev")
    for g in plan:
        print(f"    - Görev [{g.gorev_id}]: {g.tanim} (Araç: '{g.arac_adi}')")
    print("  ✓ Stratejik Plan DAG'ı Başarıyla Oluşturuldu!")

    # -------------------------------------------------------------
    # ADIM 2: Çözücü (Solver) ile Sıralı Bellek İcrası
    # -------------------------------------------------------------
    print("\n[2/4] Çözücü (Solver) Ajanı Araçları Bağlıyor ve Planı Yürütüyor...")
    ajan = PlanAndSolveAgent()

    # Araç Kayıtları
    ajan.arac_kaydet("SatisVerisiCek", lambda tablo: "Marmara: 1500, Ege: 900, Akdeniz: 1200")
    ajan.arac_kaydet("OrtalamaHesapla", lambda ham_veri: "1200.0")
    ajan.arac_kaydet("RaporHazirla", lambda ortalama: f"# Bölgesel Satış Raporu\nBölgeler Ortalaması: {ortalama} Bin TL")

    yurutme_sonucu = ajan.plani_yurut(hedef, plan)

    print(f"  • Plan Tamamlandı mı?: {yurutme_sonucu['tamamlandi_mi']}")
    print(f"  • Başarılı Görevler   : {yurutme_sonucu['tamamlanan_alt_gorev']} / {yurutme_sonucu['toplam_alt_gorev']}")
    print("  ✓ Sıralı ve Bellek Enjeksiyonlu İcra Başarıyla Tamamlandı!")

    print("\n--- [Ajan Plan İcra Raporu] ---")
    for r in yurutme_sonucu["plan_raporu"]:
        print("  " + r)

    # -------------------------------------------------------------
    # ADIM 3: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[3/4] 6 Panelli Plan-and-Solve Teşhis Panosu Oluşturuluyor...")
    profil_raporu = PlanProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "plan_and_solve_paneli.png")

    PlanAndSolveGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Plan-and-Solve Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 224 (FAZ 12): PLAN-AND-SOLVE (PS+) AJAN MİMARİSİ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
