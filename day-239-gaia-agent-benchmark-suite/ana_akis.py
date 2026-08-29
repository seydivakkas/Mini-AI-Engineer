"""
Day 239: GAIA Ajan Benchmark Paketi Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.gaia_benchmark_motoru import (
    GAIATask,
    GAIAEvaluator,
    GAIAAgentHarness,
)
from src.gaia_profilleyici import GAIAProfilleyici
from src.gorsellestirici import GAIAGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 239 (FAZ 12): GAIA (GENERAL AI ASSISTANTS) AJAN BENCHMARK PAKETİ - ÇOK MODLU VE ÇOK ADIMLI TEST")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: GAIA Benchmark Görev Havuzunun Yüklenmesi
    # -------------------------------------------------------------
    print("\n[1/4] GAIA Benchmark Görev Havuzu Yükleniyor (Seviye 1, 2, 3)...")
    harness = GAIAAgentHarness()
    harness.ornek_gaia_havuzu_olustur()
    print(f"  ✓ Toplam {len(harness.gorevler)} Standart GAIA Görevi Yüklendi.")

    # -------------------------------------------------------------
    # ADIM 2: Çok Adımlı Ajan Tahminlerinin Çalıştırılması
    # -------------------------------------------------------------
    print("\n[2/4] Ajan Görevleri İcra Ediyor ve Tahminler Üretiliyor...")
    tahminler = {
        "gaia-101": "3",
        "gaia-102": "4,500,000 USD",
        "gaia-201": "150000.0",
        "gaia-301": "128450.50",
    }

    rapor = harness.degerlendir(tahminler)

    # -------------------------------------------------------------
    # ADIM 3: Değerlendirme ve Hakem Sonuçları
    # -------------------------------------------------------------
    print("\n[3/4] GAIA Hakem Değerlendirmesi ve Seviye Bazlı Karne:")
    print(f"  • Seviye 1 (Basit Arama & PDF)        : %{rapor['seviye_1_basari']:.1f}")
    print(f"  • Seviye 2 (Çok Adımlı Araç Zinciri) : %{rapor['seviye_2_basari']:.1f}")
    print(f"  • Seviye 3 (Karmaşık Otonom İş Akışı): %{rapor['seviye_3_basari']:.1f}")
    print(f"  🏆 GENEL GAIA SKORU                   : %{rapor['genel_gaia_skoru']:.1f}")

    print("\n--- [Ayrıntılı Görev Karnesi] ---")
    for g in rapor["detaylar"]:
        durum_ikon = "✓ DOĞRU" if g.dogru_mu else "✗ YANLIŞ"
        print(f"  [{g.task_id} - Seviye {g.level}] {g.soru[:50]}...")
        print(f"    -> Beklenen: '{g.beklenen_cevap}' | Tahmin: '{g.tahmin_edilen_cevap}' -> {durum_ikon}")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli GAIA Benchmark Teşhis Panosu Oluşturuluyor...")
    profil_raporu = GAIAProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "gaia_paneli.png")

    GAIAGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ GAIA Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 239 (FAZ 12): GAIA AJAN BENCHMARK PAKETİ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
