"""
Day 147: Test-Time Compute Scaling Yasaları: Çıkarım Zamanı Hesaplama Bütçesi ve Arama Derinliği Ana Akışı.
OpenAI o1 ve Snell et al. scaling yasaları, Pareto sınırı ve bütçe dağıtımı simülasyonu.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.scaling_yasa_modeli import TestTimeScalingModeli
from src.test_time_hesaplayici import TestTimeHesaplayici
from src.pareto_sinir_analizcisi import ParetoSinirAnalizcisi
from src.gorsellestirici import TestTimeScalingGorsellestirici


def main():
    print("=" * 105)
    print(">>> Day 147: Test-Time Compute Scaling Laws & Pareto Efficiency (FAZ 8)")
    print("=" * 105)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # -------------------------------------------------------------
    # 1. POWER-LAW TEST-TIME SCALING TARAMASI
    # -------------------------------------------------------------
    print("\n[1/3] Power-Law Test-Time Compute Scaling Taraması Yapılıyor (N in [1, 4, 16, 64, 256])...")
    model = TestTimeScalingModeli(alfa=0.65, beta=0.42, gama=0.05)
    butceler = [1, 4, 16, 64, 256]
    scaling_verileri = model.butce_taramasi(butceler)

    print("\n" + "-" * 75)
    print(f"{'HESAPLAMA BÜTÇESİ (N)':<25} | {'HATA ORANI':<15} | {'DOĞRULUK ORANI':<15} | {'KATSAYI'}")
    print("-" * 75)
    for s in scaling_verileri:
        print(f"N = {s['butce_n']:<21} | %{s['hata_orani']*100:<13.1f} | %{s['dogruluk_orani']*100:<13.1f} | {s['hesaplama_kat_artisi']}")
    print("-" * 75)

    # -------------------------------------------------------------
    # 2. PARETO VERİMLİLİK VE ARAMA STRATEJİLERİ ANALİZİ
    # -------------------------------------------------------------
    print("\n[2/3] Pareto Sınırı (8B vs 70B) ve Arama Stratejileri Bütçe Dağıtımı Hesaplanıyor...")
    pareto_verileri = ParetoSinirAnalizcisi.pareto_karsilastirmasi()
    butce_analizi = TestTimeHesaplayici.butce_dagitimi_analiz_et(toplam_token_butcesi=4096, adim_basi_token=128)

    print(f"  • Sabit Token Bütçesi      : 4096 Token (Toplam 32 Adım Kapasitesi)")
    print(f"  • Paralel Örnekleme (SC)   : %{butce_analizi['paralel_ornekleme']['tahmini_dogruluk']*100:.1f} (K=32, D=1)")
    print(f"  • Derin Sıralı Zincir      : %{butce_analizi['derin_sirali_arama']['tahmini_dogruluk']*100:.1f} (K=1, D=32)")
    print(f"  • Dengeli MCTS Ağaç Araması: %{butce_analizi['dengeli_agac_aramasi']['tahmini_dogruluk']*100:.1f} (K=5, D=5, En Yüksek Başarım!)")

    s_8b_16x = next(p for p in pareto_verileri if p["model"] == "8B" and p["test_compute"] == 16)
    s_70b_1x = next(p for p in pareto_verileri if p["model"] == "70B" and p["test_compute"] == 1)
    print(f"\n  • Kritik Pareto Bulgusu    : 8B (16x Compute) Doğruluk = %{s_8b_16x['dogruluk']*100:.1f} (Bellek: 16GB)")
    print(f"                                70B (1x Compute) Doğruluk  = %{s_70b_1x['dogruluk']*100:.1f} (Bellek: 140GB)")
    print("                                => Küçük model + Çıkarım hesaplaması, dev modeli alt eder!")

    # -------------------------------------------------------------
    # 3. 6 PANELLİ TEŞHİS PANOSU ÜRETİMİ
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli Test-Time Compute Scaling Teşhis Panosu Üretiliyor...")
    gorsellestirici = TestTimeScalingGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "test_time_compute_scaling_paneli.png")
    gorsellestirici.pano_olustur(
        scaling_verileri=scaling_verileri,
        pareto_verileri=pareto_verileri,
        butce_analizi=butce_analizi,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 105)
    print("✓ Day 147: TEST-TIME COMPUTE SCALING LAWS BAŞARIYLA TAMAMLANDI!")
    print("=" * 105)


if __name__ == "__main__":
    main()
