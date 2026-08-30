"""
Day 383: Autonomous Drug Discovery & Molecular Dynamics Simulation (MM-PBSA Binding Free Energy)
Copyright (c) 2026 Seydi Eryılmaz (@seydivakkas)
Private - All Rights Reserved

Ana Akış: Protein-Ligand Moleküler Dinamik Simülasyonu, MM-PBSA ve İlaç Adayı Taraması.
"""

import sys
import os

# src yolunu ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from molecular_dynamics_engine import MolecularDynamicsBenchmark
from molecular_profilleyici import MolecularProfilleyici
from molecular_gorsellestirici import MolecularGorsellestirici


def main():
    print("=" * 75)
    print(" DAY 383: OTONOM ILAC KESFI & PROTEIN-LIGAND MOLEKULER DINAMIK SIMULASYONU")
    print("=" * 75)

    # 1. Benchmark Koşumu
    bench = MolecularDynamicsBenchmark()
    print("\n[1/4] 100 Adimli Protein-Ligand MD Simulasyonu ve MM-PBSA Enerjisi Hesaplaniyor...")
    bench_res = bench.kos(num_steps=100)

    b_res = bench_res["binding_free_energy"]
    print(f"  -> MM-PBSA Baglanma Serbest Enerjisi (Delta G): {b_res['delta_g_bind_kcal_mol']:.2f} kcal/mol")
    print(f"  -> Omurga RMSD Sapmasi                     : {bench_res['final_rmsd_angstrom']:.3f} Angstrom")
    print(f"  -> Ortalama MD Sicakligi                   : {bench_res['avg_temp_k']:.1f} K")
    print(f"  -> Lipinski 5 Kurali Uyumu                 : {bench_res['admet_profile']['lipinski_rule_compliant']}")

    # 2. Profilleme
    print("\n[2/4] Ilac Uygunlugu ve Molekuler Dinamik Profillemesi Yapiliyor...")
    profilleyici = MolecularProfilleyici()
    metrics = profilleyici.profille(bench_res)
    rapor_str = profilleyici.rapor_olustur(metrics)
    print(rapor_str)

    # 3. Görselleştirme
    print("[3/4] 6-Panelli Yuksek Cozunurluklu Molekuler Dinamik Teshis Paneli Ciziliyor...")
    gorsellestirici = MolecularGorsellestirici()
    panel_yolu = gorsellestirici.teshis_panelini_ciz(bench_res, metrics)
    print(f"  -> Teshis Paneli Kaydedildi: {panel_yolu}")

    # 4. Tamamlanma
    print("\n[4/4] *** DAY 383: OTONOM ILAC KESFI VE MOLEKULER DINAMIK BASARIYLA TAMAMLANDI! ***")
    print("=" * 75)


if __name__ == "__main__":
    main()
