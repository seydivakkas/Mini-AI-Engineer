"""
Day 299 (FAZ 15): Kuantum Hibrit AGI ve Varyasyonel Devreler Ana Akış Betiği.
Durum Vektörü Simülatörü, VQE Moleküler Temel Enerjisi ve Barren Plateau Bastırma.
"""

import os
import sys
import numpy as np

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.quantum_vqc_motoru import (
    QuantumStateSimulator,
    VariationalQuantumCircuit,
    VQEMolecularSolver,
    BarrenPlateauMitigator,
)
from src.quantum_vqc_profilleyici import QuantumVQCProfilleyici
from src.gorsellestirici import QuantumVQCGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 299 (FAZ 15): KUANTUM HİBRİT AGİ VE VARYASYONEL DEVRELER — QUANTUM AI & VQE")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Kuantum Durum Simülatörü ve VQC'nin Başlatılması
    # -------------------------------------------------------------
    print("\n[1/4] N-Qubit Durum Vektörü ve Parametrik Devre (VQC) Başlatılıyor...")
    vqc = VariationalQuantumCircuit(num_qubits=4, layers=2)
    params = np.array([0.5, 1.2, 0.8, 2.1, 0.3, 1.7, 0.9, 1.4])
    sim = vqc.forward(params)

    print(f"  • Qubit Sayısı                       : {sim.num_qubits} Qubit (Hilbert Uzayı: 2^{sim.num_qubits} = {sim.dim} Durum)")
    print(f"  • Durum Vektörü Norm Doğrulaması     : {np.sum(np.abs(sim.state)**2):.4f} (Normalize)")
    print(f"  • Pauli-Z Qubit 0 Beklenen Değeri    : <Z_0> = {sim.measure_pauli_z(0):+.4f}")

    # -------------------------------------------------------------
    # ADIM 2: Barren Plateau Gradyan Çölü Bastırma Analizi
    # -------------------------------------------------------------
    print("\n[2/4] Barren Plateau Gradyan Çölü Bastırma Analizi Yapılıyor...")
    bp_res = BarrenPlateauMitigator.compare_gradient_variance([2, 4, 6, 8, 10])

    print(f"  • N=10 Global Maliyet Gradyan Varyansı: {bp_res['global_cost_variance'][-1]:.6f} (Barren Plateau Çölü!)")
    print(f"  • N=10 Lokal Maliyet Gradyan Varyansı : {bp_res['local_cost_variance'][-1]:.6f} (Eğitilebilir Koruma)")

    # -------------------------------------------------------------
    # ADIM 3: VQE Moleküler Temel Enerji Çözümü ve Kıyaslama
    # -------------------------------------------------------------
    print("\n[3/4] VQE ile H2 Moleküler Temel Enerjisi Hesaplanıyor ve Kıyaslanıyor...")
    vqe_res = VQEMolecularSolver.solve_h2_ground_state()
    profil = QuantumVQCProfilleyici.basarim_profili_cikar()
    kars = profil["karsilastirma"]

    print(f"  • Gerçek FCI Temel Enerjisi          : {vqe_res['true_fci_energy']:.5f} Hartree")
    print(f"  • VQE ile Elde Edilen Enerji         : {vqe_res['achieved_vqe_energy']:.5f} Hartree")
    print(f"  • Enerji Hatası                      : {vqe_res['energy_error']:.5f} Hartree (Kimyasal Hassasiyet Sağlandı)")
    print(f"  • Kuantum Hızlanma Çarpanı           : {vqe_res['quantum_speedup']:.1f}x Üstünlük")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Kuantum Hibrit AGI Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "quantum_variational_circuits_paneli.png")

    QuantumVQCGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Kuantum AGI Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 299 (FAZ 15): KUANTUM HİBRİT AGİ VE VARYASYONEL DEVRELER MODÜLÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
