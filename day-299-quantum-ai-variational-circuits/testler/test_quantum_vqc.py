"""
PyTest Birim Testleri - Day 299 (FAZ 15): Kuantum Hibrit AGI ve Varyasyonel Devreler.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.quantum_vqc_motoru import (
    QuantumStateSimulator,
    VariationalQuantumCircuit,
    VQEMolecularSolver,
    BarrenPlateauMitigator,
)
from src.quantum_vqc_profilleyici import QuantumVQCProfilleyici
from src.gorsellestirici import QuantumVQCGorsellestirici


def test_quantum_state_simulator_initialization():
    """1. Durum vektörü simülatörü doğru boyutta ve normalize |00..0> durumunda başlamalıdır."""
    sim = QuantumStateSimulator(num_qubits=3)
    assert sim.dim == 8
    assert np.isclose(np.sum(np.abs(sim.state) ** 2), 1.0)
    assert np.isclose(sim.state[0], 1.0 + 0.0j)


def test_quantum_gate_rotation_operations():
    """2. Ry dönüş kapısı ve Pauli-Z ölçümü beklenen değer üretmelidir."""
    sim = QuantumStateSimulator(num_qubits=1)
    # Ry(pi) uygulayarak |0> durumunu |1> durumuna döndür
    sim.apply_ry(0, np.pi)
    z_exp = sim.measure_pauli_z(0)
    assert np.isclose(z_exp, -1.0, atol=1e-5)


def test_variational_quantum_circuit_forward():
    """3. VQC parametreli devresi geçerli bir kuantum durum simülatörü döndürmelidir."""
    vqc = VariationalQuantumCircuit(num_qubits=3, layers=2)
    params = np.random.uniform(0, 2 * np.pi, size=6)
    sim = vqc.forward(params)
    assert sim.num_qubits == 3
    assert np.isclose(np.sum(np.abs(sim.state) ** 2), 1.0)


def test_vqe_chemical_accuracy():
    """4. VQE H2 molekülü temel enerjisini kimyasal hassasiyet (<1.6 mHa) ile çözmelidir."""
    res = VQEMolecularSolver.solve_h2_ground_state()
    assert res["chemical_accuracy_met"] is True
    assert res["energy_error"] <= 0.0016


def test_barren_plateau_mitigation_variance():
    """5. Lokal maliyet gradyan varyansı N=10 için global maliyetten çok daha yüksek olmalıdır."""
    bp = BarrenPlateauMitigator.compare_gradient_variance([10])
    g_var = bp["global_cost_variance"][0]
    l_var = bp["local_cost_variance"][0]
    assert l_var > g_var * 50.0  # Lokal gradyan çok daha canlı


def test_profiler_quantum_advantage_speedup():
    """6. Kuantum hızlanma çarpanı 20x'in üzerinde olmalıdır."""
    profil = QuantumVQCProfilleyici.basarim_profili_cikar()
    assert profil["kuantum_hizlanma"] >= 20.0


def test_profiler_chemical_precision_superiority():
    """7. Hibrit QNN moleküler hatası klasik MLP'den düşük olmalıdır."""
    profil = QuantumVQCProfilleyici.basarim_profili_cikar()
    q_err = profil["karsilastirma"]["kimyasal_enerji_hatasi_hartree"]["3. Hybrid Local QNN"]
    c_err = profil["karsilastirma"]["kimyasal_enerji_hatasi_hartree"]["1. Classical MLP"]
    assert q_err < c_err


def test_gorsellestirici_dashboard_creation(tmp_path):
    """8. QuantumVQCGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_quantum_paneli.png")
    profil = QuantumVQCProfilleyici.basarim_profili_cikar()

    QuantumVQCGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
