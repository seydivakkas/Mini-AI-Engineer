"""
PyTest Birim Testleri - Day 284 (FAZ 15): Kuantum Makine Öğrenimi (QML) ve Q-Transformer.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.qml_motoru import QuantumCircuitSimulator, QuantumMachineLearningEngine
from src.qml_profilleyici import QMLProfilleyici
from src.gorsellestirici import QMLGorsellestirici


def test_quantum_circuit_initialization():
    """1. Kuantum simülatörü |0000> normalize durum vektörü oluşturmalıdır."""
    sim = QuantumCircuitSimulator(num_qubits=4)
    assert sim.dim == 16
    assert np.isclose(np.linalg.norm(sim.state), 1.0)
    assert np.isclose(sim.state[0], 1.0)


def test_single_qubit_rotation_gate():
    """2. Ry(pi) kapısı |0> durumunu |1> durumuna döndürmelidir."""
    sim = QuantumCircuitSimulator(num_qubits=1)
    sim.apply_ry(0, np.pi)
    # Beklenti <Z> = -1 olmalıdır (|1> durumu)
    assert np.isclose(sim.measure_expectation_z0(), -1.0, atol=1e-5)


def test_cnot_entanglement_gate():
    """3. CNOT kapısı iki qubit arasında dolaşıklık oluşturmalıdır."""
    sim = QuantumCircuitSimulator(num_qubits=2)
    sim.apply_ry(0, np.pi / 2.0)  # Süperpozisyon
    sim.apply_cnot(0, 1)  # Bell durumu (|00> + |11>)/sqrt(2)
    assert np.isclose(np.abs(sim.state[0]) ** 2, 0.5, atol=1e-5)
    assert np.isclose(np.abs(sim.state[3]) ** 2, 0.5, atol=1e-5)


def test_vqc_execution_expectation():
    """4. VQC ileri geçişi [-1.0, 1.0] aralığında beklenti değeri üretmelidir."""
    inputs = np.array([0.5, 0.2, 0.1, 0.4])
    params = np.array([0.1, 0.8, 0.3, 0.9])
    exp_val = QuantumMachineLearningEngine.execute_vqc(inputs, params, num_qubits=4)
    assert -1.0 <= exp_val <= 1.0


def test_parameter_shift_gradient():
    """5. Parameter-Shift analitik gradyan kuralı sonlu gradyan türetmelidir."""
    inputs = np.array([0.2, 0.3, 0.4, 0.5])
    params = np.array([0.1, 0.2, 0.3, 0.4])
    grad = QuantumMachineLearningEngine.parameter_shift_gradient(inputs, params, param_idx=0, num_qubits=4)
    assert isinstance(grad, float)
    assert not np.isnan(grad)


def test_quantum_self_attention_matrix_properties():
    """6. Kuantum self-attention matrisi satır toplamları 1.0 olan Softmax dağılımı olmalıdır."""
    tokens = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    attn = QuantumMachineLearningEngine.quantum_self_attention_matrix(tokens, num_qubits=4)
    assert attn.shape == (2, 2)
    assert np.allclose(np.sum(attn, axis=-1), 1.0)


def test_profiler_quantum_advantage():
    """7. Profilleyici Q-Transformer parametre tasarrufunu (128x) ve %96.2 doğruluğu doğrulamalıdır."""
    profil = QMLProfilleyici.basarim_profili_cikar()
    kars = profil["karsilastirma"]
    assert kars["siniflandirma_dogrulugu_yuzde"]["Q_Transformer_VQC"] > 94.0
    assert kars["parametre_sayisi"]["Q_Transformer_VQC"] < 50
    assert profil["parametre_tasarrufu_orani"] > 100.0


def test_gorsellestirici_dashboard_creation(tmp_path):
    """8. QMLGorsellestirici 6 panelli teşhis panosunu başarıyla kaydetmelidir."""
    cikti = str(tmp_path / "test_qml_paneli.png")
    profil = QMLProfilleyici.basarim_profili_cikar()

    QMLGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
