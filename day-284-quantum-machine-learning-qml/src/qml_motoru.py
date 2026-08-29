"""
Day 284 (FAZ 15): Kuantum Makine Öğrenimi (QML) ve Q-Transformer Motoru.
Parametrik Kuantum Devreleri (VQC), Dolaşıklık (Entanglement) ve Parameter-Shift Gradyanları.
"""

from typing import Dict, Any, Tuple, List
import numpy as np


class QuantumCircuitSimulator:
    """N-Qubit Durum Vektörü Simülatörü."""
    def __init__(self, num_qubits: int = 4):
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits
        self.state = np.zeros(self.dim, dtype=complex)
        self.state[0] = 1.0  # |0000> Başlangıç Durumu

    def reset(self):
        self.state = np.zeros(self.dim, dtype=complex)
        self.state[0] = 1.0

    def apply_ry(self, qubit: int, theta: float):
        """Tek Qubit Ry Rotasyon Kapısı."""
        c = np.cos(theta / 2.0)
        s = np.sin(theta / 2.0)
        u = np.array([[c, -s], [s, c]], dtype=complex)
        self._apply_single_qubit_gate(qubit, u)

    def apply_rz(self, qubit: int, phi: float):
        """Tek Qubit Rz Faz Rotasyon Kapısı."""
        u = np.array([[np.exp(-1j * phi / 2.0), 0], [0, np.exp(1j * phi / 2.0)]], dtype=complex)
        self._apply_single_qubit_gate(qubit, u)

    def apply_cnot(self, control: int, target: int):
        """İki Qubit Dolaşıklık (CNOT) Kapısı."""
        new_state = np.zeros_like(self.state)
        for i in range(self.dim):
            bit_c = (i >> (self.num_qubits - 1 - control)) & 1
            if bit_c == 1:
                # Target bitini ters çevir
                flipped_i = i ^ (1 << (self.num_qubits - 1 - target))
                new_state[flipped_i] = self.state[i]
            else:
                new_state[i] = self.state[i]
        self.state = new_state

    def _apply_single_qubit_gate(self, qubit: int, gate: np.ndarray):
        """Kronecker çarpımı ile tüm durum uzayına tek qubit kapısı uygulama."""
        # Tensör indekslerini yeniden şekillendirip matris çarpımı
        shape = [2] * self.num_qubits
        tensor_state = self.state.reshape(shape)
        tensor_state = np.moveaxis(tensor_state, qubit, 0)
        orig_shape = tensor_state.shape
        tensor_state = tensor_state.reshape((2, -1))
        tensor_state = np.dot(gate, tensor_state)
        tensor_state = tensor_state.reshape(orig_shape)
        tensor_state = np.moveaxis(tensor_state, 0, qubit)
        self.state = tensor_state.flatten()

    def measure_expectation_z0(self) -> float:
        """Qubit 0 üzerindeki Pauli-Z Beklenti Değerini Ölçer: <ψ|Z_0|ψ>."""
        exp_val = 0.0
        for i in range(self.dim):
            prob = np.abs(self.state[i]) ** 2
            bit_0 = (i >> (self.num_qubits - 1)) & 1
            sign = 1.0 if bit_0 == 0 else -1.0
            exp_val += sign * prob
        return float(exp_val)


class QuantumMachineLearningEngine:
    """
    FAZ 15 QML & Q-Transformer Motoru.
    
    Özellikler:
    - Parametrik Kuantum Devresi (Variational Quantum Circuit - VQC)
    - Kuantum Durum Süperpozisyonu ve Dolaşıklık (Entanglement CNOT Mesh)
    - Parameter-Shift Kuralı ile Analitik Kuantum Gradyan Hesabı
    - Kuantum Hilbert Uzayı Dikkat Mekanizması (Q-Self-Attention)
    """

    @classmethod
    def execute_vqc(
        cls,
        inputs: np.ndarray,
        params: np.ndarray,
        num_qubits: int = 4,
    ) -> float:
        """
        VQC İleri Geçişi:
        1. Girdi Kodlama (Data Encoding): inputs -> Ry(x_i)
        2. Parametrik Rotasyon (Ansatz): params -> Rz(θ_i)
        3. Dolaşıklık Katmanı (Entangling Layer): Ring CNOT
        4. Ölçüm: <Z_0>
        """
        sim = QuantumCircuitSimulator(num_qubits=num_qubits)

        # 1. Girdi Kodlama
        for q in range(min(num_qubits, len(inputs))):
            sim.apply_ry(q, float(inputs[q]))

        # 2. Parametrik Katman
        for q in range(min(num_qubits, len(params))):
            sim.apply_rz(q, float(params[q]))

        # 3. Dolaşıklık (CNOT Ring)
        for q in range(num_qubits):
            sim.apply_cnot(q, (q + 1) % num_qubits)

        return sim.measure_expectation_z0()

    @classmethod
    def parameter_shift_gradient(
        cls,
        inputs: np.ndarray,
        params: np.ndarray,
        param_idx: int,
        shift: float = np.pi / 2.0,
        num_qubits: int = 4,
    ) -> float:
        """
        Analitik Parameter-Shift Kuralı:
        d<Z>/dθ_j = 0.5 * [<Z>(θ_j + π/2) - <Z>(θ_j - π/2)]
        """
        params_plus = params.copy()
        params_plus[param_idx] += shift
        exp_plus = cls.execute_vqc(inputs, params_plus, num_qubits=num_qubits)

        params_minus = params.copy()
        params_minus[param_idx] -= shift
        exp_minus = cls.execute_vqc(inputs, params_minus, num_qubits=num_qubits)

        grad = 0.5 * (exp_plus - exp_minus)
        return float(grad)

    @classmethod
    def quantum_self_attention_matrix(
        cls,
        token_embeddings: np.ndarray,
        num_qubits: int = 4,
    ) -> np.ndarray:
        """
        Kuantum Durum Sadakati (Quantum State Fidelity) Tabanlı Dikkat Matrisi:
        A_ij = |<ψ(x_i) | ψ(x_j)>|^2
        """
        seq_len = token_embeddings.shape[0]
        states = []

        for i in range(seq_len):
            sim = QuantumCircuitSimulator(num_qubits=num_qubits)
            for q in range(min(num_qubits, token_embeddings.shape[1])):
                sim.apply_ry(q, float(token_embeddings[i, q]))
            states.append(sim.state)

        attn_matrix = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            for j in range(seq_len):
                inner_prod = np.vdot(states[i], states[j])
                attn_matrix[i, j] = np.abs(inner_prod) ** 2

        # Softmax normalize
        exp_mat = np.exp(attn_matrix)
        attn_norm = exp_mat / np.sum(exp_mat, axis=-1, keepdims=True)
        return attn_norm
