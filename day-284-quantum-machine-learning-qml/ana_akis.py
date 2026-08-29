"""
Day 284 (FAZ 15): Kuantum Makine Öğrenimi (QML) ve Q-Transformer Ana Akış Betiği.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from src.qml_motoru import QuantumCircuitSimulator, QuantumMachineLearningEngine
from src.qml_profilleyici import QMLProfilleyici
from src.gorsellestirici import QMLGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 284 (FAZ 15): KUANTUM MAKİNE ÖĞRENİMİ (QML) — PARAMETRİK KUANTUM DEVRELERİ VE Q-TRANSFORMER")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Kuantum Simülatörü ve Hilbert Durum Uzayı
    # -------------------------------------------------------------
    print("\n[1/4] Kuantum Durum Vektörü Simülatörü Başlatılıyor (4-Qubit Hilbert Uzayı)...")
    sim = QuantumCircuitSimulator(num_qubits=4)
    print(f"  • Qubit Sayısı (N)                   : 4 Qubits")
    print(f"  • Hilbert Durum Uzayı Boyutu (2^N)   : {sim.dim} Karmaşık Genlik (Complex Amplitudes)")
    print(f"  • Başlangıç Durumu                   : |0000> (Normalize Durum Normu: {np.linalg.norm(sim.state):.4f})")

    # -------------------------------------------------------------
    # ADIM 2: VQC ve Parameter-Shift Analitik Gradyan
    # -------------------------------------------------------------
    print("\n[2/4] Parametrik Kuantum Devresi (VQC) İleri Geçişi ve Parameter-Shift Gradyanı...")
    inputs = np.array([0.5, 1.2, 0.8, 0.3])
    params = np.array([0.2, 0.9, 1.5, 0.4])

    exp_val = QuantumMachineLearningEngine.execute_vqc(inputs, params, num_qubits=4)
    grad_0 = QuantumMachineLearningEngine.parameter_shift_gradient(inputs, params, param_idx=0, num_qubits=4)

    print(f"  • Pauli-Z0 Beklenti Ölçümü <Z0>      : {exp_val:.4f} ([-1.0, +1.0] Aralığında)")
    print(f"  • Parameter-Shift Gradyanı (d<Z>/dθ0): {grad_0:.4f} (0.5 * [<Z>(θ+π/2) - <Z>(θ-π/2)])")

    # -------------------------------------------------------------
    # ADIM 3: Q-Self-Attention Matrisi
    # -------------------------------------------------------------
    print("\n[3/4] Kuantum Sadakati Tabanlı Q-Self-Attention Matrisi Hesaplanıyor...")
    tokens = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2],
        [1.3, 1.4, 1.5, 1.6],
    ])
    q_attn = QuantumMachineLearningEngine.quantum_self_attention_matrix(tokens, num_qubits=4)
    profil = QMLProfilleyici.basarim_profili_cikar()
    kars = profil["karsilastirma"]

    print(f"  • Q-Attention Matris Boyutu          : {q_attn.shape} (Sadakat: |<ψ(x_i)|ψ(x_j)>|^2)")
    print(f"  • Sınıflandırma Başarımı             : Klasik Transformer: %{kars['siniflandirma_dogrulugu_yuzde']['Klasik_Transformer']:.1f} -> Q-Transformer: %{kars['siniflandirma_dogrulugu_yuzde']['Q_Transformer_VQC']:.1f}")
    print(f"  • Parametre Tasarrufu                : {kars['parametre_sayisi']['Klasik_Transformer']} -> {kars['parametre_sayisi']['Q_Transformer_VQC']} Parametre ({profil['parametre_tasarrufu_orani']:.0f}x Tasarruf)")
    print(f"  • Dolaşıklık Entropisi (S(ρ))        : {kars['dolasiklik_entropisi']['Q_Transformer_VQC']:.2f} (Çoklu Qubit Dolaşıklığı)")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli QML & Q-Transformer Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "quantum_machine_learning_paneli.png")

    QMLGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ QML & Q-Transformer Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 284 (FAZ 15): KUANTUM MAKİNE ÖĞRENİMİ (QML) VE Q-TRANSFORMER MODÜLÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
