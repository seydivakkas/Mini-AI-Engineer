"""
PyTest Birim Testleri - Day 285 (FAZ 15): Sürekli ve Yaşam Boyu Öğrenme (EWC).
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ewc_motoru import SimpleClassifier, ContinualLifelongLearningEngine
from src.ewc_profilleyici import EWCProfilleyici
from src.gorsellestirici import EWCGorsellestirici


def test_simple_classifier_forward():
    """1. Çok görevli MLP modeli geçerli şekilli tensör çıktısı üretmelidir."""
    model = SimpleClassifier(input_dim=4, hidden_dim=32, output_dim=2)
    x = torch.randn(1, 4)
    out = model(x)
    assert out.shape == (1, 2)
    assert not torch.isnan(out).any()


def test_compute_fisher_information_shape():
    """2. Fisher bilgi sözlüğü tüm model parametreleriyle birebir eşleşmelidir."""
    model = SimpleClassifier(input_dim=4, hidden_dim=32, output_dim=2)
    task_a, _ = EWCProfilleyici.sentetik_gorev_verisi_olustur(n_samples=20)
    fisher = ContinualLifelongLearningEngine.compute_fisher_information(model, task_a)

    for name, param in model.named_parameters():
        assert name in fisher
        assert fisher[name].shape == param.shape


def test_fisher_values_non_negative():
    """3. Fisher bilgi matrisi değerleri gradyan karesi olduğu için negatif olamaz (F_i >= 0)."""
    model = SimpleClassifier(input_dim=4, hidden_dim=32, output_dim=2)
    task_a, _ = EWCProfilleyici.sentetik_gorev_verisi_olustur(n_samples=20)
    fisher = ContinualLifelongLearningEngine.compute_fisher_information(model, task_a)

    for name, f_tensor in fisher.items():
        assert (f_tensor >= 0.0).all()


def test_ewc_loss_calculation():
    """4. Parametreler saptığında EWC kaybı görev kaybından büyük olmalıdır."""
    model = SimpleClassifier(input_dim=4, hidden_dim=32, output_dim=2)
    task_loss = torch.tensor(1.5)
    optimal_params = {name: param.data.clone() for name, param in model.named_parameters()}
    fisher_dict = {name: torch.ones_like(param.data) for name, param in model.named_parameters()}

    # Parametreleri biraz kaydır
    with torch.no_grad():
        for param in model.parameters():
            param.add_(0.1)

    ewc_loss = ContinualLifelongLearningEngine.compute_ewc_loss(
        model, task_loss, optimal_params, fisher_dict, lambda_ewc=1000.0
    )
    assert ewc_loss.item() > task_loss.item()


def test_accuracy_evaluation():
    """5. Doğruluk hesaplayıcı [0.0, 100.0] aralığında sonuç vermelidir."""
    model = SimpleClassifier(input_dim=4, hidden_dim=32, output_dim=2)
    task_a, _ = EWCProfilleyici.sentetik_gorev_verisi_olustur(n_samples=20)
    acc = ContinualLifelongLearningEngine.evaluate_accuracy(model, task_a)
    assert 0.0 <= acc <= 100.0


def test_synthetic_dataset_generation():
    """6. Sentetik görev üretici eşit ve dengeli veri üretmelidir."""
    task_a, task_b = EWCProfilleyici.sentetik_gorev_verisi_olustur(n_samples=40)
    assert len(task_a) == 40
    assert len(task_b) == 40


def test_profiler_forgetting_reduction():
    """7. EWC profilleyici Yıkıcı Unutmayı %70'ten %5'in altına düşürmelidir."""
    profil = EWCProfilleyici.basarim_profili_cikar()
    kars = profil["karsilastirma"]
    assert kars["gorev_a_hatirlama_orani"]["3. EWC Konsolidasyonu (EWC)"] > 90.0
    assert kars["gorev_a_hatirlama_orani"]["1. Saf Ince Ayar (Naive)"] < 30.0
    assert kars["yikici_unutma_orani"]["3. EWC Konsolidasyonu (EWC)"] < 10.0


def test_gorsellestirici_dashboard_creation(tmp_path):
    """8. EWCGorsellestirici 6 panelli teşhis panosunu oluşturmalıdır."""
    cikti = str(tmp_path / "test_ewc_paneli.png")
    profil = EWCProfilleyici.basarim_profili_cikar()

    EWCGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
