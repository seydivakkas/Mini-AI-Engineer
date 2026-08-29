"""
PyTest Birim Testleri - Day 282 (FAZ 15): Meta-Learning (MAML & Meta-SGD).
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.maml_meta_motoru import MAMLEngine, MetaTask
from src.maml_meta_profilleyici import MAMLMetaProfilleyici
from src.gorsellestirici import MAMLGorsellestirici


def test_meta_task_generation():
    """1. MetaTask doğru boyut ve sinüzoid verisi üretmelidir."""
    task = MetaTask(amplitude=2.0, phase=0.5)
    x, y = task.sample_data(10)
    assert x.shape == (10, 1)
    assert y.shape == (10, 1)
    assert np.all(y >= -2.0) and np.all(y <= 2.0)


def test_maml_engine_initialization():
    """2. MAMLEngine başlangıç ağırlıklarını ve Meta-SGD alfa vektörünü doğru boyutlandırmalıdır."""
    engine = MAMLEngine(input_dim=1, hidden_dim=30, output_dim=1, inner_lr=0.01)
    assert engine.w1.shape == (1, 30)
    assert engine.w2.shape == (30, 1)
    assert engine.alpha_w1.shape == (1, 30)
    assert engine.alpha_w2.shape == (30, 1)


def test_forward_pass_and_loss():
    """3. İleri geçiş ve kayıp fonksiyonu geçerli MSE ve gradyan sözlüğü üretmelidir."""
    engine = MAMLEngine()
    x = np.array([[1.0], [2.0]])
    y = np.array([[0.5], [1.0]])
    loss, grads = engine.compute_loss_and_grads(x, y)
    assert loss >= 0.0
    assert "w1" in grads and "w2" in grads
    assert grads["w1"].shape == engine.w1.shape


def test_inner_loop_adaptation():
    """4. İç döngü adaptasyonu support set üzerinde kaybı azaltmalıdır."""
    engine = MAMLEngine()
    task = MetaTask(amplitude=1.5, phase=0.2)
    x_supp, y_supp = task.sample_data(10)
    pre_loss, _ = engine.compute_loss_and_grads(x_supp, y_supp)
    adapted_w, post_loss = engine.adapt_inner_loop(task, k_shots=10, inner_steps=3)
    assert post_loss <= pre_loss + 1e-5


def test_meta_step_outer_loop_update():
    """5. Dış döngü meta-adımı ağırlıkları güncellemeli ve geçerli metrikler üretmelidir."""
    engine = MAMLEngine()
    tasks = [MetaTask(amplitude=1.0, phase=0.1 * i) for i in range(5)]
    w1_before = engine.w1.copy()
    stats = engine.train_meta_step(tasks, k_shots=5, q_queries=10)
    assert not np.array_equal(w1_before, engine.w1)
    assert "pre_adapt_loss" in stats
    assert "post_adapt_loss" in stats


def test_profiler_few_shot_progression():
    """6. Profilleyici shot sayısına göre doğruluğun monoton arttığını (%48.2 -> %94.8) teyit etmelidir."""
    profil = MAMLMetaProfilleyici.basarim_profili_cikar()
    kars = profil["karsilastirma"]
    assert kars["few_shot_dogruluk_yuzde"]["5_Shot_Meta_SGD"] > kars["few_shot_dogruluk_yuzde"]["1_Shot_MAML"]
    assert kars["few_shot_dogruluk_yuzde"]["1_Shot_MAML"] > kars["few_shot_dogruluk_yuzde"]["0_Shot_Naive"]
    assert kars["adaptasyon_mse_kaybi"]["5_Shot_Meta_SGD"] < 0.15


def test_meta_sgd_learning_rate_dimensions():
    """7. Meta-SGD öğrenilebilir alfa parametreleri pozitif olmalıdır."""
    engine = MAMLEngine(inner_lr=0.03)
    assert np.all(engine.alpha_w1 > 0.0)
    assert np.all(engine.alpha_w2 > 0.0)


def test_gorsellestirici_dashboard_creation(tmp_path):
    """8. MAMLGorsellestirici 6 panelli teşhis panosunu başarıyla kaydetmelidir."""
    cikti = str(tmp_path / "test_maml_paneli.png")
    profil = MAMLMetaProfilleyici.basarim_profili_cikar()

    MAMLGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
