"""
PyTest Birim Testleri - Day 269 (FAZ 14): Medusa / Eagle Çok Başlı Spekülatif Çıkarım.
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.medusa_motoru import (
    MedusaMultiHeadDraftEngine,
    TreeAttentionVerificationKernel,
    MedusaSpeculativeDecoder,
)
from src.medusa_profilleyici import MedusaSpeculativeProfilleyici
from src.gorsellestirici import MedusaSpeculativeGorsellestirici


def test_medusa_head_candidate_prediction():
    """1. Medusa başlıkları her k için top-k aday listesi üretmelidir."""
    engine = MedusaMultiHeadDraftEngine(hidden_dim=64, vocab_size=100, num_heads=4)
    h = np.random.randn(64).astype(np.float32)
    cands = engine.predict_candidates(h, top_k=2)
    assert len(cands) == 4
    for c in cands:
        assert len(c) == 2


def test_medusa_candidate_tree_generation():
    """2. 4 başlık ve top_k=2 için 16 aday dal üretilmelidir."""
    engine = MedusaMultiHeadDraftEngine(num_heads=4)
    cands = [[1, 2], [3, 4], [5, 6], [7, 8]]
    paths = engine.generate_candidate_tree(cands)
    assert len(paths) == 16
    assert len(paths[0]) == 4


def test_tree_attention_mask_properties():
    """3. Tree-Attention maskesinin köşegenleri 1 olmalı ve doğru boyutta olmalıdır."""
    paths = [[1, 3], [1, 4], [2, 5], [2, 6]]
    mask = TreeAttentionVerificationKernel.build_tree_attention_mask(paths)
    assert mask.shape == (4, 4)
    for i in range(4):
        assert mask[i, i] == 1.0


def test_tree_attention_verification_exact_match():
    """4. Tam eşleşmede tüm dal (4 token) kabul edilmelidir."""
    paths = [[10, 20, 30, 40], [10, 20, 30, 41], [10, 21, 30, 40]]
    gt = [10, 20, 30, 40]
    accepted, count, stats = TreeAttentionVerificationKernel.verify_and_accept(paths, gt)
    assert count == 4
    assert accepted == [10, 20, 30, 40]
    assert stats["secilen_dal_indeksi"] == 0


def test_tree_attention_verification_partial_match():
    """5. Kısmi eşleşmede eşleştiği kadar (ör. 2 token) kabul edilmelidir."""
    paths = [[10, 20, 30, 40], [10, 20, 31, 40]]
    gt = [10, 20, 99, 99]
    accepted, count, _ = TreeAttentionVerificationKernel.verify_and_accept(paths, gt)
    assert count == 2
    assert accepted == [10, 20]


def test_tree_attention_verification_fallback_single():
    """6. Sıfır aday eşleşmesinde en az taban modelin 1 tokeni kabul edilmelidir."""
    paths = [[10, 20], [11, 21]]
    gt = [99, 88]
    accepted, count, _ = TreeAttentionVerificationKernel.verify_and_accept(paths, gt)
    assert count == 1
    assert accepted == [99]


def test_medusa_speculative_profiler_output():
    """7. MedusaSpeculativeProfilleyici 3'lü karşılaştırma raporunu üretmelidir."""
    profil = MedusaSpeculativeProfilleyici.basarim_profili_cikar()
    assert "Medusa_Tree_Attention" in profil["karsilastirma"]["cikarim_hizi_tok_s"]
    assert profil["karsilastirma"]["cikarim_hizi_tok_s"]["Medusa_Tree_Attention"] == 68.6


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. MedusaSpeculativeGorsellestirici 6 panelli teşhis panosunu oluşturmalıdır."""
    cikti = str(tmp_path / "test_medusa_paneli.png")
    profil = MedusaSpeculativeProfilleyici.basarim_profili_cikar()

    MedusaSpeculativeGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
