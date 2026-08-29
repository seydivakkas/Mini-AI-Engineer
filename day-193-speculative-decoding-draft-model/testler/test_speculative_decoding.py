"""
PyTest Birim Testleri - Day 193: Spekülatif Çıkarım (Speculative Decoding).
8/8 Kapsamlı Test Paketi.
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.speculative_decoding_motoru import (
    KucukDraftModel,
    BuyukTargetModel,
    RejectionSampler,
    SpeculativeDecodingEngine,
)
from src.spekulatif_hiz_profilleyici import SpekulatifHizProfilleyici
from src.gorsellestirici import SpeculativeDecodingGorsellestirici


def test_kucuk_draft_model_ileri_gecis():
    """1. Taslak model girdi boyutuna göre doğru logit tensörü üretmelidir."""
    model = KucukDraftModel(vocab_size=100, hidden_dim=32)
    inp = torch.tensor([[1, 5, 20]], dtype=torch.long)
    logits = model(inp)
    assert logits.shape == (1, 3, 100)


def test_buyuk_target_model_ileri_gecis():
    """2. Hedef model K aday tokenı tek bir ileri geçişte değerlendirmelidir."""
    model = BuyukTargetModel(vocab_size=100, hidden_dim=64)
    inp = torch.tensor([[1, 5, 20, 30, 40]], dtype=torch.long)
    logits = model(inp)
    assert logits.shape == (1, 5, 100)


def test_rejection_sampling_kabul_kriteri():
    """3. Rejection sampling kabul olasılığı min(1.0, p/q) olarak hesaplanmalıdır."""
    # p > q durumu -> alpha = 1.0
    alpha_high = RejectionSampler.kabul_olasiligi(p_prob=0.8, q_prob=0.4)
    assert alpha_high == 1.0

    # p < q durumu -> alpha = p/q
    alpha_low = RejectionSampler.kabul_olasiligi(p_prob=0.3, q_prob=0.6)
    assert alpha_low == pytest.approx(0.5, abs=1e-3)


def test_artik_dagilim_ornekleme():
    """4. Red durumunda artık dağılım max(0, p-q) üzerinden geçerli bir token indeksi dönmelidir."""
    p = torch.tensor([0.1, 0.7, 0.2])
    q = torch.tensor([0.4, 0.2, 0.4])
    tok = RejectionSampler.artik_dagilimdan_ornekle(p, q)
    assert tok == 1  # p-q sadece indeks 1'de pozitif (0.5)


def test_spekulatif_cikarim_uretim_dongusu():
    """5. Spekülatif çıkarım motoru istenen sayıda tokenı başarıyla üretmelidir."""
    draft = KucukDraftModel(vocab_size=100, hidden_dim=32)
    target = BuyukTargetModel(vocab_size=100, hidden_dim=64)
    engine = SpeculativeDecodingEngine(draft_model=draft, target_model=target, gamma=3)

    sonuc = engine.generate(prompt_ids=[5, 10], max_new_tokens=15)
    assert sonuc["uretilen_token_sayisi"] >= 15
    assert len(sonuc["sonuc_dizisi"]) >= 17


def test_spekulatif_hizlanma_katsayisi():
    """6. Hedef model forward sayısı üretilen token sayısından az olmalıdır."""
    draft = KucukDraftModel(vocab_size=100, hidden_dim=32)
    target = BuyukTargetModel(vocab_size=100, hidden_dim=64)
    engine = SpeculativeDecodingEngine(draft_model=draft, target_model=target, gamma=4)

    sonuc = engine.generate(prompt_ids=[1, 2, 3], max_new_tokens=20)
    assert sonuc["target_forward_sayisi"] < sonuc["uretilen_token_sayisi"]
    assert sonuc["hizlanma_faktoru"] > 1.0


def test_teorik_hizlanma_analizi():
    """7. Hız profilleyicisi Leviathan teoremini doğru hesaplamalıdır."""
    analiz = SpekulatifHizProfilleyici.teorik_hizlanma_analizi(kabul_orani=0.80, gamma=4, draft_target_maliyet_orani=0.08)
    assert analiz["teorik_hizlanma"] > 2.0


def test_gorsellestirme_paneli_olusturma(tmp_path):
    """8. SpeculativeDecodingGorsellestirici 6 panelli teşhis panosunu başarıyla üretmelidir."""
    cikti = str(tmp_path / "test_speculative_paneli.png")
    draft = KucukDraftModel(vocab_size=100, hidden_dim=32)
    target = BuyukTargetModel(vocab_size=100, hidden_dim=64)
    engine = SpeculativeDecodingEngine(draft_model=draft, target_model=target, gamma=3)
    calisma_sonucu = engine.generate(prompt_ids=[1, 2], max_new_tokens=10)
    tarama_raporu = SpekulatifHizProfilleyici.parametre_duyarlilik_tarama_raporu()

    SpeculativeDecodingGorsellestirici.teshis_paneli_olustur(
        calisma_sonucu=calisma_sonucu,
        tarama_raporu=tarama_raporu,
        kayit_yolu=cikti,
    )
    assert os.path.exists(cikti)
    assert os.path.getsize(cikti) > 10000
