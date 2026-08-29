"""
SFT Token Paketleme ve Blok-Diyagonal Maske Testleri (Day 106).
Tüm testler endüstriyel standartlarda %100 PASSED hedefiyle tasarlanmıştır.
"""

import os
import tempfile
import pytest
import torch

from src.token_paketleyici import Ornek, PaketlenmisDizi, olustur_blok_diyagonal_maske, TokenPaketleyici
from src.sft_egitim_motoru import SFTEgitimMotoru
from src.paketleme_laboratuvari import PaketlemeLaboratuvari
from src.gorsellestirici import SFTGorsellestirici


def test_ornek_olusturma_ve_uzunluk():
    """Ornek veri yapısının uzunluk hesabını test eder."""
    o = Ornek(prompt_ids=[1, 2, 3], response_ids=[4, 5])
    assert o.toplam_uzunluk == 5


def test_token_paketleyici_ffd_algoritmasi():
    """TokenPaketleyici FFD algoritmasının verimli paketleme yaptığını test eder."""
    ornekler = [
        Ornek(prompt_ids=[1]*30, response_ids=[2]*20),  # 50 tok
        Ornek(prompt_ids=[3]*40, response_ids=[4]*60),  # 100 tok
        Ornek(prompt_ids=[5]*20, response_ids=[6]*30),  # 50 tok
    ]
    paketleyici = TokenPaketleyici(max_seq_len=200)
    paketlenmis = paketleyici.paketle(ornekler)

    # 50 + 100 + 50 = 200 tok -> Tek bir pakete tam (%100) sığmalı
    assert len(paketlenmis) == 1
    assert paketlenmis[0].doluluk_orani == 1.0
    assert len(paketlenmis[0].ornek_uzunluklari) == 3


def test_prompt_maskeleme_ignore_index():
    """Prompt token'larının -100 ile maskelendiğini doğrular."""
    o = Ornek(prompt_ids=[10, 20], response_ids=[30, 40])
    paketleyici = TokenPaketleyici(max_seq_len=4)
    paketlenmis = paketleyici.paketle([o])

    labels = paketlenmis[0].labels.tolist()
    assert labels[:2] == [-100, -100]
    assert labels[2:] == [30, 40]


def test_pozisyon_id_sifirlama():
    """Her alt-örnek için pozisyon ID'lerinin 0'dan başladığını doğrular."""
    o1 = Ornek(prompt_ids=[1], response_ids=[2])     # len 2
    o2 = Ornek(prompt_ids=[3, 4], response_ids=[5])  # len 3 (FFD'de önce gelir)
    paketleyici = TokenPaketleyici(max_seq_len=5)
    paketlenmis = paketleyici.paketle([o1, o2])

    pos = paketlenmis[0].position_ids.tolist()
    # FFD uzunluğa göre azalan sıraladığı için o2 (len 3) önce, o1 (len 2) sonra paketlenir
    assert pos == [0, 1, 2, 0, 1]


def test_olustur_blok_diyagonal_maske():
    """Blok-diyagonal maskenin örnekler arası sızıntıyı (-inf) engellediğini doğrular."""
    ornek_lens = [3, 2]  # Toplam 5 token
    maske = olustur_blok_diyagonal_maske(ornek_lens, toplam_uzunluk=5, device=torch.device("cpu"))

    assert maske.shape == (5, 5)
    # Örnek 1 içi nedensel: (0,0)->0.0, (1,0)->0.0, (0,1)-> -inf
    assert maske[0, 0] == 0.0
    assert maske[1, 0] == 0.0
    assert maske[0, 1] == float("-inf")

    # Örnek 2 token'ı (i=3) Örnek 1 token'ına (j=0) BAKAMAZ! -> -inf
    assert maske[3, 0] == float("-inf")
    assert maske[3, 1] == float("-inf")
    assert maske[3, 2] == float("-inf")
    # Örnek 2 içi (i=3, j=3) -> 0.0
    assert maske[3, 3] == 0.0


def test_sft_egitim_motoru_ileri_gecis():
    """SFTEgitimMotoru ileri geçiş ve SFT loss hesaplamasını test eder."""
    model = SFTEgitimMotoru(vocab_size=100, dim=64, num_heads=2, num_layers=2, max_seq_len=64)
    inp = torch.randint(1, 99, (2, 16))
    lbl = torch.randint(1, 99, (2, 16))
    lbl[:, :4] = -100  # İlk 4 token prompt maskeli

    logits, loss = model(inp, labels=lbl)
    assert logits.shape == (2, 16, 100)
    assert loss is not None
    assert loss.item() > 0.0


def test_sft_egitim_motoru_paketlenmis_adim():
    """SFTEgitimMotoru paketlenmiş eğitim adımını test eder."""
    model = SFTEgitimMotoru(vocab_size=100, dim=64, num_heads=2, num_layers=2, max_seq_len=32)
    o1 = Ornek(prompt_ids=[1, 2], response_ids=[3, 4])
    o2 = Ornek(prompt_ids=[5, 6], response_ids=[7, 8])
    paketleyici = TokenPaketleyici(max_seq_len=8)
    paket = paketleyici.paketle([o1, o2])[0]

    loss = model.egitim_adimi_paketlenmis(paket)
    assert isinstance(loss, float)
    assert loss > 0.0


def test_gorsellestirici_pano_olusturma():
    """SFTGorsellestirici modülünün 6 panelli teşhis panosu ürettiğini test eder."""
    gorsellestirici = SFTGorsellestirici(dpi=100)
    israf = {
        "standart": {"israf_orani_yuzde": 68.5},
        "token_packing": {"israf_orani_yuzde": 1.2},
    }
    hiz = {
        "Standart Paddingli SFT": {"ornek_saniye": 45.0},
        "Token Packed SFT (FFD)": {"ornek_saniye": 135.0},
    }
    lens = [50, 120, 80, 200, 150]

    with tempfile.TemporaryDirectory() as tmp_dir:
        kayit = os.path.join(tmp_dir, "test_sft_pano.png")
        gorsellestirici.pano_olustur(israf, hiz, lens, kayit_yolu=kayit)
        assert os.path.exists(kayit)
        assert os.path.getsize(kayit) > 1000
