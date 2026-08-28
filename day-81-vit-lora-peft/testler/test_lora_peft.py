"""
Vision Transformer LoRA PEFT Birim Testleri
-------------------------------------------
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import pytest
import torch
import torch.nn as nn

from src.lora_katmani import LoRADogrusalKatman
from src.lora_enjekte_edici import ViTLoRAEnjekteEdici
from src.minivit_modeli import MiniVisionTransformer


def test_lora_katmani_baslangic_durumu():
    lin = nn.Linear(32, 64)
    lora_lin = LoRADogrusalKatman(lin, r=4, lora_alpha=8.0)

    x = torch.randn(2, 32)
    with torch.no_grad():
        cikis_orig = lin(x)
        cikis_lora = lora_lin(x)

    # Başlangıçta B=0 olduğu için çıktılar birebir eşit olmalıdır
    assert torch.allclose(cikis_orig, cikis_lora, atol=1e-6)
    assert torch.all(lora_lin.lora_B == 0.0)


def test_lora_katmani_gradyan_akisi():
    lin = nn.Linear(32, 64)
    lora_lin = LoRADogrusalKatman(lin, r=4, lora_alpha=8.0)

    x = torch.randn(2, 32)
    out = lora_lin(x)
    loss = out.sum()
    loss.backward()

    # Orijinal katman dondurulmuş olmalı
    assert lora_lin.orijinal_katman.weight.grad is None
    # LoRA matrisleri gradyan almalı
    assert lora_lin.lora_A.grad is not None
    assert lora_lin.lora_B.grad is not None


def test_lora_agirlik_birlestirme_ve_ayirma():
    lin = nn.Linear(16, 16)
    orig_weight = lin.weight.clone()
    lora_lin = LoRADogrusalKatman(lin, r=2, lora_alpha=4.0)

    # B matrisine rastgele ağırlık ata
    lora_lin.lora_B.data.normal_(0, 1)

    x = torch.randn(2, 16)
    cikis_ayrik = lora_lin(x)

    # Ağırlıkları birleştir
    lora_lin.birlestir()
    assert lora_lin.birlestirildi is True
    assert not torch.allclose(lora_lin.orijinal_katman.weight, orig_weight)

    cikis_birlesik = lora_lin(x)
    assert torch.allclose(cikis_ayrik, cikis_birlesik, atol=1e-5)

    # Geri ayır
    lora_lin.ayir()
    assert lora_lin.birlestirildi is False
    assert torch.allclose(lora_lin.orijinal_katman.weight, orig_weight, atol=1e-5)


def test_vit_lora_enjeksiyonu_ve_parametre_oranlari():
    model = MiniVisionTransformer(
        gorsel_boyutu=32,
        yama_boyutu=4,
        sinif_sayisi=10,
        gomulme_boyutu=64,
        derinlik=4,
        kafa_sayisi=4
    )

    enjekte_edici = ViTLoRAEnjekteEdici(hedef_moduller=["w_q", "w_v"], r=4, lora_alpha=8.0)
    model = enjekte_edici.enjekte_et(model)

    istatistikler = enjekte_edici.parametre_istatistikleri(model)

    # 4 blok * 2 katman (wq, wv) = 8 LoRA katmanı
    assert istatistikler["lora_katman_sayisi"] == 8
    # Eğitilebilir oran %5'in altında olmalı (PEFT)
    assert istatistikler["egitilebilir_yuzde"] < 5.0
    assert istatistikler["egitilebilir_param"] < istatistikler["toplam_param"]


def test_vit_lora_ileri_gecis_ve_cikti_boyutu():
    model = MiniVisionTransformer(
        gorsel_boyutu=32,
        yama_boyutu=4,
        sinif_sayisi=10,
        gomulme_boyutu=64,
        derinlik=2,
        kafa_sayisi=2
    )
    enjekte_edici = ViTLoRAEnjekteEdici(hedef_moduller=["w_q", "w_v"], r=4)
    model = enjekte_edici.enjekte_et(model)

    x = torch.randn(3, 3, 32, 32)
    logits = model(x)

    assert logits.shape == (3, 10)


def test_vit_lora_egitimi_sadece_adapter_guncelleme():
    model = MiniVisionTransformer(
        gorsel_boyutu=32,
        yama_boyutu=4,
        sinif_sayisi=10,
        gomulme_boyutu=32,
        derinlik=2,
        kafa_sayisi=2
    )
    enjekte_edici = ViTLoRAEnjekteEdici(hedef_moduller=["w_q", "w_v"], r=2)
    model = enjekte_edici.enjekte_et(model)

    # Orijinal w_k ağırlığını kopyala (LoRA uygulanmamış)
    wk_orig = model.bloklar[0].dikkat.w_k.weight.clone()
    wq_orig = model.bloklar[0].dikkat.w_q.orijinal_katman.weight.clone()

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)

    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([1, 3])
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()

    # Dondurulmuş ana ağırlıklar değişmemeli
    assert torch.allclose(model.bloklar[0].dikkat.w_k.weight, wk_orig)
    assert torch.allclose(model.bloklar[0].dikkat.w_q.orijinal_katman.weight, wq_orig)
    # LoRA B matrisi güncellenmeli (sıfırdan farklı olmalı)
    assert not torch.all(model.bloklar[0].dikkat.w_q.lora_B == 0.0)


def test_birlestir_tum_adapterleri_matematiksel_esitlik():
    model = MiniVisionTransformer(
        gorsel_boyutu=32,
        yama_boyutu=4,
        sinif_sayisi=5,
        gomulme_boyutu=32,
        derinlik=2,
        kafa_sayisi=2
    )
    enjekte_edici = ViTLoRAEnjekteEdici(hedef_moduller=["w_q", "w_v"], r=2)
    model = enjekte_edici.enjekte_et(model)

    # Rastgele ağırlıklar ata
    for katman in enjekte_edici.enjekte_edilen_katmanlar:
        katman.lora_B.data.normal_(0, 0.5)

    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out_ayrik = model(x)
        enjekte_edici.birlestir_tum_adapterleri()
        out_birlesik = model(x)

    assert torch.allclose(out_ayrik, out_birlesik, atol=1e-5)


def test_state_dict_sadece_lora_boyut():
    model = MiniVisionTransformer(
        gorsel_boyutu=32,
        yama_boyutu=4,
        sinif_sayisi=10,
        gomulme_boyutu=32,
        derinlik=2,
        kafa_sayisi=2
    )
    enjekte_edici = ViTLoRAEnjekteEdici(hedef_moduller=["w_q", "w_v"], r=2)
    model = enjekte_edici.enjekte_et(model)

    lora_sd = enjekte_edici.state_dict_sadece_lora(model)
    full_sd = model.state_dict()

    assert len(lora_sd) < len(full_sd)
    for k in lora_sd.keys():
        assert "lora_" in k or "head" in k
