"""
Day 77: Sıfırdan Scaled Dot-Product & Multi-Head Self-Attention Ana Akış Scripti
--------------------------------------------------------------------------------
D_model=64, Kafa=4 boyutlarında Multi-Head Self-Attention modülünü çalıştıran,
dikkat haritalarını, baş entropisini ve gradyan akışını doğrulayan laboratuvar.

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
Özel Lisans - Tüm Hakları Saklıdır.
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.multi_head_attention import CokKafaliOzDikkat
from src.dikkat_analizcisi import DikkatAnalizcisi
from src.gorsellestirici import DikkatGorsellestirici


def tohum_belirle(tohum: int = 42):
    random.seed(tohum)
    np.random.seed(tohum)
    torch.manual_seed(tohum)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tohum)


def main():
    print("=" * 80)
    print("🚀 Day 77: Sıfırdan Scaled Dot-Product & Multi-Head Self-Attention (MHSA)")
    print("=" * 80)

    tohum_belirle(42)
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📌 Çalışma Ortamı Cihazı: {cihaz.upper()}")

    # 1. MHSA Modelinin Kurulması
    b_size = 4
    seq_len = 16
    d_model = 64
    kafa_sayisi = 4
    d_k = d_model // kafa_sayisi

    print(f"\n[1/5] Multi-Head Self-Attention Modülü Başlatılıyor...")
    print(f"  • Girdi Şekli: (Batch={b_size}, Dizi_Uzunlugu={seq_len}, Model_Boyutu={d_model})")
    print(f"  • Kafa Sayısı (H): {kafa_sayisi}, Her Başın Boyutu (d_k): {d_k}")

    mhsa = CokKafaliOzDikkat(
        model_boyutu=d_model,
        kafa_sayisi=kafa_sayisi,
        dropout_orani=0.1,
        bias=True
    ).to(cihaz)

    # 2. Sentetik Dizi Girdisi ve İleri Geçiş (Forward Pass)
    print("\n[2/5] İleri Geçiş (Forward Pass) ve Dikkat Haritaları Çıkarılıyor...")
    x = torch.randn(b_size, seq_len, d_model, device=cihaz, requires_grad=True)
    
    cikti, dikkat_haritalari = mhsa(x)
    print(f"  ✓ Çıktı Tensör Şekli: {list(cikti.shape)} (Beklenen: [{b_size}, {seq_len}, {d_model}])")
    print(f"  ✓ Dikkat Haritaları Şekli: {list(dikkat_haritalari.shape)} (Beklenen: [{b_size}, {kafa_sayisi}, {seq_len}, {seq_len}])")

    # Satır toplamlarının 1.0 (olasılık dağılımı) olduğu güvencesi
    satir_toplamlari = dikkat_haritalari.sum(dim=-1)
    assert torch.allclose(satir_toplamlari, torch.ones_like(satir_toplamlari), atol=1e-5), "Dikkat satırları 1.0 toplamalıdır!"
    print("  ✓ Matematiksel Doğrulama: Softmax satır toplamları = 1.00000 (%100 Tutarlı)")

    # 3. Geriye Yayılım ve Gradyan Akışı Doğrulaması
    print("\n[3/5] Geriye Yayılım (Backpropagation) ve Gradyan Akışı Test Ediliyor...")
    kayip = cikti.sum()
    kayip.backward()

    print(f"  ✓ W_Q Gradyan Normu: {mhsa.w_q.weight.grad.norm().item():.4f}")
    print(f"  ✓ W_K Gradyan Normu: {mhsa.w_k.weight.grad.norm().item():.4f}")
    print(f"  ✓ W_V Gradyan Normu: {mhsa.w_v.weight.grad.norm().item():.4f}")
    print(f"  ✓ W_O Gradyan Normu: {mhsa.w_o.weight.grad.norm().item():.4f}")
    print(f"  ✓ Girdi (x) Gradyan Normu: {x.grad.norm().item():.4f}")

    # 4. Dikkat Mekanizması Derin Analizi
    print("\n[4/5] Dikkat Analizcisi Çalıştırılıyor (Entropi, Mesafe, Ölçekleme Etkisi)...")
    analizci = DikkatAnalizcisi()
    
    entropiler = analizci.hesapla_dikkat_entropisi(dikkat_haritalari)
    mesafeler = analizci.hesapla_dikkat_mesafesi(dikkat_haritalari)
    bas_cesitliligi = analizci.hesapla_baslar_arasi_cesitlilik(dikkat_haritalari)
    olcek_analizi = analizci.olcek_etkisi_analizi(d_k=d_k, seq_len=seq_len)

    print("\n" + "=" * 70)
    print(f"{'Dikkat Başı':^15} | {'Ortalama Entropi (Bit)':^22} | {'Ort. Dikkat Mesafesi':^22}")
    print("=" * 70)
    for h in range(kafa_sayisi):
        print(f"{f'Baş {h+1}':^15} | {entropiler[h]:^22.3f} | {f'{mesafeler[h]:.2f} token':^22}")
    print("-" * 70)
    print(f"📌 Başlar Arası Çeşitlilik Skoru (Kosinüs Mesafesi): {bas_cesitliligi:.4f} / 1.000")
    print(f"📌 1/√d_k Ölçekleme Öncesi Entropi: {olcek_analizi['olceksiz_entropi']:.3f} -> Ölçekleme Sonrası: {olcek_analizi['olcekli_entropi']:.3f}")
    print("=" * 70)

    # 5. 6 Panelli Teşhis Panosu Üretimi
    print("\n[5/5] 6 Panelli Teşhis Panosu Çiziliyor...")
    gorsellestirici = DikkatGorsellestirici()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "self_attention_paneli.png")
    
    gorsellestirici.olustur_teshis_paneli(
        dikkat_haritalari=dikkat_haritalari,
        olcek_analizi=olcek_analizi,
        entropiler=entropiler,
        mesafeler=mesafeler,
        bas_cesitliligi=bas_cesitliligi,
        kayit_yolu=cikti_yolu
    )
    print(f"  ✓ 6 Panelli Teşhis Panosu Kaydedildi: {cikti_yolu}")
    print("\n✅ Day 77: Sıfırdan Self-Attention Mekanizması Başarıyla Tamamlandı!")


if __name__ == "__main__":
    main()
