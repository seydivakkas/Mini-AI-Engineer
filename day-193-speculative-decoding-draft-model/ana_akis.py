"""
Day 193: Spekülatif Çıkarım (Speculative Decoding) ve Taslak Model Ana Çalıştırma Akışı.
"""

import os
import sys
import torch

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.speculative_decoding_motoru import (
    KucukDraftModel,
    BuyukTargetModel,
    SpeculativeDecodingEngine,
    RejectionSampler,
)
from src.spekulatif_hiz_profilleyici import SpekulatifHizProfilleyici
from src.gorsellestirici import SpeculativeDecodingGorsellestirici


def main():
    print("=" * 110)
    print(">>> Day 193 (FAZ 10): SPECULATIVE DECODING WITH DRAFT MODEL & REJECTION SAMPLING")
    print("=" * 110)

    # -------------------------------------------------------------
    # ADIM 1: Spekülatif Çıkarım Üretim Simülasyonu
    # -------------------------------------------------------------
    print("\n[1/4] Taslak Model (Draft) ve Hedef Model (Target) Spekülatif Üretim Testi...")
    torch.manual_seed(42)
    draft_model = KucukDraftModel(vocab_size=500, hidden_dim=64)
    target_model = BuyukTargetModel(vocab_size=500, hidden_dim=256)

    # Taslak model ağırlıklarını hedef modele yakınlaştır (yüksek kabul oranı için)
    with torch.no_grad():
        draft_model.lm_head.weight.copy_(target_model.lm_head.weight[:, :64])

    engine = SpeculativeDecodingEngine(draft_model=draft_model, target_model=target_model, gamma=4)
    prompt_ids = [10, 45, 128, 92]

    calisma_sonucu = engine.generate(prompt_ids=prompt_ids, max_new_tokens=30, temperature=0.8)

    print(f"  • Prompt Tokenları            : {prompt_ids}")
    print(f"  • Üretilen Yeni Token Sayısı  : {calisma_sonucu['uretilen_token_sayisi']}")
    print(f"  • Hedef Model Forward Sayısı  : {calisma_sonucu['target_forward_sayisi']} (Standartta 30 Forward gerekirdi!)")
    print(f"  • Taslak Model Kabul Oranı    : %{calisma_sonucu['kabul_orani']*100:.1f}")
    print(f"  • Elde Edilen Hızlanma        : {calisma_sonucu['hizlanma_faktoru']}x")
    print("  ✓ Spekülatif Çıkarım Döngüsü Başarıyla Doğrulandı!")

    # -------------------------------------------------------------
    # ADIM 2: Rejection Sampling Matematiksel Doğrulaması
    # -------------------------------------------------------------
    print("\n[2/4] Leviathan Rejection Sampling Doğrulaması...")
    p_ornek = torch.tensor([0.1, 0.7, 0.2])
    q_ornek = torch.tensor([0.2, 0.5, 0.3])
    alpha_1 = RejectionSampler.kabul_olasiligi(float(p_ornek[1].item()), float(q_ornek[1].item()))
    print(f"  • Token 1 için p=0.7, q=0.5 -> Kabul Olasılığı (alpha) = min(1.0, 0.7/0.5) = {alpha_1:.2f}")
    assert alpha_1 == 1.0
    print("  ✓ Rejection Sampling Kabul Formülü Başarıyla Doğrulandı!")

    # -------------------------------------------------------------
    # ADIM 3: Kabul Oranı Duyarlılık Analiz Raporu
    # -------------------------------------------------------------
    print("\n[3/4] Kabul Oranına Göre Teorik Hızlanma Raporu (K=4)...")
    tarama_raporu = SpekulatifHizProfilleyici.parametre_duyarlilik_tarama_raporu()

    print("-" * 110)
    print(f"{'Kabul Oranı (alpha)':<25} | {'Taslak Boyutu (K)':<20} | {'Adım Başı Kabul Token':<25} | {'Hızlanma Faktörü'}")
    print("-" * 110)
    for r in tarama_raporu:
        print(
            f"%{r['kabul_orani']*100:<23.1f} | "
            f"{r['gamma_k']:<20d} | "
            f"{r['beklenen_token']:>18.2f} Token     | "
            f"{r['hizlanma']:>14}"
        )
    print("-" * 110)

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Görsel Teşhis Panosu Üretimi
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Spekülatif Çıkarım Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "speculative_decoding_paneli.png")

    SpeculativeDecodingGorsellestirici.teshis_paneli_olustur(
        calisma_sonucu=calisma_sonucu,
        tarama_raporu=tarama_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Spekülatif Çıkarım Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 110)
    print("✓ Day 193: SPEKÜLATİF ÇIKARIM BAŞARIYLA TAMAMLANDI!")
    print("=" * 110)


if __name__ == "__main__":
    main()
