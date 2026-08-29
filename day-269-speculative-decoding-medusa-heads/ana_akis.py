"""
Day 269 (FAZ 14): Medusa / Eagle Çok Başlı Spekülatif Çıkarım Ana Akışı.
"""

import os
import sys

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from src.medusa_motoru import (
    MedusaMultiHeadDraftEngine,
    TreeAttentionVerificationKernel,
    MedusaSpeculativeDecoder,
)
from src.medusa_profilleyici import MedusaSpeculativeProfilleyici
from src.gorsellestirici import MedusaSpeculativeGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 269 (FAZ 14): MEDUSA & EAGLE — ÇOK BAŞLI SPEKÜLATİF ÇIKARIM VE TREE-ATTENTION DOĞRULAMA ÇEKİRDEĞİ")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: Medusa Çok Başlı Taslak Motorunun Başlatılması
    # -------------------------------------------------------------
    print("\n[1/4] K=4 Medusa MLP Başlıkları Başlatılıyor...")
    draft_engine = MedusaMultiHeadDraftEngine(hidden_dim=128, vocab_size=500, num_heads=4, seed=42)
    hidden_state = np.random.randn(128).astype(np.float32)

    head_cands = draft_engine.predict_candidates(hidden_state, top_k=2)
    print(f"  • Taban Model Gizli Durumu Boyutu   : {hidden_state.shape}")
    print(f"  • Medusa Başlık Sayısı (K)           : {draft_engine.num_heads}")
    for k, cands in enumerate(head_cands):
        print(f"    - Head {k+1} (t+{k+1} Tahminleri)       : Top-2 Adaylar -> {cands}")

    # -------------------------------------------------------------
    # ADIM 2: Ağaç Yapılı Aday Yollarının ve Tree-Attention Maskesinin Üretilmesi
    # -------------------------------------------------------------
    print("\n[2/4] Ağaç Yapılı Aday Yolları (Tree Drafting) ve Dikkat Maskesi Oluşturuluyor...")
    tree_paths = draft_engine.generate_candidate_tree(head_cands)
    tree_mask = TreeAttentionVerificationKernel.build_tree_attention_mask(tree_paths)

    print(f"  • Üretilen Ağaç Dalı Sayısı          : {len(tree_paths)} Aday Yol")
    print(f"  • Tree-Attention Maskesi Boyutu      : {tree_mask.shape} (Ata-Çocuk 1/0 İkili Matrisi)")
    print(f"  • İlk Aday Yol Örneği                : {tree_paths[0]}")

    # -------------------------------------------------------------
    # ADIM 3: Tek İleri Geçişte Tree-Attention Doğrulama ve Kabul
    # -------------------------------------------------------------
    print("\n[3/4] Taban Model Tek İleri Geçişte Tree-Attention ile Doğruluyor...")
    ground_truth_target = head_cands[0][:1] + head_cands[1][:1] + head_cands[2][:1] + [999]  # 3 token eşleşir
    accepted_tokens, accepted_count, stats = TreeAttentionVerificationKernel.verify_and_accept(
        tree_paths, ground_truth_target
    )

    print(f"  • Hedef Gerçek Token Dizisi          : {ground_truth_target}")
    print(f"  • Kabul Edilen Tokenlar              : {accepted_tokens}")
    print(f"  • Tek Adımda Kabul Edilen Token Sayısı: {accepted_count} Token (Klasik AR: 1.0 token/adım)")
    print(f"  • KV-Cache Geri Alma Durumu          : {stats['kv_cache_geri_alma']}")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Medusa Spekülatif Çözme Teşhis Panosu Oluşturuluyor...")
    profil_raporu = MedusaSpeculativeProfilleyici.basarim_profili_cikar()
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "medusa_speculative_paneli.png")

    MedusaSpeculativeGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil_raporu,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ Medusa Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 269 (FAZ 14): MEDUSA & EAGLE SPEKÜLATİF ÇIKARIM MODÜLÜ BAŞARIYLA TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
