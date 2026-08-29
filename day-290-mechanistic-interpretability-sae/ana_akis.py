"""
Day 290 (FAZ 15): Mekanistik Yorumlanabilirlik ve Seyrek Otokodlayıcılar (SAE) Ana Akış Betiği.
Anthropic Monosemanticity, Activation Steering ve Nöral Devre İncelemesi.
"""

import os
import sys
import torch

# UTF-8 Konsol Ayarı (Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.sparse_autoencoder_motoru import SparseAutoencoder, ActivationSteeringEngine
from src.sparse_autoencoder_profilleyici import SAEProfilleyici
from src.gorsellestirici import SAEGorsellestirici


def main():
    print("=" * 115)
    print(">>> Day 290 (FAZ 15): MEKANİSTİK YORUMLANABİLİRLİK — SPARSE AUTOENCODERS (SAE)")
    print("=" * 115)

    # -------------------------------------------------------------
    # ADIM 1: SAE Modelinin Başlatılması
    # -------------------------------------------------------------
    print("\n[1/4] Aşırı Tamamlanmış Seyrek Otokodlayıcı (Overcomplete SAE) Başlatılıyor...")
    d_in = 64
    d_sae = 256
    sae = SparseAutoencoder(d_in=d_in, d_sae=d_sae, l1_coeff=0.005)
    print(f"  • Residual Akım Giriş Boyutu (d_in)   : {d_in} Nöron")
    print(f"  • Seyrek Sözlük Boyutu (d_sae)       : {d_sae} Monosemantic Öznitelik (4x Genişleme)")
    print(f"  • L1 Seyreklik Katsayısı (λ)         : 0.005 (Aşırı Seyrek Ateşleme Hedefi)")

    # -------------------------------------------------------------
    # ADIM 2: Aktivasyonların Kodlanması ve Seyreklik Ölçümü
    # -------------------------------------------------------------
    print("\n[2/4] Residual Akım Aktivasyonları Kodlanıyor ve L0 Seyreklik Analiz Ediliyor...")
    torch.manual_seed(42)
    x_sample = torch.randn(1, d_in)
    x_hat, f, l2_loss, total_loss = sae(x_sample)

    active_features = (f > 1e-4).sum().item()
    print(f"  • Token Başına Aktif Öznitelik Sayısı: {active_features} / {d_sae} (L0 Seyreklik: %{active_features/d_sae*100:.1f})")
    print(f"  • Yeniden İnşa Hatası (L2 MSE)       : {l2_loss.item():.5f}")

    # -------------------------------------------------------------
    # ADIM 3: Karşılaştırmalı Performans Raporu
    # -------------------------------------------------------------
    print("\n[3/4] Ham Nöronlar vs PCA vs Sparse Autoencoder (SAE) Kıyaslama Raporu...")
    profil = SAEProfilleyici.basarim_profili_cikar()
    kars = profil["karsilastirma"]

    print(f"  • 1. Ham Nöronlar Tek Anlamlılık     : %{kars['tek_anlamlilik_safligi_yuzde']['1. Ham Nöronlar']:.1f} (L0 Aktiflik: {kars['l0_aktiflik_sayisi']['1. Ham Nöronlar']:.1f})")
    print(f"  • 2. Klasik PCA Tek Anlamlılık       : %{kars['tek_anlamlilik_safligi_yuzde']['2. Klasik PCA']:.1f} (L0 Aktiflik: {kars['l0_aktiflik_sayisi']['2. Klasik PCA']:.1f})")
    print(f"  • 3. Sparse Autoencoder Tek Anlamlılık: %{kars['tek_anlamlilik_safligi_yuzde']['3. Sparse Autoencoder']:.1f} (L0 Aktiflik: {kars['l0_aktiflik_sayisi']['3. Sparse Autoencoder']:.1f})")
    print(f"  • Tek Anlamlılık Saflık Artışı       : +%{kars['tek_anlamlilik_safligi_yuzde']['3. Sparse Autoencoder'] - kars['tek_anlamlilik_safligi_yuzde']['1. Ham Nöronlar']:.1f}")
    print(f"  • Nöral Müdahale (Activation Steer)  : %{kars['guvenlik_yonlendirme_yuzde']['3. Sparse Autoencoder']:.1f} Hassasiyet")
    print(f"  • Yeniden İnşa Varyans Korunumu (R^2): %{profil['r2_score']:.1f}")

    # -------------------------------------------------------------
    # ADIM 4: 6 Panelli Teşhis Panosu Oluşturma
    # -------------------------------------------------------------
    print("\n[4/4] 6 Panelli Mekanistik Yorumlanabilirlik (SAE) Teşhis Panosu Oluşturuluyor...")
    cikti_yolu = os.path.join(os.path.dirname(__file__), "ciktilar", "mechanistic_interpretability_sae_paneli.png")

    SAEGorsellestirici.teshis_paneli_olustur(
        profil_raporu=profil,
        kayit_yolu=cikti_yolu,
    )
    print(f"  ✓ SAE Teşhis Panosu Başarıyla Kaydedildi: {os.path.abspath(cikti_yolu)}")

    print("\n" + "=" * 115)
    print("✓ Day 290 (FAZ 15): MEKANİSTİK YORUMLANABİLİRLİK VE SEYREK OTOKODLAYICILAR (SAE) TAMAMLANDI!")
    print("=" * 115)


if __name__ == "__main__":
    main()
