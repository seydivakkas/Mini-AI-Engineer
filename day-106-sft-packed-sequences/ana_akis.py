"""
Day 106: Instruction Supervised Fine-Tuning (SFT) & Token Packing Ana Akışı.
First-Fit Decreasing (FFD) Bin-Packing, Sıfır Padding Kaybı ve 6 Panelli Teşhis Panosu.
"""

import os
import sys
import torch

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.token_paketleyici import Ornek, TokenPaketleyici
from src.sft_egitim_motoru import SFTEgitimMotoru
from src.paketleme_laboratuvari import PaketlemeLaboratuvari
from src.gorsellestirici import SFTGorsellestirici


def main():
    print("=" * 95)
    print(">>> Day 106: Instruction Supervised Fine-Tuning (SFT) & Token Packing (Zero Padding)")
    print("=" * 95)

    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Çalışma Donanımı: {cihaz.type.upper()}")

    # -------------------------------------------------------------
    # ADIM 1: Laboratuvar ve Sentetik SFT Veri Seti
    # -------------------------------------------------------------
    print("\n[1/3] SFT Veri Seti ve Paketleme Laboratuvarı Başlatılıyor...")
    lab = PaketlemeLaboratuvari(max_seq_len=1024, vocab_size=1000, cihaz=cihaz)
    ornekler = lab.sentetik_sft_veri_seti_uret(ornek_sayisi=300)
    ornek_uzunluklari = [o.toplam_uzunluk for o in ornekler]

    print(f"  * Üretilen Sohbet Örneği Sayısı : {len(ornekler)}")
    print(f"  * Ortalama Sohbet Uzunluğu      : {sum(ornek_uzunluklari)/len(ornekler):.1f} Token")
    print(f"  * Min / Maksimum Uzunluk        : {min(ornek_uzunluklari)} / {max(ornek_uzunluklari)} Token")

    # -------------------------------------------------------------
    # ADIM 2: Padding İsrafı ve Throughput Analizi
    # -------------------------------------------------------------
    print("\n[2/3] Standart Padding vs Token Packing İsraf Analizi Yapılıyor...")
    israf_raporu = lab.padding_israf_analizi(ornekler, batch_size=4)

    print("\n--- SFT EĞİTİMİNDE TOKEN İSRAFI VE VERİMLİLİK KARŞILAŞTIRMASI ---")
    print(f"{'METRİK':<35} | {'STANDART PADDING':<22} | {'TOKEN PACKING (FFD)':<22}")
    print("-" * 95)
    print(f"{'Toplam Gerçek Token Sayısı':<35} | {israf_raporu['toplam_gercek_token']:>19,} | {israf_raporu['toplam_gercek_token']:>19,}")
    lbl_gpu = "GPU'da Islenen Toplam Token"
    print(f"{lbl_gpu:<35} | {israf_raporu['standart']['toplam_islenen_token']:>19,} | {israf_raporu['token_packing']['toplam_islenen_token']:>19,}")
    print(f"{'Boşa Giden (Padding) Token':<35} | {israf_raporu['standart']['pad_token_sayisi']:>19,} | {israf_raporu['token_packing']['pad_token_sayisi']:>19,}")
    print(f"{'Padding İsraf Oranı (%)':<35} | %{israf_raporu['standart']['israf_orani_yuzde']:>18.1f} | %{israf_raporu['token_packing']['israf_orani_yuzde']:>18.1f}")
    print(f"{'Gereken Eğitim Adımı (Adım)':<35} | {israf_raporu['standart']['adim_sayisi']:>19,} | {israf_raporu['token_packing']['adim_sayisi']:>19,}")
    print(f"{'Paket Başına Ortalama Doluluk':<35} | {'N/A (Dinamik Pad)':>22} | %{israf_raporu['token_packing']['ortalama_doluluk_yuzde']:>18.1f}")
    print("-" * 95)

    print("\n[>>] Eğitim Modeli Başlatılıyor ve Throughput Ölçülüyor...")
    model = SFTEgitimMotoru(vocab_size=1000, dim=256, num_heads=4, num_layers=4, max_seq_len=1024)
    hiz_raporu = lab.hiz_ve_throughput_karsilastir(ornekler, model, iterasyon=20)

    print("\n" + "=" * 95)
    print(f"{'YÖNTEM':<30} | {'ÖRNEK / SANİYE':<20} | {'TOPLAM SÜRE (s)':<18} | {'HIZLANMA':<15}")
    print("-" * 95)
    for isim, met in hiz_raporu.items():
        hiz_notu = met.get("hizlanma_orani", "1.00x (Referans)")
        print(f"{isim:<30} | {met['ornek_saniye']:>17.1f} | {met['sure_s']:>15.3f} s | {hiz_notu:<15}")
    print("=" * 95)

    # -------------------------------------------------------------
    # ADIM 3: 6 Panelli Teşhis Panosu Çizimi
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli SFT ve Token Packing Teşhis Panosu Çiziliyor...")
    gorsellestirici = SFTGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ciktilar",
        "sft_token_packing_paneli.png",
    )
    gorsellestirici.pano_olustur(
        israf_raporu,
        hiz_raporu,
        ornek_uzunluklari,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 95)
    print("[OK] Day 106: SFT & Token Packing Analizleri Başarıyla Tamamlandı!")
    print("=" * 95)


if __name__ == "__main__":
    main()
