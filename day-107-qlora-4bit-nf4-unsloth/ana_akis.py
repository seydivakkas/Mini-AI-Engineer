"""
Day 107: QLoRA (NF4 Kuantizasyon, Double Quantization) & Unsloth Autograd Ana Akışı.
VRAM tasarruf analizi, 16-noktalı kuantile doğrulaması ve 6 panelli teşhis panosu.
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

from src.nf4_kuantizasyon import NF4_SEVIYELER
from src.qlora_laboratuvari import QLoRALaboratuvari
from src.gorsellestirici import QLoRAGorsellestirici


def main():
    print("=" * 95)
    print(">>> Day 107: QLoRA (4-bit NormalFloat4, Double Quantization) & Unsloth Fused Autograd")
    print("=" * 95)

    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Çalışma Donanımı: {cihaz.type.upper()}")

    # -------------------------------------------------------------
    # ADIM 1: Laboratuvar Başlatılması ve VRAM Analizi
    # -------------------------------------------------------------
    print("\n[1/3] QLoRA Laboratuvarı Başlatılıyor ve VRAM Ölçeklenmesi Hesaplanıyor...")
    lab = QLoRALaboratuvari(cihaz=cihaz)
    vram_raporu = lab.vram_olceklenme_analizi()

    print("\n--- MODEL ÖLÇEKLERİNE GÖRE EĞİTİM VRAM İHTİYACI (GB) ---")
    print(f"{'MODEL BOYUTU':<15} | {'FULL FT (GB)':<16} | {'FP16 LoRA (GB)':<16} | {'QLoRA (NF4+DQ)':<18} | {'TASARRUF':<12}")
    print("-" * 95)
    for model_adi, degerler in vram_raporu.items():
        print(
            f"{model_adi:<15} | "
            f"{degerler['Full Fine-Tuning (GB)']:>13.1f} GB | "
            f"{degerler['FP16 LoRA (GB)']:>13.1f} GB | "
            f"{degerler['QLoRA (NF4 + DQ) (GB)']:>15.1f} GB | "
            f"%{degerler['Tasarruf Orani (%)']:>9.1f}"
        )
    print("-" * 95)

    # -------------------------------------------------------------
    # ADIM 2: NF4 Kuantizasyon Sadakati ve Autograd Ölçümü
    # -------------------------------------------------------------
    print("\n[2/3] NF4 Kuantizasyon Sadakati ve Hızlı Autograd Ölçülüyor (2048x2048 Matris)...")
    sadakat_raporu = lab.kuantizasyon_sadakati_olc(dim_in=2048, dim_out=2048, block_size=64)

    print("\n[-] NF4 4-BIT KUANTİZASYON KALİTE METRİKLERİ:")
    print(f"  * Ortalama Karesel Hata (MSE) : {sadakat_raporu['mse_kaybi']:.8f}")
    print(f"  * Sinyal-Gürültü Oranı (SNR)  : {sadakat_raporu['snr_db']:.2f} dB")
    print(f"  * Kosinüs Benzerliği (Cosine) : %{sadakat_raporu['kosinus_benzerligi']*100:.3f}")

    hiz_raporu = lab.autograd_ve_hiz_olc(in_features=1024, out_features=1024, batch_size=8, seq_len=128, iterasyon=30)
    print("\n[-] UNSLOTH TARZI FÜZYONLU AUTOGRAD PERFORMANSI:")
    print(f"  * Ortalama İleri+Geri Adım    : {hiz_raporu['ortalama_adim_sure_ms']:.2f} ms")
    print(f"  * Toplam Ana Parametre       : {hiz_raporu['parametre_sayisi_ana']:,}")
    print(f"  * Eğitilebilir LoRA Parametre: {hiz_raporu['parametre_sayisi_lora']:,} (%{hiz_raporu['egitilebilir_parametre_orani_yuzde']:.2f})")

    # -------------------------------------------------------------
    # ADIM 3: 6 Panelli Teşhis Panosu Çizimi
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli QLoRA Teşhis Panosu Çiziliyor...")
    gorsellestirici = QLoRAGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ciktilar",
        "qlora_nf4_unsloth_paneli.png",
    )
    gorsellestirici.pano_olustur(
        vram_raporu,
        sadakat_raporu,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 95)
    print("[OK] Day 107: QLoRA & NF4 Analizleri Başarıyla Tamamlandı!")
    print("=" * 95)


if __name__ == "__main__":
    main()
