"""
Day 163: Görsel Komut İnce Ayarı (Visual Instruction Tuning / Visual SFT) Ana Akışı.
Kayıp Maskeleme (Target-Only Loss Masking -100) ve Çok Turlu Görsel Sohbet Eğitimi.
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

from src.gorsel_komut_veri_seti import GorselKomutVeriSeti
from src.vlm_model import HafifVLM
from src.visual_sft_egitici import VisualSFTEgitici
from src.gorsellestirici import VisualSFTGorsellestirici


def main():
    print("=" * 110)
    print(">>> Day 163 (FAZ 9): VISUAL INSTRUCTION TUNING: TARGET-ONLY LOSS MASKING & MULTI-TURN VISUAL CHAT")
    print("=" * 110)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # -------------------------------------------------------------
    # 1. VERİ SETİ VE MODEL YÜKLEME
    # -------------------------------------------------------------
    print("\n[1/3] LLaVA-Instruct Formatında Görsel Komut Veri Seti Yükleniyor...")
    veriler = GorselKomutVeriSeti.ornek_verileri_getir()
    for v in veriler:
        print(f"  • [{v['kategori']:<18}] ({v['goruntu_adi']:<18}) : {v['diyalog'][0]['metin'][:45]}...")

    print("\n[2/3] Görsel Dil Modeli (HafifVLM) ve SFT Eğitici Hazırlanıyor...")
    vlm = HafifVLM(
        d_vision=768,
        d_text=512,
        vocab_size=1000,
        num_patches=256,
        katman_sayisi=3,
        kafa_sayisi=8,
    )

    # -------------------------------------------------------------
    # 2. GÖRSEL SFT EĞİTİM DÖNGÜSÜ (Target-Only Loss Masking)
    # -------------------------------------------------------------
    print("  • Visual SFT Adımları Yürütülüyor (256 Görsel + Prompt Maskesi: -100)...")
    rapor = VisualSFTEgitici.egitim_dongusu_yurut(vlm, adim_sayisi=5, ogrenme_orani=5e-4)

    print("\n" + "-" * 80)
    print(f"{'Metrik':<35} | {'Değer'}")
    print("-" * 80)
    print(f"{'Başlangıç Kaybı (Step 1)':<35} | {rapor['baslangic_kaybi']}")
    print(f"{'Bitiş Kaybı (Step 5)':<35} | {rapor['bitis_kaybi']}")
    print(f"{'Kayıp Azalışı':<35} | %{rapor['kayip_dususu_yuzdesi']} İyileşme")
    print("-" * 80)

    # -------------------------------------------------------------
    # 3. 6 PANELLİ TEŞHİS PANOSU ÜRETİMİ
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli Görsel SFT Teşhis Panosu Üretiliyor...")
    gorsellestirici = VisualSFTGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "visual_instruction_tuning_paneli.png")
    gorsellestirici.pano_olustur(rapor, veriler, kayit_yolu=cikis_resmi)

    print("\n" + "=" * 110)
    print("✓ Day 163: VISUAL INSTRUCTION TUNING BAŞARIYLA TAMAMLANDI!")
    print("=" * 110)


if __name__ == "__main__":
    main()
