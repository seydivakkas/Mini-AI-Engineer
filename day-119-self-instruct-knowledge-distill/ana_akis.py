"""
Day 119: LLM Knowledge Distillation (Bilgi Damıtma) ve Self-Instruct Ana Akışı.
Öğretmen Modelden (Teacher) Öğrenci Modele (Student) yumuşatılmış logit KL aktarımı ve 6 panelli teşhis panosu.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.damitma_laboratuvari import DamitmaLaboratuvari
from src.gorsellestirici import DamitmaGorsellestirici


def main():
    print("=" * 95)
    print(">>> Day 119: Knowledge Distillation from Teacher LLM to Student LLM & Self-Instruct Pipeline")
    print("=" * 95)

    # -------------------------------------------------------------
    # ADIM 1: Damıtma Laboratuvarı ve Modellerin Başlatılması
    # -------------------------------------------------------------
    print("\n[1/3] Öğretmen ve Öğrenci Modeller Başlatılıyor...")
    lab = DamitmaLaboratuvari(
        vocab_size=1000,
        seq_len=32,
        sicaklik=2.5,
        alpha=0.3,
        lr=1e-3,
        seed=42,
    )

    p_ogretmen = lab.ogretmen.toplam_parametre()
    p_ogrenci = lab.ogrenci_kd.toplam_parametre()
    tasarruf_yuzdesi = ((p_ogretmen - p_ogrenci) / p_ogretmen) * 100.0

    print(f"  * Öğretmen Model (Teacher) : {p_ogretmen:,} Parametre (d_model=256, 4 Katman)")
    print(f"  * Öğrenci Model (Student)  : {p_ogrenci:,} Parametre (d_model=64, 2 Katman)")
    print(f"  * Parametre Küçülme Oranı  : %{tasarruf_yuzdesi:.1f}")
    print(f"  * Damıtma Sıcaklığı (T)    : T = {lab.kd_loss_fn.sicaklik}")
    print(f"  * Kayıp Dengesi (Alpha)    : Alpha = {lab.kd_loss_fn.alpha} (Hard CE) / {1.0 - lab.kd_loss_fn.alpha:.1f} (Soft KL)")

    # -------------------------------------------------------------
    # ADIM 2: Eğitim ve Çıkarım Kıyaslaması (Benchmark)
    # -------------------------------------------------------------
    print("\n[2/3] Standart SFT vs Knowledge Distillation (KD) Eğitimi Koşturuluyor...")
    rapor = lab.egitim_ve_kiyaslama_kostur(adim_sayisi=30, batch_size=16)

    hiz_str = f"{rapor['hizlanma_orani']:.1f}x Hızlı"

    print("\n--- ÖĞRETMEN vs ÖĞRENCİ MODEL PERFORMANS VE VERİMLİLİK TABLOSU ---")
    print(f"{'MİMARİ':<26} | {'PARAMETRE':<16} | {'SON KAYIP (LOSS)':<18} | {'GECİKME (ms)':<16} | {'HIZLANMA':<12}")
    print("-" * 98)
    print(
        f"{'Öğretmen Model (Teacher)':<26} | {rapor['ogretmen_parametre']:>14,d} | "
        f"{'Referans':>18} | {rapor['ogretmen_gecikme_ms']:>13.2f} ms | {'1.0x (Taban)':<12}"
    )
    print(
        f"{'Standart SFT Öğrenci':<26} | {rapor['ogrenci_parametre']:>14,d} | "
        f"{rapor['son_sft_kayip']:>18.4f} | {rapor['ogrenci_gecikme_ms']:>13.2f} ms | {hiz_str:<12}"
    )
    print(
        f"{'Knowledge Distilled (KD)':<26} | {rapor['ogrenci_parametre']:>14,d} | "
        f"{rapor['son_kd_kayip']:>18.4f} | {rapor['ogrenci_gecikme_ms']:>13.2f} ms | {hiz_str:<12}"
    )
    print("-" * 98)

    print(f"\n[-] KAZANIM ÖZETİ:")
    print(f"  * Parametre Tasarrufu : %{rapor['parametre_tasarrufu']:.1f} daha az bellek kullanımı")
    print(f"  * Çıkarım Hızlanması  : {rapor['hizlanma_orani']:.1f}x daha düşük gecikme")
    print(f"  * Nihai KL Diverjansı : {rapor['kl_kayiplar'][-1]:.4f} (Öğretmen bilgi dağılımı başarıyla aktarıldı)")

    # -------------------------------------------------------------
    # ADIM 3: 6 Panelli Teşhis Panosu Çizimi
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli Knowledge Distillation Teşhis Panosu Çiziliyor...")
    gorsellestirici = DamitmaGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ciktilar",
        "knowledge_distillation_paneli.png",
    )
    gorsellestirici.pano_olustur(
        rapor,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 95)
    print("[OK] Day 119: Knowledge Distillation & Self-Instruct Analizleri Başarıyla Tamamlandı!")
    print("=" * 95)


if __name__ == "__main__":
    main()
