"""
Day 116: Evol-Instruct & UltraFeedback ile Sentetik Veri Üretim Ana Akışı.
Tohum istemlerin çok nesilli evrimi, In-Depth/In-Breadth operatörleri, kalite elemesi ve 6 panelli teşhis panosu.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.sentetik_laboratuvar import SentetikVeriLaboratuvari
from src.gorsellestirici import SentetikVeriGorsellestirici


def main():
    print("=" * 95)
    print(">>> Day 116: Evol-Instruct & UltraFeedback Synthetic Data Generation Pipeline")
    print("=" * 95)

    # -------------------------------------------------------------
    # ADIM 1: Sentetik Veri Laboratuvarı Başlatma
    # -------------------------------------------------------------
    print("\n[1/3] Sentetik Veri Laboratuvarı ve Tohum İstem Havuzu Başlatılıyor...")
    lab = SentetikVeriLaboratuvari(seed=42)

    print(f"  * Başlangıç Tohum İstem Sayısı  : {len(lab.TOHUM_ISTEMLER)}")
    print("  * Evrim Operatörleri            : Kısıt Ekleme, Derinleştirme, Somutlaştırma, Muhakeme, Mutasyon")
    print("  * Kalite Filtresi               : Jaccard Benzerlik + Karmaşıklık Kazancı Denetimi")
    print("  * Tercih Puanlama Motoru        : UltraFeedback 4 Boyutlu Puanlayıcı")

    # -------------------------------------------------------------
    # ADIM 2: 3 Nesilli Evrimsel Veri Üretimi
    # -------------------------------------------------------------
    print("\n[2/3] 3 Nesilli Evol-Instruct Evrimi ve UltraFeedback Çift Üretimi Koşturuluyor...")
    rapor = lab.evrim_laboratuvarini_kostur(nesil_sayisi=3)

    print("\n--- EVOL-INSTRUCT NESİLLER BOYUNCA KARMAŞIKLIK GELİŞİMİ ---")
    print(f"{'NESİL':<14} | {'İSTEM SAYISI':<14} | {'ORTALAMA KARMAŞIKLIK (0-100)':<30} | {'DURUM':<15}")
    print("-" * 95)
    for g, skor in enumerate(rapor["ortalama_skorlar"]):
        tur = "Tohum (Seed)" if g == 0 else f"Evrimleşmiş (Gen {g})"
        print(f"Nesil {g:<8} | {len(rapor['nesil_havuzlari'][g]):<14} | {skor:>28.2f} Puan | {tur:<15}")
    print("-" * 95)

    print("\n[-] EVRİM OPERATÖRÜ KULLANIM İSTATİSTİKLERİ:")
    for op, sayi in rapor["operator_istatistikleri"].items():
        print(f"  * {op:<20}: {sayi:>3} kez başarıyla uygulandı")
    print(f"  * Kalite Filtresi Kabul Oranı: %{rapor['kabul_orani']:.2f}")
    print(f"  * Üretilen Toplam Çiftli Veri: {rapor['toplam_cift_sayisi']} Adet (DPO / SimPO Hazır)")

    # Örnek Bir Tercih Çifti Gösterimi
    ornek_cift = rapor["tercih_veri_seti"][0]
    print("\n[+] ÖRNEK EVRİLMİŞ ULTRAFEEDBACK TERCİH ÇİFTİ:")
    print(f"  [PROMPT]: {ornek_cift['prompt']}")
    print(f"  [CHOSEN YANIT (Skor: {ornek_cift['chosen_skor']:.2f})]: {ornek_cift['chosen'][:120]}...")
    print(f"  [REJECTED YANIT (Skor: {ornek_cift['rejected_skor']:.2f})]: {ornek_cift['rejected'][:120]}...")

    # -------------------------------------------------------------
    # ADIM 3: 6 Panelli Teşhis Panosu Çizimi
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli Sentetik Veri Teşhis Panosu Çiziliyor...")
    gorsellestirici = SentetikVeriGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ciktilar",
        "evol_instruct_paneli.png",
    )
    gorsellestirici.pano_olustur(
        rapor,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 95)
    print("[OK] Day 116: Evol-Instruct & UltraFeedback Analizleri Başarıyla Tamamlandı!")
    print("=" * 95)


if __name__ == "__main__":
    main()
