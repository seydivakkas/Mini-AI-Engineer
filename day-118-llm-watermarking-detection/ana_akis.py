"""
Day 118: LLM Filigranlama ve Tespit Ana Akışı.
Kirchenbauer algoritması, Yeşil/Kırmızı liste logit yanlılığı, Z-Skoru hipotez testi ve 6 panelli teşhis panosu.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.filigran_laboratuvari import FiligranLaboratuvari
from src.gorsellestirici import FiligranGorsellestirici


def main():
    print("=" * 95)
    print(">>> Day 118: LLM Cryptographic Watermarking & Z-Score Detection Pipeline (Kirchenbauer et al.)")
    print("=" * 95)

    # -------------------------------------------------------------
    # ADIM 1: Filigran Laboratuvarı Başlatma
    # -------------------------------------------------------------
    print("\n[1/3] LLM Filigran Laboratuvarı ve Kriptografik Parametreler Başlatılıyor...")
    lab = FiligranLaboratuvari(
        vocab_size=1000,
        gamma=0.5,
        delta=2.5,
        gizli_anahtar=15485863,
        z_esigi=4.0,
        seed=42,
    )

    print("  * Algoritmik Referans        : Kirchenbauer et al. (ICML 2023 En İyi Makale)")
    print(f"  * Sözlük Boyutu (|V|)        : {lab.vocab_size} Token")
    print(f"  * Yeşil Liste Oranı (gamma)  : %{lab.ekleyici.gamma * 100:.1f}")
    print(f"  * Logit Yanlılığı (delta)    : +{lab.ekleyici.delta}")
    print(f"  * Karar Eşiği (Z-Score)      : Z >= {lab.tespitci.z_esigi} (p < 0.00003)")

    # -------------------------------------------------------------
    # ADIM 2: Benchmark ve İstatistiksel Tespit Deneyi
    # -------------------------------------------------------------
    print("\n[2/3] Filigranlı vs Filigransız Metin Benchmark Deneyi Koşturuluyor...")
    rapor = lab.benchmark_kostur(ornek_sayisi=30, dizi_uzunlugu=100)

    print("\n--- LLM FİLİGRAN TESPİT VE Z-SKORU KARŞILAŞTIRMA TABLOSU ---")
    print(f"{'METİN GRUBU':<26} | {'YEŞİL ORAN (%)':<16} | {'ORTALAMA Z-SKORU':<18} | {'TESPİT ORANI (TPR/FPR)':<24} | {'DURUM':<10}")
    print("-" * 105)
    print(
        f"{'Filigransız / İnsan Metni':<26} | %{rapor['filigransiz_yesil_oran'] * 100:>13.2f} | "
        f"Z = {rapor['filigransiz_ort_z']:>12.2f} | FPR = %{rapor['fpr_yanlis_alarm_orani']:>16.1f} | {'TEMİZ':<10}"
    )
    print(
        f"{'Filigranlı Model Çıktısı':<26} | %{rapor['filigranli_yesil_oran'] * 100:>13.2f} | "
        f"Z = {rapor['filigranli_ort_z']:>12.2f} | TPR = %{rapor['tpr_dogru_tespit_orani']:>16.1f} | {'AI DAMGALI':<10}"
    )
    print("-" * 105)

    print("\n[-] DELTA (δ) YANLILIK DEĞERİNE GÖRE Z-SKORU GELİŞİMİ:")
    for d, z in zip(rapor["delta_degerleri"], rapor["delta_z_skorlari"]):
        print(f"  * Delta = {d:.1f} -> Ortalama Z-Skoru: {z:>5.2f} ({'TESPİT EDİLEBİLİR' if z >= 4.0 else 'EŞİK ALTI'})")

    print("\n[-] METİN DÜZENLEME (PARAPHRASE) SALDIRISI DAYANIKLILIĞI:")
    for e, z in zip(rapor["edit_oranlari"], rapor["paraphrase_z"]):
        print(f"  * %{int(e * 100):<2} Token Değiştirildi -> Kalan Z-Skoru: {z:>5.2f} ({'GÜÇLÜ İMZA' if z >= 4.0 else 'ZAYIFLAMIŞ'})")

    # Canlı Örnek İncelemesi
    ornek_f = lab.ekleyici.token_dizisi_uret(baslangic_token=50, uzunluk=80, filigran_aktif=True)
    analiz_f = lab.tespitci.filigran_analizi(ornek_f)
    print("\n[+] CANLI METİN İNCELEMESİ (Tek Bir Çıktı):")
    print(f"  * İncelenen Token Sayısı    : {analiz_f['toplam_token']}")
    print(f"  * Gözlenen Yeşil Token      : {analiz_f['yesil_token_sayisi']}/{analiz_f['toplam_gecis']} (%{analiz_f['yesil_oran']*100:.1f})")
    print(f"  * Hesaplanan Z-Skoru        : Z = {analiz_f['z_skoru']:.2f}")
    print(f"  * Hipotez Testi p-Değeri    : p = {analiz_f['p_degeri']:.8f}")
    print(f"  * Nihai Karar               : {'[ONAYLANDI] BU METİN KESİNLİKLE YAPAY ZEKA TARAFINDAN ÜRETİLMİŞTİR' if analiz_f['filigran_var_mi'] else '[TEMİZ] İNSAN METNİ'}")

    # -------------------------------------------------------------
    # ADIM 3: 6 Panelli Teşhis Panosu Çizimi
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli LLM Filigran Teşhis Panosu Çiziliyor...")
    gorsellestirici = FiligranGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ciktilar",
        "filigran_tespit_paneli.png",
    )
    gorsellestirici.pano_olustur(
        rapor,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 95)
    print("[OK] Day 118: LLM Filigranlama ve Z-Skoru Analizleri Başarıyla Tamamlandı!")
    print("=" * 95)


if __name__ == "__main__":
    main()
