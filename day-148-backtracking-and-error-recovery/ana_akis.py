"""
Day 148: Düşünce Yollarında Geri İzleme (Backtracking) ve Çıkmaz Sokakları Fark Etme Ana Akışı.
OpenAI o1 ve DeepSeek-R1 tarzı içsel monolog, kontrol noktaları ve durum geri sarma simülasyonu.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.geri_izleme_yoneticisi import GeriIzlemeYoneticisi
from src.gorsellestirici import BacktrackingGorsellestirici


def main():
    print("=" * 105)
    print(">>> Day 148: Backtracking & Error Recovery in Reasoning LLMs (FAZ 8)")
    print("=" * 105)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # -------------------------------------------------------------
    # 1. PROBLEM TANIMI (Klasik Bilişsel Çıkmaz: Sopa & Top Paradoksu)
    # -------------------------------------------------------------
    print("\n[1/3] Problem Tanımlanıyor: 'Bir beyzbol sopası ve topun toplam fiyatı $1.10'dur.")
    print("                          Sopa toptan $1.00 daha pahalıdır. Topun fiyatı nedir?'")

    baslangic_durumu = {"toplam": 1.10, "fark": 1.00}
    aday_adımlar = [
        {"metin": "1. Denklem: Sopa + Top = 1.10 dolar", "yeni_durum": {"denklem1": "s+t=1.10"}, "kontrol_noktasi_mi": True},
        {"metin": "2. Hızlı Çıkarım: 1.10 - 1.00 = 0.10 o halde top = 0.10 dolar", "yeni_durum": {"top": 0.10}}, # Hatalı Adım!
        {"metin": "2. Düzeltilmiş Denklem: Sopa = Top + 1.00 dolar", "yeni_durum": {"denklem2": "s=t+1.00"}},    # Alternatif Dal
        {"metin": "3. Yerine Koyma: (Top + 1.00) + Top = 1.10 => 2 * Top + 1.00 = 1.10", "yeni_durum": {"2t": 0.10}},
        {"metin": "4. Çözüm: 2 * Top = 0.10 => Top = 0.05 dolar", "yeni_durum": {"top": 0.05, "sopa": 1.05}},
        {"metin": "5. Sağlama: Sopa ($1.05) + Top ($0.05) = $1.10 (Kusursuz Doğrulandı!)", "yeni_durum": {"dogrulandi": True}},
    ]

    # -------------------------------------------------------------
    # 2. BACKTRACKING & İÇSEL MONOLOG ÇALIŞTIRMA
    # -------------------------------------------------------------
    print("\n[2/3] Düşünce Yığını (Call-Stack) & İçsel Monolog Motoru Çalıştırılıyor...")
    yonetici = GeriIzlemeYoneticisi()
    kurtarma_sonucu = yonetici.akil_yurut_ve_kurtar(baslangic_durumu, aday_adımlar)

    print("\n" + "-" * 75)
    print("LLM İÇSEL MONOLOG KAYITLARI:")
    for m in kurtarma_sonucu["ic_monologlar"]:
        print(f"  ⚡ {m}")
    print("-" * 75)

    print(f"\n  • Toplam Geri İzleme (Backtrack) Sayısı : {kurtarma_sonucu['toplam_geri_izleme_sayisi']}")
    print(f"  • Nihai Geçerli Düşünce Adımı Sayısı   : {len(kurtarma_sonucu['nihai_gecerli_zincir'])}")
    print(f"  • Bulunan Nihai Doğru Çözüm            : Top = $0.05 (Sopa = $1.05)")

    # -------------------------------------------------------------
    # 3. 6 PANELLİ TEŞHİS PANOSU ÜRETİMİ
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli Backtracking Teşhis Panosu Üretiliyor...")
    gorsellestirici = BacktrackingGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "backtracking_and_error_recovery_paneli.png")
    gorsellestirici.pano_olustur(kurtarma_sonucu, kayit_yolu=cikis_resmi)

    print("\n" + "=" * 105)
    print("✓ Day 148: BACKTRACKING & ERROR RECOVERY BAŞARIYLA TAMAMLANDI!")
    print("=" * 105)


if __name__ == "__main__":
    main()
