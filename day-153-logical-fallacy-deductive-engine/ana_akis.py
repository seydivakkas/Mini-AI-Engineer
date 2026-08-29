"""
Day 153: Tümdengelimsel Mantık Doğrulayıcı & Mantıksal Safsata Dedektörü Ana Akışı.
Geçerlilik (Validity), Sağlamlık (Soundness), Kıyas (Syllogism) ve 6+ Safsata Analizi.
"""

import os
import sys

# Windows terminali için UTF-8 stdout yapılandırması
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.tumdengelim_motoru import TumdengelimMotoru
from src.gorsellestirici import MantikGorsellestirici


def main():
    print("=" * 105)
    print(">>> Day 153: Deductive Logic Engine & Fallacy Detector (FAZ 8)")
    print("=" * 105)

    cikis_dizini = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ciktilar")
    os.makedirs(cikis_dizini, exist_ok=True)

    # -------------------------------------------------------------
    # 1. 5 FARKLI MANTIKSAL ARGÜMAN SENARYOSU
    # -------------------------------------------------------------
    argumanlar = [
        # 1. Klasik Sokrates Kıyası (Geçerli ve Sağlam)
        "Tüm insanlar ölümlüdür. Sokrates bir insandır. Dolayısıyla Sokrates ölümlüdür.",
        # 2. Sonucun Doğrulanması (Biçimsel Safsata)
        "Eğer yağmur yağarsa yerler ıslanır. Yerler şu an ıslak. O halde kesinlikle yağmur yağdı.",
        # 3. Kişiye Saldırı (Ad Hominem)
        "Sen zaten eğitimsizsin ve diplomasız birisin. Bu yüzden senin argümanın tamamen yanlıştır.",
        # 4. Korkuluk / Çarpıtma (Straw Man)
        "Karşı taraf yapay zekanın denetlenmesini istiyor. Demek ki onlar bütün teknolojiyi çöpe atıp tamamen yasaklamak istiyor.",
        # 5. Yanlış İkilem (False Dilemma)
        "Bu teklifi derhal kabul etmiyorsun. O halde ya bizimlesin ya da başarısızlığımızı isteyen bir düşmansın.",
    ]

    print(f"\n[1/3] {len(argumanlar)} Adet Argüman Mantık ve Safsata Motoruna Yükleniyor...")

    # -------------------------------------------------------------
    # 2. TÜMDENGELİMSEL VE SAFSATA DEĞERLENDİRMESİ
    # -------------------------------------------------------------
    print("\n[2/3] Tümdengelimsel Geçerlilik (Validity) ve Sağlamlık (Soundness) Taraması Yapılıyor...")
    motor = TumdengelimMotoru()
    degerlendirmeler = []

    print("-" * 100)
    for i, arg in enumerate(argumanlar, start=1):
        deg = motor.argumani_degerlendir(arg)
        degerlendirmeler.append(deg)

        print(f"\n[ARGÜMAN {i}]: '{arg}'")
        print(f"  • Öncüller ({len(deg['onculler'])} adet) : {deg['onculler']}")
        print(f"  • Varılan Sonuç        : '{deg['sonuc']}'")
        print(f"  • Geçerli mi (Valid)   : {deg['gecerli_mi']}")
        print(f"  • Sağlam mı (Sound)    : {deg['saglam_mi']}")
        print(f"  • Safsata Raporu       : {deg['safsata_bilgisi']['safsata_adi']} ({deg['safsata_bilgisi']['kategori']})")
        print(f"  • Sembolik Mantık Formu: {deg['safsata_bilgisi']['sembolik_yapi']}")
        print(f"  • Güven Skoru          : %{deg['guven_skoru']*100:.0f}")

    print("-" * 100)

    # -------------------------------------------------------------
    # 3. 6 PANELLİ TEŞHİS PANOSU ÜRETİMİ
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli Mantıksal Akıl Yürütme Teşhis Panosu Üretiliyor...")
    gorsellestirici = MantikGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(cikis_dizini, "logical_fallacy_deductive_engine_paneli.png")
    gorsellestirici.pano_olustur(degerlendirmeler, kayit_yolu=cikis_resmi)

    print("\n" + "=" * 105)
    print("✓ Day 153: LOGICAL FALLACY & DEDUCTIVE ENGINE BAŞARIYLA TAMAMLANDI!")
    print("=" * 105)


if __name__ == "__main__":
    main()
