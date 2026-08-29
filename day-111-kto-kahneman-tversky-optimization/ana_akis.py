"""
Day 111: Kahneman-Tversky Optimization (KTO) ile LLM Hizalama Ana Akışı.
Eşleştirilmemiş ikili geri bildirimler (binary feedback), Beklenti Teorisi asimetrisi ve 6 panelli teşhis panosu.
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

from src.kto_laboratuvari import KTOLaboratuvari
from src.gorsellestirici import KTOGorsellestirici


def main():
    print("=" * 95)
    print(">>> Day 111: Kahneman-Tversky Optimization (KTO) & Prospect Theory Alignment")
    print("=" * 95)

    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Çalışma Donanımı: {cihaz.type.upper()}")

    # -------------------------------------------------------------
    # ADIM 1: Model ve Tekil İkili Veri Seti Başlatma
    # -------------------------------------------------------------
    print("\n[1/3] KTO Eşleştirilmemiş İkili Veri Seti ve Modeller Başlatılıyor...")
    lab = KTOLaboratuvari(
        vocab_size=1000,
        dim=256,
        num_heads=4,
        num_layers=4,
        cihaz=cihaz,
    )
    input_ids, maske, etiketler = lab.tekil_ikili_veri_uret(
        ornek_sayisi=400, prompt_len=10, resp_len=14
    )

    policy_p = sum(p.numel() for p in lab.policy_model.parameters())
    print(f"  * Üretilen Tekil Örnek Sayısı   : {input_ids.shape[0]} (200 Beğenildi [+1], 200 Beğenilmedi [-1])")
    print(f"  * Dizi Uzunluğu (Prompt+Yanıt)  : {input_ids.shape[1]} Token")
    print(f"  * 1. Politika Modeli (pi_theta) : {policy_p:,} Parametre (Eğitilebilir)")
    print(f"  * 2. Referans Modeli (pi_ref)   : {policy_p:,} Parametre (Dondurulmuş)")
    print("  * 3. Çiftli Eşleştirme (Pairs)  : GEREK YOK (Eşleştirilmemiş / Unpaired)")

    # -------------------------------------------------------------
    # ADIM 2: KTO Tercih Eğitimi (20 Epok, Beta=0.1, lambda_d=1.0, lambda_u=1.33)
    # -------------------------------------------------------------
    print("\n[2/3] KTO Beklenti Teorisi Tabanlı Hizalama Koşturuluyor (20 Epok, λ_D=1.0, λ_U=1.33)...")
    egitim_raporu = lab.kto_egit(
        input_ids=input_ids,
        maske=maske,
        etiketler=etiketler,
        epok_sayisi=20,
        batch_size=32,
        lr=1e-3,
        beta=0.1,
    )

    print("\n--- KTO TERCİH HİZALAMA EĞİTİM GELİŞİMİ ---")
    print(f"{'EPOK':<8} | {'TOPLAM LOSS':<12} | {'KAYIP (D)':<10} | {'KAYIP (U)':<10} | {'DOĞRULUK (%)':<14} | {'ÖRTÜK r_D':<10} | {'ÖRTÜK r_U':<10} | {'MARJİN':<10}")
    print("-" * 95)
    for ep in range(0, len(egitim_raporu["toplam_kayiplar"]), 4):
        print(
            f"{ep+1:<8} | "
            f"{egitim_raporu['toplam_kayiplar'][ep]:>10.4f} | "
            f"{egitim_raporu['kayiplar_d'][ep]:>8.4f} | "
            f"{egitim_raporu['kayiplar_u'][ep]:>8.4f} | "
            f"%{egitim_raporu['dogruluklar'][ep]:>11.2f} | "
            f"{egitim_raporu['r_d_ort'][ep]:>8.2f} | "
            f"{egitim_raporu['r_u_ort'][ep]:>8.2f} | "
            f"{egitim_raporu['marjinler'][ep]:>8.2f}"
        )
    # Son epok
    son_ep = len(egitim_raporu["toplam_kayiplar"]) - 1
    print(
        f"{son_ep+1:<8} | "
        f"{egitim_raporu['toplam_kayiplar'][son_ep]:>10.4f} | "
        f"{egitim_raporu['kayiplar_d'][son_ep]:>8.4f} | "
        f"{egitim_raporu['kayiplar_u'][son_ep]:>8.4f} | "
        f"%{egitim_raporu['dogruluklar'][son_ep]:>11.2f} | "
        f"{egitim_raporu['r_d_ort'][son_ep]:>8.2f} | "
        f"{egitim_raporu['r_u_ort'][son_ep]:>8.2f} | "
        f"{egitim_raporu['marjinler'][son_ep]:>8.2f}"
    )
    print("-" * 95)

    print("\n[-] KTO BEKLENTİ VE KAYIP KAÇINMASI ÖZETİ:")
    print(f"  * Başlangıç Toplam Kaybı : {egitim_raporu['toplam_kayiplar'][0]:.4f}")
    print(f"  * Nihai Toplam Kayıp     : {egitim_raporu['toplam_kayiplar'][-1]:.4f} (Kararlı Düşüş)")
    print(f"  * Nihai Tercih Doğruluğu : %{egitim_raporu['dogruluklar'][-1]:.2f}")
    print(f"  * Net Ödül Ayrışması     : {egitim_raporu['marjinler'][-1]:+.2f} Puan")

    # -------------------------------------------------------------
    # ADIM 3: 6 Panelli Teşhis Panosu Çizimi
    # -------------------------------------------------------------
    print("\n[3/3] 6 Panelli KTO Hizalama Teşhis Panosu Çiziliyor...")
    gorsellestirici = KTOGorsellestirici(dpi=300)
    cikis_resmi = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ciktilar",
        "kto_alignment_paneli.png",
    )
    gorsellestirici.pano_olustur(
        egitim_raporu,
        kayit_yolu=cikis_resmi,
    )

    print("\n" + "=" * 95)
    print("[OK] Day 111: Kahneman-Tversky Optimization (KTO) Analizleri Başarıyla Tamamlandı!")
    print("=" * 95)


if __name__ == "__main__":
    main()
