"""
Day 50: Model Değerlendirme & Eşik Değeri Mühendisliği Ana Yürütme Betiği.
"""

import os
import numpy as np
from src.kalibrasyon_motoru import OlasilikKalibratoru
from src.esik_muhendisi import EsikDegeriMuhendisi
from src.gorsellestirici import EsikMuhendisligiGorsellestirici


def main():
    print("=" * 85)
    print(">>> DAY 50: MODEL DEĞERLENDİRME & EŞİK DEĞERİ MÜHENDİSLİĞİ")
    print("=" * 85)

    # 1. Gerçekçi Risk Olasılıkları ve Hedef Değerlerin Simülasyonu
    np.random.seed(42)
    n = 1000
    p_risk = 0.08  # %8 Pozitif Risk Sınıfı

    y_true = np.random.binomial(n=1, p=p_risk, size=n)
    # Gerçekçi ve hafif aşırı özgüvenli ham model olasılığı
    raw_prob = y_true * 0.75 + (1 - y_true) * 0.15 + np.random.normal(0, 0.20, size=n)
    y_prob = np.clip(raw_prob, 0.01, 0.99)

    print(f"[+] Değerlendirme Verisi Üretildi: {n} İşlem, {int(y_true.sum())} Riskli (%{y_true.mean()*100:.1f})")

    # 2. Olasılık Kalibrasyonu ve Brier Skoru Analizi
    print("\n[+] 1. Adım: Brier Skoru ve Güvenilirlik (ECE) Kalibrasyon Analizi...")
    kalibrasyon_ham = OlasilikKalibratoru.kalibrasyon_analizi_yap(y_true, y_prob)
    print(f"    - Ham Brier Skoru        : {kalibrasyon_ham['brier_skoru']:.4f}")
    print(f"    - Ham ECE Hatası         : %{kalibrasyon_ham['ece_skoru'] * 100:.2f}")
    print(f"    - Kalibrasyon Durumu     : {kalibrasyon_ham['kalibrasyon_durumu']}")

    # İzotonik kalibrasyon uygulama
    y_prob_kalibre = OlasilikKalibratoru.izotonik_kalibre_et(y_true, y_prob, y_prob)
    kalibrasyon_opt = OlasilikKalibratoru.kalibrasyon_analizi_yap(y_true, y_prob_kalibre)
    print(f"    - Kalibre Edilmiş Brier  : {kalibrasyon_opt['brier_skoru']:.4f} (İyileşme: {kalibrasyon_ham['brier_skoru'] - kalibrasyon_opt['brier_skoru']:+.4f})")
    print(f"    - Kalibre Edilmiş ECE    : %{kalibrasyon_opt['ece_skoru'] * 100:.2f}")

    # 3. İşletme Maliyet-Fayda Matrisinin Tanımlanması
    maliyet_matrisi = {
        "b_tp": 3000.0,   # Yakalanan dolandırıcılık/risk faydası ($)
        "b_tn": 20.0,     # Sorunsuz müşteri işlem faydası ($)
        "c_fp": 100.0,    # Yanlış alarm müşteri sürtünmesi / inceleme maliyeti ($)
        "c_fn": 4500.0    # Kaçırılan dolandırıcılık / tazminat zararı ($)
    }

    # 4. Eşik Değeri Mühendisliği ve F-Beta Optimizasyonu
    print("\n[+] 2. Adım: Eşik Taraması (F0.5, F1, F2 ve Finansal Net Kazanç)...")
    esik_sonuc = EsikDegeriMuhendisi.esik_tarama_analizi(
        y_gercek=y_true,
        y_olasilik=y_prob_kalibre,
        maliyet_matrisi=maliyet_matrisi
    )

    print(f"    - F0.5-Optimal Eşik (Precision Öncelikli) : {esik_sonuc['optimal_f05_esigi']:.3f}")
    print(f"    - F1.0-Optimal Eşik (Dengeli Tercih)       : {esik_sonuc['optimal_f1_esigi']:.3f}")
    print(f"    - F2.0-Optimal Eşik (Recall/Risk Öncelikli): {esik_sonuc['optimal_f2_esigi']:.3f}")
    print(f"    - Finansal Optimal Eşik (Maks. Net Kazanç) : {esik_sonuc['optimal_finansal_esik']:.3f}")
    print(f"    - Maksimum Net Finansal Kazanç             : ${esik_sonuc['maksimum_net_kazanc']:,.2f}")

    # 5. 6 Panelli Teşhis Panosunun Çizilmesi
    print("\n" + "=" * 85)
    print(">>> 3. 6 PANELLİ EŞİK MÜHENDİSLİĞİ VE MALİYET-FAYDA PANOSUNUN ÜRETİLMESİ")
    print("=" * 85)

    cikis_yolu = EsikMuhendisligiGorsellestirici.panel_ciz(
        kalibrasyon_sonuc=kalibrasyon_opt,
        esik_sonuc=esik_sonuc,
        hedef_path="day-50-model-evaluation-threshold-engineering/ciktilar/esik_muhendisligi_paneli.png"
    )
    print(f"[+] 6 Panelli Eşik Mühendisliği Panosu Kaydedildi: {os.path.abspath(cikis_yolu)}")
    print("=" * 85)
    print("DAY 50: MODEL DEĞERLENDİRME & EŞİK MÜHENDİSLİĞİ PAKETİ BAŞARIYLA TAMAMLANDI!")
    print("=" * 85)


if __name__ == "__main__":
    main()
