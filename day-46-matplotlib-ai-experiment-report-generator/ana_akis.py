"""
Day 46: Matplotlib/Seaborn ile Otomatik AI Deney Raporlama Motoru Ana Yürütme Betiği.
"""

import os
import numpy as np
from src.egitim_izleyici import EgitimGecmisi
from src.metrik_hesaplayici import MetrikHesaplayici
from src.raporlayici import OtomatikDeneyRaporlayici
from src.gorsellestirici import DeneyRaporuGorsellestirici


def main():
    print("=" * 85)
    print(">>> DAY 46: MATPLOTLIB/SEABORN İLE OTOMATİK AI DENEY RAPORLAMA MOTORU")
    print("=" * 85)

    # 1. 25 Epoch'luk Model Eğitimi Telemetrisinin Simülasyonu
    np.random.seed(42)
    toplam_epoch = 25
    gecmis = EgitimGecmisi(model_adi="VisionTransformer-Textile-v3", sabir_patience=6)

    t_loss = 0.85
    v_loss = 0.88
    t_acc = 0.60
    v_acc = 0.58

    for ep in range(1, toplam_epoch + 1):
        # Gerçekçi yakınsama ve epoch 18'den sonra hafif overfitting simülasyonu
        t_loss = max(0.08, t_loss * 0.91 + np.random.normal(0, 0.005))
        if ep <= 18:
            v_loss = max(0.15, v_loss * 0.92 + np.random.normal(0, 0.008))
        else:
            v_loss = v_loss + np.random.uniform(0.002, 0.012)  # Overfitting başlangıcı

        t_acc = min(0.985, t_acc + (1.0 - t_acc) * 0.12 + np.random.normal(0, 0.004))
        v_acc = min(0.942, v_acc + (0.95 - v_acc) * 0.10 + np.random.normal(0, 0.006))

        gecmis.epoch_ekle(
            epoch=ep,
            t_loss=t_loss,
            v_loss=v_loss,
            t_acc=t_acc,
            v_acc=v_acc,
            lr=1e-3 * (0.95 ** ep)
        )

    egitim_analizi = gecmis.analiz_et()
    print(f"[+] Eğitim Telemetrisi Kaydedildi: {toplam_epoch} Epoch Tamamlandı")
    print(f"    - En İyi Epoch       : {egitim_analizi['en_iyi_epoch']}")
    print(f"    - En İyi Val Loss    : {egitim_analizi['en_iyi_val_loss']}")
    print(f"    - En İyi Val Accuracy: %{egitim_analizi['en_iyi_val_acc']}")
    print(f"    - Overfitting Farkı  : {egitim_analizi['overfitting_gap']} ({egitim_analizi['overfitting_riski']})")

    # 2. Test Değerlendirme Verisi ve Metriklerin Hesaplanması
    print("\n[+] 2. Adım: Test Seti Çıkarımları ve İstatistiksel Eğrilerin Hesaplanması...")
    n_test = 500
    y_true = np.random.binomial(n=1, p=0.35, size=n_test)
    
    # Pozitif sınıflara daha yüksek, negatiflere daha düşük olasılıklar
    y_prob = np.clip(
        y_true * 0.70 + (1 - y_true) * 0.20 + np.random.normal(0, 0.18, size=n_test),
        0.01, 0.99
    )

    cm_analizi = MetrikHesaplayici.karmasiklik_matrisi_hesapla(y_true, y_prob, esik=0.50)
    roc_analizi = MetrikHesaplayici.roc_egrisi_hesapla(y_true, y_prob)
    pr_analizi = MetrikHesaplayici.pr_egrisi_hesapla(y_true, y_prob)

    print(f"    - Doğruluk (Accuracy) : %{cm_analizi['dogruluk_acc']}")
    print(f"    - F1-Skoru            : {cm_analizi['f1_skoru']}")
    print(f"    - Matthews Corr (MCC) : {cm_analizi['mcc_skoru']}")
    print(f"    - ROC-AUC             : {roc_analizi['roc_auc']}")
    print(f"    - Average Precision   : {pr_analizi['average_precision_ap']}")

    hiperparametreler = {
        "Optimizer": "AdamW (lr=0.001, weight_decay=1e-4)",
        "Batch Size": "64 (Dinamik Batch)",
        "Loss Function": "CrossEntropyLoss + LabelSmoothing(0.1)",
        "Scheduler": "CosineAnnealingLR (T_max=25)",
        "Hardware": "NVIDIA RTX 4090 / Mixed Precision (FP16)"
    }

    # 3. İnteraktif HTML Raporunun Üretilmesi
    print("\n" + "=" * 85)
    print(">>> 3. İNTERAKTİF HTML VE MARKDOWN DENEY RAPORUNUN DERLENMESİ")
    print("=" * 85)
    html_yolu = OtomatikDeneyRaporlayici.html_raporu_olustur(
        egitim_analizi=egitim_analizi,
        cm_analizi=cm_analizi,
        roc_analizi=roc_analizi,
        pr_analizi=pr_analizi,
        hiperparametreler=hiperparametreler,
        hedef_path="day-46-matplotlib-ai-experiment-report-generator/ciktilar/deney_raporu.html"
    )
    print(f"[+] Bağımsız HTML Deney Raporu Oluşturuldu: {os.path.abspath(html_yolu)}")

    # 4. 6 Panelli Teşhis Panosunun Üretilmesi
    print("\n" + "=" * 85)
    print(">>> 4. 6 PANELLİ MATPLOTLIB/SEABORN TEŞHİS PANOSUNUN ÇİZİLMESİ")
    print("=" * 85)
    panel_yolu = DeneyRaporuGorsellestirici.panel_ciz(
        egitim_gecmisi=gecmis,
        egitim_analizi=egitim_analizi,
        cm_analizi=cm_analizi,
        roc_analizi=roc_analizi,
        pr_analizi=pr_analizi,
        hedef_path="day-46-matplotlib-ai-experiment-report-generator/ciktilar/deney_raporu_paneli.png"
    )
    print(f"[+] 6 Panelli Deney Grafiği Kaydedildi: {os.path.abspath(panel_yolu)}")
    print("=" * 85)
    print("DAY 46: OTOMATİK AI DENEY RAPORLAMA MOTORU BAŞARIYLA TAMAMLANDI!")
    print("=" * 85)


if __name__ == "__main__":
    main()
