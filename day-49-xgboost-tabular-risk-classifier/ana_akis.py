"""
Day 49: XGBoost ile Dengesiz Tabüler Risk Sınıflandırıcısı Ana Yürütme Betiği.
"""

import os
import numpy as np
import pandas as pd
from src.risk_veri_ureteci import RiskVeriSimulasyonu
from src.xgboost_risk_egitici import XGBoostRiskSiniflandirici
from src.gorsellestirici import XGBoostRiskGorsellestirici


def main():
    print("=" * 85)
    print(">>> DAY 49: XGBOOST İLE DENGESİZ TABÜLER RİSK SINIFLANDIRICISI")
    print("=" * 85)

    # 1. Dengesiz Risk Veri Setinin Simülasyonu (%5 Pozitif Sınıf)
    n_orneklem = 2000
    pozitif_oran = 0.05
    X, y = RiskVeriSimulasyonu.veri_seti_olustur(n_orneklem=n_orneklem, pozitif_oran=pozitif_oran, random_state=42)

    X_train, X_val, X_test, y_train, y_val, y_test = RiskVeriSimulasyonu.train_val_test_bol(X, y, random_state=42)

    print(f"[+] Veri Seti Üretildi: {n_orneklem} Satır, %{pozitif_oran*100:.1f} Pozitif Risk Sınıfı")
    print(f"    - Eğitim Seti    : {len(X_train)} Satır (Pozitif: {int(y_train.sum())}, Negatif: {int(len(y_train) - y_train.sum())})")
    print(f"    - Doğrulama Seti : {len(X_val)} Satır (Pozitif: {int(y_val.sum())})")
    print(f"    - Test Seti      : {len(X_test)} Satır (Pozitif: {int(y_test.sum())})")

    # 2. XGBoost Modeli Eğitimi (scale_pos_weight ve Early Stopping)
    print("\n[+] 1. Adım: scale_pos_weight ve Erken Durdurma ile XGBoost Eğitimi...")
    siniflandirici = XGBoostRiskSiniflandirici(
        max_depth=5,
        learning_rate=0.05,
        n_estimators=300,
        early_stopping_rounds=20,
        random_state=42
    )

    siniflandirici.fit(X_train, y_train, X_val, y_val)
    print(f"    - Otomatik scale_pos_weight : {siniflandirici.scale_pos_weight:.2f}x")
    print(f"    - Tamamlanan İterasyon     : {len(siniflandirici.egitim_gecmisi['validation_0']['logloss'])} Epoch")

    # 3. Validation Setinde Karar Eşiği (Threshold) Optimizasyonu
    print("\n[+] 2. Adım: Doğrulama Setinde F1-Score Maksimizasyonu ile Eşik Optimizasyonu...")
    esik_sonuc = siniflandirici.esik_degeri_optimize_et(X_val, y_val)
    print(f"    - Optimize Edilmiş Karar Eşiği : {esik_sonuc['optimal_esik']:.3f}")
    print(f"    - En Yüksek Validation F1      : {esik_sonuc['en_iyi_val_f1']:.4f}")

    # 4. Test Seti Değerlendirmesi
    print("\n[+] 3. Adım: Test Seti Üzerinde Model Performansının Değerlendirilmesi...")
    test_sonuc = siniflandirici.degerlendir(X_test, y_test)
    print(f"    - Test PR-AUC Skoru        : %{test_sonuc['pr_auc'] * 100:.2f} (Dengesiz Veri Ana Metriği)")
    print(f"    - Test ROC-AUC Skoru       : %{test_sonuc['roc_auc'] * 100:.2f}")
    print(f"    - Test F1-Skoru            : {test_sonuc['f1_skoru']:.4f}")
    print(f"    - Yakalama Oranı (Recall)  : %{test_sonuc['recall_yuzde']:.2f}")
    print(f"    - Kesinlik (Precision)     : %{test_sonuc['precision_yuzde']:.2f}")
    print(f"    - TP: {test_sonuc['tp']}, TN: {test_sonuc['tn']}, FP: {test_sonuc['fp']}, FN: {test_sonuc['fn']}")

    # 5. TreeSHAP Global Özellik Katkısı Çıkarımı
    print("\n[+] 4. Adım: Yerel TreeSHAP ile Öznitelik Katkılarının Hesaplanması...")
    shap_sonuc = siniflandirici.shap_katkilarini_cikar(X_test)
    print("    - En Etkili 5 Risk Faktörü (Mean |SHAP|):")
    for i, (feat, val) in enumerate(list(shap_sonuc["ortalama_mutlak_shap"].items())[:5], 1):
        print(f"      {i}. {feat:<24} : {val:.4f} SHAP puanı")

    # 6. 6 Panelli Teşhis Panosunun Çizilmesi
    print("\n" + "=" * 85)
    print(">>> 5. 6 PANELLİ XGBOOST RİSK VE SHAP TEŞHİS PANOSUNUN ÜRETİLMESİ")
    print("=" * 85)

    cikis_yolu = XGBoostRiskGorsellestirici.panel_ciz(
        test_sonuclari=test_sonuc,
        esik_sonuclari=esik_sonuc,
        shap_sonuclari=shap_sonuc,
        egitim_gecmisi=siniflandirici.egitim_gecmisi,
        scale_pos_weight=siniflandirici.scale_pos_weight,
        y_test=y_test,
        hedef_path="day-49-xgboost-tabular-risk-classifier/ciktilar/xgboost_risk_paneli.png"
    )
    print(f"[+] 6 Panelli Risk Panosu Kaydedildi: {os.path.abspath(cikis_yolu)}")
    print("=" * 85)
    print("DAY 49: XGBOOST DENGESİZ RİSK SINIFLANDIRICISI PAKETİ BAŞARIYLA TAMAMLANDI!")
    print("=" * 85)


if __name__ == "__main__":
    main()
