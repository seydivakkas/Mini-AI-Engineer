"""Day 24 Ana Çalıştırma Akışı: Model Değerlendirme & Hata Analizi.

Bu betik; çok sınıflı bir sınıflandırma modelinin çıktılarını alır,
Karışıklık Matrisi, Çok Sınıflı ROC-AUC, PR-AUC, Olasılık Kalibrasyonu (ECE),
Sıcaklık Ölçekleme (Temperature Scaling) ve Hata Denetimi analizlerini uçtan uca yürütür.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Proje kök dizinini ekle
proje_kok = Path(__file__).resolve().parent
if str(proje_kok) not in sys.path:
    sys.path.insert(0, str(proje_kok))

import numpy as np
from scipy.special import softmax

from src.metrik_hesaplayici import MetrikHesaplayici
from src.kalibrasyon_analizcisi import KalibrasyonAnalizcisi
from src.hata_denetcisi import HataDenetcisi
from src.gorsellestirici import DegerlendirmeGorsellestirici


def baslik(metin: str) -> None:
    """Konsol çıktıları için formatlı bölüm başlığı basar."""
    cizgi = "=" * 80
    print(f"\n{cizgi}\n>>> {metin}\n{cizgi}")


def sentetik_test_verisi_uret(
    n_samples: int = 240, n_classes: int = 4
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Gerçekçi model çıktıları, ham logitler ve olasılık matrisi simüle eder."""
    np.random.seed(42)
    sinif_isimleri = ["Vazo", "Kumaş", "Rozet", "Ahşap"]

    # Dengeli gerçek etiketler
    y_true = np.random.randint(0, n_classes, size=n_samples)

    # Logit matrisi üret (çoğunlukla doğru tahmin, az miktarda zor/yanlış örnekler)
    logits = np.random.normal(0, 1.0, (n_samples, n_classes))

    for i in range(n_samples):
        dogru_c = y_true[i]
        # %85 ihtimalle doğru sınıfa yüksek logit ata
        if np.random.rand() < 0.85:
            logits[i, dogru_c] += np.random.uniform(2.5, 4.5)
        else:
            # Yanlış sınıfa yüksek logit ata (Hata simülasyonu)
            yanlis_c = (dogru_c + np.random.randint(1, n_classes)) % n_classes
            logits[i, yanlis_c] += np.random.uniform(2.0, 3.8)

    y_probs = softmax(logits, axis=1)
    return y_true, logits, y_probs, sinif_isimleri


def main() -> None:
    """Day 24 kapsamlı değerlendirme ve hata analizi akışını yürütür."""
    baslik("AŞAMA 1: Test Kümesi Tahminleri ve Model Logitlerinin Hazırlanması")
    y_true, logits, y_probs, sinif_isimleri = sentetik_test_verisi_uret(n_samples=240, n_classes=4)

    print(f"[+] Değerlendirilen Toplam Test Örneği: {len(y_true)}")
    print(f"[+] Sınıf Dağılımı: {dict(zip(sinif_isimleri, np.bincount(y_true)))}")
    print(f"[+] Ham Logitler Şekli (Logits Shape) : {logits.shape}")
    print(f"[+] Olasılık Matrisi (Probs Shape)    : {y_probs.shape}")

    baslik("AŞAMA 2: Çok Sınıflı Metrikler ve Top-k Doğruluk Analizi")
    rapor = MetrikHesaplayici.kapsamli_rapor(y_true, y_probs, sinif_isimleri)

    print(f"[+] Genel Test Doğruluğu (Top-1 Accuracy) : %{rapor['dogruluk'] * 100:.2f}")
    print(f"[+] F1-Macro Skoru                        : {rapor['f1_macro']:.4f}")
    print(f"[+] Precision-Macro                       : {rapor['precision_macro']:.4f}")
    print(f"[+] Recall-Macro                          : {rapor['recall_macro']:.4f}")
    print(f"[+] Top-2 Doğruluğu (Top-2 Accuracy)      : %{rapor['top_k'][2] * 100:.2f}")
    print(f"[+] Top-3 Doğruluğu (Top-3 Accuracy)      : %{rapor['top_k'][3] * 100:.2f}")

    print("\n--- Sınıf Bazında Performans Tablosu ---")
    for ad, d in rapor["sinif_raporu"].items():
        print(f"  - {ad:<12}: Doğruluk = %{d['dogruluk']*100:<5.1f} ({d['dogru']}/{d['toplam']}) | ROC-AUC = {d['auc']:.3f} | PR-AP = {d['ap']:.3f}")

    baslik("AŞAMA 3: Çok Sınıflı ROC-AUC ve Precision-Recall (PR) Analizi")
    roc_bilgi = rapor["roc_bilgi"]
    pr_bilgi = rapor["pr_bilgi"]

    print(f"[+] Macro-Average ROC-AUC : {roc_bilgi['macro_auc']:.4f}")
    print(f"[+] Micro-Average ROC-AUC : {roc_bilgi['micro'][2]:.4f}")
    print(f"[+] Macro-Average PR-AUC  : {pr_bilgi['macro_ap']:.4f}")
    print(f"[+] Micro-Average PR-AUC  : {pr_bilgi['micro'][2]:.4f}")

    baslik("AŞAMA 4: Olasılık Kalibrasyonu ve Sıcaklık Ölçekleme (Temperature Scaling)")
    kalibrasyon_ham = KalibrasyonAnalizcisi.kalibrasyon_egrisi_ve_ece(y_true, y_probs, n_bins=10)
    brier_ham = KalibrasyonAnalizcisi.brier_skoru(y_true, y_probs, n_classes=4)

    print(f"[+] Ham Model Beklenen Kalibrasyon Hatası (ECE): {kalibrasyon_ham['ece']:.4f}")
    print(f"[+] Ham Model Maksimum Kalibrasyon Hatası (MCE): {kalibrasyon_ham['mce']:.4f}")
    print(f"[+] Ham Model Brier Skoru                     : {brier_ham:.4f}")

    # Optimal Sıcaklık T* Optimizasyonu
    optimal_T = KalibrasyonAnalizcisi.sicaklik_olcekleme_optimize_et(y_true, logits)
    y_probs_kalibre = KalibrasyonAnalizcisi.sicaklik_uygula(logits, optimal_T)

    kalibrasyon_kalibre = KalibrasyonAnalizcisi.kalibrasyon_egrisi_ve_ece(y_true, y_probs_kalibre, n_bins=10)
    brier_kalibre = KalibrasyonAnalizcisi.brier_skoru(y_true, y_probs_kalibre, n_classes=4)

    print(f"\n[*] Optimize Edilen Sıcaklık Katsayısı (T*): {optimal_T:.3f}")
    print(f"[+] Kalibre Model Beklenen Kalibrasyon Hatası (ECE): {kalibrasyon_kalibre['ece']:.4f} (İyileşme: -{(kalibrasyon_ham['ece'] - kalibrasyon_kalibre['ece']):.4f})")
    print(f"[+] Kalibre Model Brier Skoru                     : {brier_kalibre:.4f}")

    baslik("AŞAMA 5: Hata Denetimi (Error Audit & Failure Analysis)")
    asiri_guvenli_hatalar = HataDenetcisi.asiri_guvenli_yanlislar(y_true, y_probs, en_fazla=5)
    en_cok_karisan = HataDenetcisi.en_cok_karisan_ciftler(y_true, np.argmax(y_probs, axis=1), sinif_isimleri)

    print(f"[+] Tespit Edilen Toplam Yanlış Tahmin Sayısı: {sum(y_true != np.argmax(y_probs, axis=1))}")
    print("\n--- En Yüksek Güvenle Yapılan Yanlış Tahminler (Overconfident Failures) ---")
    for h in asiri_guvenli_hatalar:
        print(f"  - Örnek #{h['ornek_indeks']:<3}: Gerçek = {sinif_isimleri[h['gercek_sinif']]:<8} | Tahmin = {sinif_isimleri[h['tahmin_sinif']]:<8} | Güven = %{h['guven']*100:.1f}")

    print("\n--- En Sık Karışan Sınıf Çiftleri (Confusion Pairs) ---")
    for g, t, count in en_cok_karisan[:4]:
        print(f"  - Gerçek '{g}' -> Tahmin '{t}': {count} kez karıştırıldı.")

    baslik("AŞAMA 6: 6 Panelli Teşhis Panosunun (Dashboard) Kaydedilmesi")
    cikti_klasor = proje_kok / "ciktilar"
    cikti_klasor.mkdir(parents=True, exist_ok=True)
    dashboard_dosyasi = cikti_klasor / "model_degerlendirme_paneli.png"

    DegerlendirmeGorsellestirici.dashboard_ciz(
        metrik_raporu=rapor,
        kalibrasyon_ham=kalibrasyon_ham,
        kalibrasyon_kalibre=kalibrasyon_kalibre,
        asiri_guvenli_hatalar=asiri_guvenli_hatalar,
        sinif_isimleri=sinif_isimleri,
        hedef_dosya=dashboard_dosyasi,
    )
    print(f"[+] 6 panelli değerlendirme panosu başarıyla kaydedildi: {dashboard_dosyasi}")


if __name__ == "__main__":
    main()
