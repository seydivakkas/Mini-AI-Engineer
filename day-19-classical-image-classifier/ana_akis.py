"""Day 19 Ana Çalıştırma Akışı: Geleneksel Makine Öğrenmesi ile Görsel Sınıflandırma.

Bu betik; 4 farklı görsel sınıfında (Vazo, Kumaş, Rozet, Ahşap) toplam 60 sentetik
görsel üretir, HOG + LBP + Renk Momentleri özniteliklerini çıkarır,
veri sızıntısız StandardScaler + Model Pipeline'ı ile SVM (RBF & Linear) ve
Random Forest modellerini eğitir, 5-Katlı Çapraz Doğrulama yapar,
test kümesinde performanslarını kıyaslar ve 4 panelli grafik raporu kaydeder.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Proje kök dizinini Python path'e ekle
proje_kok = Path(__file__).resolve().parent
if str(proje_kok) not in sys.path:
    sys.path.insert(0, str(proje_kok))

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from src.oznitelik_cikarici import KlasikOznitelikCikarici
from src.siniflandirici import GorselSiniflandirici, SiniflandiriciTipi, ModelSonucu
from src.degerlendirici import SiniflandirmaDegerlendirici


def baslik(metin: str) -> None:
    """Konsol çıktıları için formatlı bölüm başlığı basar."""
    cizgi = "=" * 80
    print(f"\n{cizgi}\n>>> {metin}\n{cizgi}")


def cok_sinifli_veri_seti_uret() -> Tuple[List[np.ndarray], List[int], List[str]]:
    """4 sınıfta 15'er adet (toplam 60) varyasyonlu sentetik görsel üretir."""
    sinif_isimleri = ["Vazo", "Kumaş", "Rozet", "Ahşap"]
    gorseller: List[np.ndarray] = []
    etiketler: List[int] = []
    h, w = 128, 128

    # --- Sınıf 0: Vazo (Kırmızı / Terracotta Tonları, Elips Gövde) ---
    for i in range(15):
        img = np.full((h, w, 3), (220 + i, 220, 225), dtype=np.uint8)
        bgr = (30 + i * 2, 50 + i * 3, 190 + i * 3)
        genislik = 25 + (i % 6) * 3
        cv2.ellipse(img, (64, 75), (genislik, 40), 0, 0, 360, bgr, -1)
        cv2.rectangle(img, (64 - int(genislik * 0.5), 25), (64 + int(genislik * 0.5), 45), bgr, -1)
        if i % 3 == 0:
            cv2.ellipse(img, (64, 75), (genislik, 8), 0, 0, 360, (20, 200, 240), -1)
        gorseller.append(img)
        etiketler.append(0)

    # --- Sınıf 1: Kumaş (Mavi / Lacivert Tonları, Çizgili / Dokulu) ---
    for i in range(15):
        ton = (180 + (i % 4) * 15, 90 + i * 3, 20 + i * 2)
        img = np.full((h, w, 3), ton, dtype=np.uint8)
        adim = 6 + (i % 5) * 2
        for x in range(0, w, adim):
            cv2.line(img, (x, 0), (x + 35, h), (max(0, ton[0] - 30), max(0, ton[1] - 30), max(0, ton[2] - 10)), 2)
        gorseller.append(img)
        etiketler.append(1)

    # --- Sınıf 2: Rozet (Altın / Parlak Sarı Daireler) ---
    for i in range(15):
        img = np.full((h, w, 3), (35, 35, 40), dtype=np.uint8)
        boyut = 28 + (i % 5) * 3
        cv2.circle(img, (64, 64), boyut, (30, 200 + (i % 4) * 10, 245), -1)
        cv2.circle(img, (64, 64), int(boyut * 0.7), (20, 160, 220), -1)
        cv2.circle(img, (64, 64), int(boyut * 0.35), (255, 255, 255), -1)
        gorseller.append(img)
        etiketler.append(2)

    # --- Sınıf 3: Ahşap (Kahverengi / Ahşap Dokusu) ---
    for i in range(15):
        ton = (35 + (i % 3) * 15, 75 + i * 2, 120 + i * 4)
        img = np.full((h, w, 3), ton, dtype=np.uint8)
        for y in range(0, h, 5 + (i % 3)):
            cizgi_renk = (max(0, ton[0] - 15), max(0, ton[1] - 15), max(0, ton[2] - 15))
            cv2.line(img, (0, y), (w, y), cizgi_renk, 1)
        gorseller.append(img)
        etiketler.append(3)

    return gorseller, etiketler, sinif_isimleri


def main() -> None:
    """Geleneksel görsel sınıflandırma tam eğitim ve kıyaslama akışını çalıştırır."""
    baslik("AŞAMA 1: Çok Sınıflı Sentetik Görsel Veri Setinin Üretilmesi")
    gorseller, etiketler_list, sinif_isimleri = cok_sinifli_veri_seti_uret()
    y = np.array(etiketler_list, dtype=np.int64)

    print(f"[+] Toplam Üretilen Görsel Sayısı: {len(gorseller)}")
    print(f"[+] Sınıf Dağılımı: {dict(zip(sinif_isimleri, [int(np.sum(y == i)) for i in range(4)]))}")

    baslik("AŞAMA 2: HOG + LBP + Renk Momentleri Öznitelik Çıkarımı")
    cikarici = KlasikOznitelikCikarici(hedef_boyut=(64, 64))
    t0 = time.perf_counter()
    X = np.array([cikarici.cikar(img) for img in gorseller])
    t_cikarim = (time.perf_counter() - t0) * 1000.0

    print(f"[+] Öznitelik Çıkarımı Tamamlandı: {t_cikarim:.2f} ms")
    print(f"[+] Toplam Öznitelik Matrisi Boyutu: {X.shape} (N={X.shape[0]}, D={X.shape[1]})")
    print(f"    - HOG Bileşeni Boyutu: 288 Boyut (9 blok x 32)")
    print(f"    - Uniform LBP Bileşeni: 10 Boyut")
    print(f"    - Renk Momentleri (BGR + HSV): 12 Boyut")
    print(f"    - Toplam Vektör Uzunluğu: 310 Boyut")

    baslik("AŞAMA 3: Veri Setinin Stratified Olarak Bölünmesi (Train / Test Split)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    print(f"[+] Eğitim Kümesi (Train): {X_train.shape[0]} örnek")
    print(f"[+] Test Kümesi (Test / Holdout): {X_test.shape[0]} örnek")

    baslik("AŞAMA 4: 5-Katlı Çapraz Doğrulama (5-Fold Stratified Cross-Validation)")
    yonetici = GorselSiniflandirici(random_state=42)

    cv_modelleri = [
        SiniflandiriciTipi.SVM_RBF,
        SiniflandiriciTipi.SVM_LINEAR,
        SiniflandiriciTipi.RANDOM_FOREST,
    ]

    print("Model Türü              | 5-Fold Ortalama Doğruluk (Acc) | Standart Sapma (±)")
    print("-" * 75)
    for model_tipi in cv_modelleri:
        ort_acc, std_acc = yonetici.capraz_dogrulama_yap(X_train, y_train, model_tipi, k_kat=5)
        print(f"{model_tipi.value:<23} | %{ort_acc * 100:<29.2f} | ±%{std_acc * 100:.2f}")

    baslik("AŞAMA 5: Modellerin Eğitilmesi ve Test Kümesinde Değerlendirilmesi")
    sonuclar: List[ModelSonucu] = []

    # 1. Model: SVM (RBF Kernel)
    sonuc_svm_rbf = yonetici.egit_ve_degerlendir(
        X_train, y_train, X_test, y_test, SiniflandiriciTipi.SVM_RBF, C=10.0, gamma="scale"
    )
    sonuclar.append(sonuc_svm_rbf)
    print(f"[+] {sonuc_svm_rbf.ozet()}")

    # 2. Model: SVM (Linear Kernel)
    sonuc_svm_lin = yonetici.egit_ve_degerlendir(
        X_train, y_train, X_test, y_test, SiniflandiriciTipi.SVM_LINEAR, C=1.0
    )
    sonuclar.append(sonuc_svm_lin)
    print(f"[+] {sonuc_svm_lin.ozet()}")

    # 3. Model: Random Forest
    sonuc_rf = yonetici.egit_ve_degerlendir(
        X_train, y_train, X_test, y_test, SiniflandiriciTipi.RANDOM_FOREST, n_estimators=150
    )
    sonuclar.append(sonuc_rf)
    print(f"[+] {sonuc_rf.ozet()}")

    baslik("AŞAMA 6: Kapsamlı Karşılaştırma ve Metrik Tablosu")
    print(f"{'Model':<22} | {'Doğruluk':<10} | {'F1-Macro':<10} | {'Precision':<10} | {'Recall':<10} | {'Eğitim (ms)':<12} | {'Çıkarım (ms)'}")
    print("-" * 95)
    for s in sonuclar:
        print(
            f"{s.model_adi:<22} | "
            f"%{s.accuracy * 100:<9.1f} | "
            f"{s.f1_macro:<10.4f} | "
            f"{s.precision_macro:<10.4f} | "
            f"{s.recall_macro:<10.4f} | "
            f"{s.egitim_suresi_ms:<12.2f} | "
            f"{s.tahmin_suresi_ms:.2f} ms"
        )
    print("-" * 95)

    baslik("AŞAMA 7: Görselleştirme ve Analiz Raporunun Kaydedilmesi")
    cikti_klasor = proje_kok / "ciktilar"
    cikti_klasor.mkdir(parents=True, exist_ok=True)
    rapor_dosyasi = cikti_klasor / "siniflandirma_raporu.png"

    SiniflandirmaDegerlendirici.kapsamli_rapor_olustur(
        sonuclar=sonuclar,
        sinif_isimleri=sinif_isimleri,
        hedef_dosya=rapor_dosyasi,
    )
    print(f"[+] Sınıflandırma teşhis raporu başarıyla kaydedildi: {rapor_dosyasi}")


if __name__ == "__main__":
    main()
