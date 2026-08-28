"""
Day 32: Yoğun Vektör Tabanlı Semantik Arama Motoru Ana Yürütme Betiği.
"""

import os
import numpy as np
from src.semantik_arama_motoru import SemantikAramaMotoru
from src.gorsellestirici import SemantikAramaGorsellestirici


def main():
    print("=" * 80)
    print(">>> ASAMA 1: Semantik Arama Motorunun ve Yoğun Vektör İndeksinin Kurulması")
    print("=" * 80)

    motor = SemantikAramaMotoru(embed_dim=128, device="cpu")

    korpus = [
        {
            "id": "DOC-001",
            "baslik": "Evrişimli Sinir Ağları ve Görüntü Analizi",
            "icerik": "Piksel matrisleri üzerinden uzamsal filtreler ile kenar ve doku özniteliklerini hiyerarşik olarak öğrenen derin ağlar.",
            "kategori": "Görüntü İşleme"
        },
        {
            "id": "DOC-002",
            "baslik": "Gerçek Zamanlı Nesne Konumlandırma ve YOLO",
            "icerik": "Kamera kareleri içindeki hedef öğelerin koordinat sınırlarını ve sınıflarını tek bir ileri beslemede tespit eden sistem.",
            "kategori": "Görüntü İşleme"
        },
        {
            "id": "DOC-003",
            "baslik": "U-Net ile Tıbbi Görüntü Segmentasyonu",
            "icerik": "Atlama bağlantıları kullanarak mikroskop ve röntgen görüntülerinde piksel düzeyinde lezyon ve hücre sınırlarını ayırır.",
            "kategori": "Görüntü İşleme"
        },
        {
            "id": "DOC-004",
            "baslik": "Vektör Veritabanları ve Benzerlik Araması",
            "icerik": "Yüksek boyutlu embedding uzayında yaklaşık en yakın komşuları hızlıca bularak anlamsal eşleme gerçekleştiren veritabanı motoru.",
            "kategori": "Veri Mimarisi"
        },
        {
            "id": "DOC-005",
            "baslik": "RAG Asistanı ve LLM Doküman Entegrasyonu",
            "icerik": "Harici bilgi depolarından dinamik bağlam çekerek dil modellerinin doğru ve güncel yanıtlar üretmesini sağlayan mimari.",
            "kategori": "Doğal Dil İşleme"
        },
        {
            "id": "DOC-006",
            "baslik": "BM25 Leksikal Arama ve Ters İndeks",
            "icerik": "Anahtar kelimelerin terim frekansı ve doküman uzunluğu oranını kullanarak tam metin eşleştirmesi yapan bilgi erişim modeli.",
            "kategori": "Bilgi Erişimi"
        },
        {
            "id": "DOC-007",
            "baslik": "Kalman Filtresi ve Çoklu Hedef Takibi",
            "icerik": "Hareket halindeki nesnelerin gürültülü sensör verilerinden hız ve konum kestirimi yaparak kimlik sürekliliğini korur.",
            "kategori": "Görüntü İşleme"
        },
        {
            "id": "DOC-008",
            "baslik": "Model Kuantizasyonu ve INT8 Hızlandırma",
            "icerik": "Ağırlık hassasiyetini düşürerek işlemci üzerinde çıkarım gecikmesini azaltan ve bellek tasarrufu sağlayan optimizasyon.",
            "kategori": "MLOps"
        },
        {
            "id": "DOC-009",
            "baslik": "Hibrit Arama ve Sıralama Füzyonu",
            "icerik": "Kelime bazlı leksikal skorlar ile yoğun vektör benzerliklerini RRF yöntemiyle birleştirerek en iyi sıralamayı üretir.",
            "kategori": "Bilgi Erişimi"
        },
        {
            "id": "DOC-010",
            "baslik": "Asenkron REST API ve Model Dağıtımı",
            "icerik": "FastAPI ve Pydantic ile yüksek verimli, tip güvenli ve eşzamanlı makine öğrenmesi mikroservisleri sunumu.",
            "kategori": "MLOps"
        }
    ]

    motor.toplu_dokuman_ekle(korpus)
    print(f"[+] İndekslenen Toplam Doküman : {motor.indeks.toplam_vektor_sayisi} adet")
    print(f"[+] Embedding Vektör Boyutu   : {motor.indeks.boyut}D (L2-Normalized)")

    print("\n" + "=" * 80)
    print(">>> ASAMA 2: Anlamsal / Kavramsal Sorgu Yürütme (Kelimeler Birebir Eşleşmese Bile)")
    print("=" * 80)
    ornek_sorgu = "akıllı doküman soru cevaplama ve dil modeli"
    print(f"[*] Arama Sorgusu: '{ornek_sorgu}'")

    sonuclar = motor.semantik_ara(ornek_sorgu, top_k=4)

    for sira, r in enumerate(sonuclar, 1):
        print(f"\n[{sira}] {r['doc_id']} — {r['baslik']} [{r['kategori']}]")
        print(f"    - Kosinüs Benzerliği Skoru : {r['skor']:.4f}")
        print(f"    - Özet İçerik               : {r['icerik'][:65]}...")

    print("\n" + "=" * 80)
    print(">>> ASAMA 3: Çoklu Sorgu-Doküman Çapraz Benzerlik Matrisi ve PCA Analizi")
    print("=" * 80)
    capraz_sorgular = [
        "akıllı doküman soru cevaplama ve dil modeli",
        "kamera karelerinde hedef öğe koordinat tespiti",
        "yüksek boyutlu embedding benzerlik araması"
    ]

    sorgu_vektorleri = motor.vektorlestirici.vektorlestir(capraz_sorgular)
    capraz_matris = np.dot(sorgu_vektorleri, motor.indeks.vektorler.T)
    print(f"[+] Çapraz Benzerlik Matrisi Şekli: {capraz_matris.shape} (3 Sorgu x 10 Doküman)")

    pca_2d, doc_idler, kategoriler = motor.temsil_uzayi_pca_projeksiyonu()
    print(f"[+] PCA 2D İndirgeme Tamamlandı: {pca_2d.shape}")

    print("\n" + "=" * 80)
    print(">>> ASAMA 4: 6 Panelli Semantik Arama Teşhis Panosunun Üretilmesi")
    print("=" * 80)
    cikis_resmi = SemantikAramaGorsellestirici.semantik_panel_ciz(
        arama_sonuclari=sonuclar,
        sorgu=ornek_sorgu,
        pca_2d=pca_2d,
        doc_idler=doc_idler,
        kategoriler=kategoriler,
        capraz_benzerlik_matrisi=capraz_matris,
        capraz_sorgular=capraz_sorgular,
        hedef_path="day-32-semantic-search-engine/ciktilar/semantik_arama_paneli.png"
    )
    print(f"[+] 6 Panelli Teşhis Panosu Kaydedildi: {os.path.abspath(cikis_resmi)}")
    print("=" * 80)
    print("DAY 32: YOĞUN VEKTÖR TABANLI SEMANTİK ARAMA MOTORU BAŞARIYLA TAMAMLANDI!")
    print("=" * 80)


if __name__ == "__main__":
    main()
