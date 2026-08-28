"""
Day 33: Hibrit Arama & Reciprocal Rank Fusion (RRF) Ana Yürütme Betiği.
"""

import os
from src.hibrit_arama_yoneticisi import HibritAramaYoneticisi
from src.gorsellestirici import HibritAramaGorsellestirici


def main():
    print("=" * 80)
    print(">>> ASAMA 1: Hibrit Arama Motorunun ve Korpusun Hazırlanması")
    print("=" * 80)

    yonetici = HibritAramaYoneticisi(rrf_k=60, embed_dim=128, device="cpu")

    korpus = [
        {
            "id": "DOC-001",
            "baslik": "Evrişimli Sinir Ağları ve Görüntü Analizi",
            "icerik": "Piksel filtreleri ve uzamsal öznitelik haritaları ile görsel sınıflandırma yapan derin öğrenme modeli.",
            "kategori": "Görüntü İşleme"
        },
        {
            "id": "DOC-002",
            "baslik": "Nesne Tespiti ve YOLO Mimarisi",
            "icerik": "Kamera karelerindeki öğelerin sınırlayıcı kutularını ve koordinatlarını gerçek zamanlı hesaplayan dedektör.",
            "kategori": "Görüntü İşleme"
        },
        {
            "id": "DOC-003",
            "baslik": "U-Net ile Anlamsal Bölütleme",
            "icerik": "Kodlayıcı ve kod çözücü arasındaki atlama bağlantılarıyla piksel düzeyinde sınır maskeleri üretir.",
            "kategori": "Görüntü İşleme"
        },
        {
            "id": "DOC-004",
            "baslik": "Vektör Veritabanları ve Semantik İndeksleme",
            "icerik": "Yüksek boyutlu embedding uzayında yaklaşık en yakın komşuları hızlıca bularak anlamsal kosinüs benzerliği eşlemesi yapar.",
            "kategori": "Veri Mimarisi"
        },
        {
            "id": "DOC-005",
            "baslik": "RAG (Retrieval-Augmented Generation) Mimarisi",
            "icerik": "Harici doküman bağlamını LLM modellerine dinamik olarak aktararak bilgi erişimini ve doğru yanıt üretimini sağlar.",
            "kategori": "Doğal Dil İşleme"
        },
        {
            "id": "DOC-006",
            "baslik": "BM25 Leksikal Arama ve Ters İndeks",
            "icerik": "Anahtar kelimelerin terim frekansı doygunluğu ve belge uzunluğu oranı üzerinden ters indeks sorgulaması yapar.",
            "kategori": "Bilgi Erişimi"
        },
        {
            "id": "DOC-007",
            "baslik": "Kalman Filtresi ve Çoklu Hedef Takibi",
            "icerik": "Sensör gürültülerini filtreleyip hız ve konum kestirimi yaparak video boyunca nesne kimliklerini korur.",
            "kategori": "Görüntü İşleme"
        },
        {
            "id": "DOC-008",
            "baslik": "Model Kuantizasyonu ve INT8 Hızlandırma",
            "icerik": "Ağırlık hassasiyetini düşürerek işlemci gecikmesini azaltan ve çıkarım hızını artıran donanım optimizasyonu.",
            "kategori": "MLOps"
        },
        {
            "id": "DOC-009",
            "baslik": "Hibrit Arama ve Reciprocal Rank Fusion",
            "icerik": "BM25 leksikal tam kelime eşleşmesi ile yoğun semantik vektör aramasını RRF sıralama füzyonuyla birleştirir.",
            "kategori": "Bilgi Erişimi"
        },
        {
            "id": "DOC-010",
            "baslik": "Üretim Seviyesi Asenkron FastAPI Servisi",
            "icerik": "Pydantic şemaları ile tip güvenli REST API uç noktaları oluşturarak yüksek eşzamanlı model çıkarımı sağlar.",
            "kategori": "MLOps"
        }
    ]

    yonetici.toplu_dokuman_ekle(korpus)
    print(f"[+] İndekslenen Toplam Doküman : {len(korpus)} adet")

    print("\n" + "=" * 80)
    print(">>> ASAMA 2: Hibrit Sorgu Yürütme (BM25 + Dense -> RRF Füzyonu)")
    print("=" * 80)
    sorgu = "RAG mimarisi ve vektör veritabanları ile anlamsal doküman sorgulama"
    print(f"[*] Hibrit Sorgu: '{sorgu}'")

    hibrit_sonuc = yonetici.hibrit_ara(
        sorgu=sorgu,
        agirlik_bm25=0.5,
        agirlik_semantik=0.5,
        top_k=4,
        fuzyon_yontemi="rrf"
    )

    print("\n[+] RRF NİHAİ SIRALAMA SONUÇLARI:")
    for sira, r in enumerate(hibrit_sonuc["final_sonuclar"], 1):
        bm25_r = r.get("siralama_gecmisi", {}).get("bm25", "-")
        sem_r = r.get("siralama_gecmisi", {}).get("semantik", "-")
        print(f"[{sira}] {r['doc_id']} — {r['baslik']}")
        print(f"    - RRF Skoru         : {r['skor']:.5f}")
        print(f"    - Sıralama Geçmişi  : BM25 Rank #{bm25_r}, Semantik Rank #{sem_r}")
        print(f"    - Ham Skorlar       : {r.get('orijinal_skorlar', {})}")

    print("\n" + "=" * 80)
    print(">>> ASAMA 3: 6 Panelli Hibrit Arama Teşhis Panosunun Üretilmesi")
    print("=" * 80)
    cikis_resmi = HibritAramaGorsellestirici.hibrit_panel_ciz(
        hibrit_ciktisi=hibrit_sonuc,
        hedef_path="day-33-hybrid-search-rrf/ciktilar/hibrit_arama_paneli.png"
    )
    print(f"[+] 6 Panelli Teşhis Panosu Kaydedildi: {os.path.abspath(cikis_resmi)}")
    print("=" * 80)
    print("DAY 33: HİBRİT ARAMA & RECIPROCAL RANK FUSION (RRF) BAŞARIYLA TAMAMLANDI!")
    print("=" * 80)


if __name__ == "__main__":
    main()
