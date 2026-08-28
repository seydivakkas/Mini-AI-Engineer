"""
Day 31: BM25 Leksikal Belge Arama Motoru Ana Yürütme Betiği.
"""

import os
from src.arama_sunucusu import BelgeAramaSunucusu
from src.gorsellestirici import BM25Gorsellestirici


def main():
    print("=" * 80)
    print(">>> ASAMA 1: Belge Arama Sunucusu ve Korpusun Hazırlanması")
    print("=" * 80)

    sunucu = BelgeAramaSunucusu(k1=1.5, b=0.75)

    korpus = [
        {
            "id": "DOC-001",
            "baslik": "Derin Öğrenme ve Evrişimli Sinir Ağları (CNN)",
            "icerik": "Evrişimli sinir ağları (CNN), görsel veri analizi ve görüntü sınıflandırma görevlerinde filtre çekirdekleri kullanarak uzamsal özellikleri çıkarır."
        },
        {
            "id": "DOC-002",
            "baslik": "Nesne Tespiti ve YOLO Mimarisi",
            "icerik": "YOLO nesne tespiti algoritması, tek bir ileri besleme ile sınırlayıcı kutu regresyonu ve sınıf olasılıklarını eşzamanlı olarak tahmin eder."
        },
        {
            "id": "DOC-003",
            "baslik": "Anlamsal Bölütleme ve U-Net Modeli",
            "icerik": "U-Net mimarisi, kodlayıcı ve kod çözücü blokları arasındaki atlama bağlantıları (skip connections) sayesinde piksel düzeyinde hassas bölütleme maskeleri üretir."
        },
        {
            "id": "DOC-004",
            "baslik": "Vektör Veritabanları ve Semantik Arama",
            "icerik": "Vektör veritabanları, yoğun gömme (dense embedding) vektörleri üzerinde kosinüs benzerliği ve yaklaşık en yakın komşu araması (ANN) gerçekleştirir."
        },
        {
            "id": "DOC-005",
            "baslik": "RAG (Retrieval-Augmented Generation) Mimarisi",
            "icerik": "RAG mimarisi, bilgi erişim (retrieval) motoru ile büyük dil modellerini (LLM) birleştirerek harici doküman bağlamı ile halüsinasyonsuz yanıtlar üretir."
        },
        {
            "id": "DOC-006",
            "baslik": "BM25 Leksikal Arama ve Ters İndeks",
            "icerik": "Okapi BM25 algoritması, terim frekansı doygunluğu ve belge uzunluğu normalizasyonu ile ters indeks üzerinde anahtar kelime eşleştirmesi yapar."
        },
        {
            "id": "DOC-007",
            "baslik": "Çoklu Nesne Takibi ve DeepSORT Algoritması",
            "icerik": "DeepSORT, Kalman filtresi durum kestirimi ve derin Re-ID görsel görünüş gömmelerini Macar algoritması ile birleştirerek zamansal kimlik takibi yapar."
        },
        {
            "id": "DOC-008",
            "baslik": "Model Kuantizasyonu ve ONNX Runtime Optimizasyonu",
            "icerik": "FP32 modellerinin INT8 dinamik kuantizasyonu, model boyutunu küçültür ve çıkarım hızını (FPS) donanım üzerinde katlayarak gecikmeyi düşürür."
        },
        {
            "id": "DOC-009",
            "baslik": "Hibrit Arama ve Reciprocal Rank Fusion (RRF)",
            "icerik": "Hibrit arama, BM25 leksikal arama motoru ile semantik yoğun vektör aramasını RRF sıralama füzyonu ile birleştirerek en iyi arama doğruluğunu sağlar."
        },
        {
            "id": "DOC-010",
            "baslik": "Üretim Seviyesi FastAPI Model Servisi",
            "icerik": "FastAPI ile geliştirilen asenkron REST API servisleri, Pydantic şema doğrulaması ile yüksek eşzamanlı model çıkarım isteklerini karşılar."
        }
    ]

    sunucu.toplu_belge_ekle(korpus)
    print(f"[+] Toplam İndekslenen Belge Sayısı : {sunucu.indeks.belge_sayisi}")
    print(f"[+] Ortalama Belge Uzunluğu (avgdl) : {sunucu.indeks.ortalama_belge_uzunlugu:.2f} kelime")
    print(f"[+] Benzersiz Terim (Vocabulary)   : {len(sunucu.indeks.postings)} adet")

    print("\n" + "=" * 80)
    print(">>> ASAMA 2: Örnek Leksikal Sorgu Yürütme ve Terim Katkısı Analizi")
    print("=" * 80)
    ornek_sorgu = "vektör veritabanı semantik arama ve rag mimarisi"
    print(f"[*] Sorgu: '{ornek_sorgu}'")

    sonuclar = sunucu.ara(ornek_sorgu, top_k=4)

    for sira, r in enumerate(sonuclar, 1):
        print(f"\n[{sira}] {r['doc_id']} — {r['baslik']}")
        print(f"    - BM25 Skoru      : {r['skor']:.4f}")
        print(f"    - Belge Uzunluğu  : {r['uzunluk']} kelime")
        print(f"    - Terim Katkıları : {r['terim_katkilari']}")

    print("\n" + "=" * 80)
    print(">>> ASAMA 3: Parametre Duyarlılık Analizi (k1 ve b Değerleri)")
    print("=" * 80)
    duyarlilik = sunucu.parametre_duyarlilik_analizi(
        sorgu_metni=ornek_sorgu,
        k1_listesi=[1.2, 1.5, 2.0],
        b_listesi=[0.25, 0.75]
    )
    for conf_name, res in list(duyarlilik.items())[:4]:
        en_iyi_doc = res[0]['doc_id'] if res else "Yok"
        en_iyi_skor = res[0]['skor'] if res else 0.0
        print(f"[+] Konfigürasyon: {conf_name:12s} | En İyi Belge: {en_iyi_doc} | Skor: {en_iyi_skor:.4f}")

    print("\n" + "=" * 80)
    print(">>> ASAMA 4: 6 Panelli BM25 Teşhis Panosunun Üretilmesi")
    print("=" * 80)
    istatistikler = {
        "belge_sayisi": sunucu.indeks.belge_sayisi,
        "ortalama_uzunluk": sunucu.indeks.ortalama_belge_uzunlugu
    }

    cikis_resmi = BM25Gorsellestirici.arama_panosu_ciz(
        arama_sonuclari=sonuclar,
        sorgu=ornek_sorgu,
        indeks_istatistikleri=istatistikler,
        hedef_path="day-31-bm25-document-search/ciktilar/bm25_arama_paneli.png"
    )
    print(f"[+] 6 Panelli Teşhis Panosu Kaydedildi: {os.path.abspath(cikis_resmi)}")
    print("=" * 80)
    print("DAY 31: BM25 LEKSİKAL BELGE ARAMA MOTORU BAŞARIYLA TAMAMLANDI!")
    print("=" * 80)


if __name__ == "__main__":
    main()
