"""
Day 34: Mini RAG Asistanı & Doküman Soru-Cevap Motoru Ana Yürütme Betiği.
"""

import os
from src.rag_asistani import MiniRAGAsistani
from src.gorsellestirici import RAGGorsellestirici


def main():
    print("=" * 80)
    print(">>> ASAMA 1: Doküman Havuzunun Hazırlanması ve RAG İndeksleme")
    print("=" * 80)

    asistan = MiniRAGAsistani(chunk_boyutu=35, cakisma_miktari=10, embed_dim=128, guven_esigi=0.10, device="cpu")

    dokumanlar = [
        {
            "id": "KB-001",
            "baslik": "Endüstriyel Görü ve Kalite Kontrol Kılavuzu",
            "icerik": (
                "Fabrika üretim hatlarında nesne tespiti için YOLO mimarileri tercih edilir. "
                "Hatalı kumaş ve dokuma kusurlarının tespiti için piksel düzeyinde Mask R-CNN ve SegFormer bölütleme motorları kullanılır. "
                "Hareketli konveyör bantları üzerinde parçaların takibi için Kalman Filtresi ile durum kestirimi ve DeepSORT ile kimlik koruma sağlanır."
            ),
            "kategori": "Görüntü İşleme"
        },
        {
            "id": "KB-002",
            "baslik": "Kurumsal RAG ve Bilgi Erişimi Mimarisi",
            "icerik": (
                "RAG mimarisi büyük dil modellerinin halüsinasyon görmesini engelleyen ve harici doküman bağlamını sisteme aktaran yöntemdir. "
                "Metinler kayan pencere yöntemiyle belirli çakışma payı bırakılarak parçalara ayrılır. "
                "Vektör veritabanlarında kosinüs benzerliği eşleşmesiyle en alakalı bilgi parçaları seçilip sistem promptuna enjekte edilir."
            ),
            "kategori": "Doğal Dil İşleme"
        },
        {
            "id": "KB-003",
            "baslik": "MLOps ve Model Servisleme Standartları",
            "icerik": (
                "Üretim ortamına alınan modeller FastAPI üzerinden asenkron REST API uç noktalarıyla servis edilir. "
                "Model gecikmesini düşürmek ve bellek kullanımını azaltmak için INT8 kuantizasyonu ve ONNX Runtime derlemesi uygulanır. "
                "Veri kaymasını izlemek için Population Stability Index (PSI) ve Kolmogorov-Smirnov testleri kullanılır."
            ),
            "kategori": "MLOps"
        }
    ]

    asistan.toplu_dokuman_ekle(dokumanlar)
    print(f"[+] İndekslenen Doküman Sayısı : {asistan.dokuman_sayisi} adet")
    print(f"[+] Üretilen Toplam Chunk Sayısı: {asistan.parca_sayisi} adet (ChunkSize=35, Overlap=10)")

    print("\n" + "=" * 80)
    print(">>> ASAMA 2: Çoklu Soru-Cevap & Kaynak Atfı (Citation) Sorguları")
    print("=" * 80)

    sorular = [
        "RAG mimarisinde metin parçalama ve bağlam enjeksiyonu nasıl çalışır?",
        "Fabrika üretim bantlarında nesne takibi ve durum kestirimi nasıl yapılır?",
        "Kuantum fiziğinde süperiletken kubit manipülasyonu nasıl gerçekleştirilir?"  # Out of domain
    ]

    en_son_cikis = None
    for i, soru in enumerate(sorular, 1):
        print(f"\n[?] Soru {i}: '{soru}'")
        cikis = asistan.soru_sor(soru, top_k=2)
        en_son_cikis = cikis

        print(f"    - Yanıt Durumu : {cikis['durum']}")
        print(f"    - Güven Skoru  : {cikis['guven_skoru']:.4f}")
        print(f"    - Kaynaklar    : {cikis['kaynaklar']}")
        print(f"    - Sentez Yanıt : {cikis['yanit']}")

    print("\n" + "=" * 80)
    print(">>> ASAMA 3: 6 Panelli RAG Teşhis Panosunun Üretilmesi")
    print("=" * 80)
    cikis_resmi = RAGGorsellestirici.rag_paneli_ciz(
        rag_ciktisi=en_son_cikis,
        hedef_path="day-34-mini-rag-assistant/ciktilar/rag_analiz_paneli.png"
    )
    print(f"[+] 6 Panelli RAG Teşhis Panosu Kaydedildi: {os.path.abspath(cikis_resmi)}")
    print("=" * 80)
    print("DAY 34: MİNİ RAG ASİSTANI & DOKÜMAN SORU-CEVAP MOTORU BAŞARIYLA TAMAMLANDI!")
    print("=" * 80)


if __name__ == "__main__":
    main()
